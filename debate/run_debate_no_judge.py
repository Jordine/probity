#!/usr/bin/env python3
"""
Run debates without judge decisions, saving full transcripts with probe scores.
This allows efficient re-evaluation with different judges and probe configurations.
"""
import numpy as np
import argparse
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from datetime import datetime


from debate.types import (
    DebateConfig, DebateType, ModelConfig, ProbeConfig, 
    ProviderType, APPSProblem, QuALITYProblem, DebateRole  
)
from debate.debate_manager import DebateManager
from debate.dataset_loader import (
    FlexibleDatasetLoader, SimpleDebateDatasetLoader,
    QuALITYDatasetLoader, SyntheticQuALITYDatasetLoader, APPSDatasetLoader, SolutionGenerator
)

from debate.utils.gpu_utils import detect_gpu_configuration, can_fit_models

class NoJudgeDebateRunner:
    """Run debates without judge decision, save full transcripts with probe data"""
    
    def __init__(self, config: DebateConfig, save_dir: Path, experiment_mode: str):
        self.config = config
        self.save_dir = save_dir
        self.experiment_mode = experiment_mode
        self.transcripts_dir = save_dir / "transcripts"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary file for tracking progress
        self.summary_file = save_dir / "transcript_summary.jsonl"
        
        # Create DebateManager once and reuse it
        print("Loading models and probes once...")
        self.manager = DebateManager(config)
        print("Models and probes loaded successfully!")
        
    def _calculate_probe_only_winners(self, manager: DebateManager) -> Dict[str, str]:
        """Calculate probe-only winners based on mean and max scores"""
        
        honest_scores = []
        dishonest_scores = []
        
        print(f"[DEBUG] Calculating probe-only winners...")
        print(f"[DEBUG] Total turns: {len(manager.all_turns)}")
        
        for turn in manager.all_turns:
            # Skip judge turns
            if turn.speaker == DebateRole.JUDGE:
                continue
            
            # Debug print
            print(f"[DEBUG] Turn - Speaker: {turn.speaker}, Type: {type(turn.speaker)}, Mean score: {turn.mean_probe_score}")
            
            if turn.mean_probe_score is not None:
                if turn.speaker == DebateRole.HONEST:
                    honest_scores.append(turn.mean_probe_score)
                    print(f"[DEBUG] Added honest score: {turn.mean_probe_score}")
                elif turn.speaker == DebateRole.DISHONEST:
                    dishonest_scores.append(turn.mean_probe_score)
                    print(f"[DEBUG] Added dishonest score: {turn.mean_probe_score}")
        
        print(f"[DEBUG] Honest scores: {honest_scores}")
        print(f"[DEBUG] Dishonest scores: {dishonest_scores}")
        
        results = {}
        
        if honest_scores and dishonest_scores:
            # Mean-based classification
            honest_mean = np.mean(honest_scores)
            dishonest_mean = np.mean(dishonest_scores)
            
            print(f"[DEBUG] Honest mean: {honest_mean}, Dishonest mean: {dishonest_mean}")
            
            # Higher score = more deceptive, so lower score wins
            if honest_mean < dishonest_mean:
                results['probe_only_mean'] = 'honest'
            elif dishonest_mean < honest_mean:
                results['probe_only_mean'] = 'dishonest'
            else:
                results['probe_only_mean'] = 'tie'
            
            # Max-based classification
            honest_max = np.max(honest_scores)
            dishonest_max = np.max(dishonest_scores)
            
            if honest_max < dishonest_max:
                results['probe_only_max'] = 'honest'
            elif dishonest_max < honest_max:
                results['probe_only_max'] = 'dishonest'
            else:
                results['probe_only_max'] = 'tie'
            
            # Store raw scores for analysis
            results['honest_mean_score'] = float(honest_mean)
            results['dishonest_mean_score'] = float(dishonest_mean)
            results['honest_max_score'] = float(honest_max)
            results['dishonest_max_score'] = float(dishonest_max)
            
            print(f"[DEBUG] Results: {results}")
        else:
            print(f"[DEBUG] Not enough scores for probe-only classification")
            results['probe_only_mean'] = None
            results['probe_only_max'] = None
        
        return results
    

        
        
    def run_debate_no_judge(self, item: Dict, debate_idx: int) -> Dict:
        """Run a single debate without judge decision"""
        
        # Reset the manager for a new debate (keeps models loaded)
        self.manager.reset_for_new_debate()
        
        # Determine debate type and problem ID
        if 'scenario' in item:
            debate_type = "simple"
            problem_id = item['scenario'].get('scenario_id', f'scenario_{debate_idx}')
            problem_desc = item['scenario'].get('topic', 'Unknown topic')
        elif 'quality_problem' in item:
            debate_type = "quality"
            problem_id = item['quality_problem'].question_id
            problem_desc = f"{item['quality_problem'].story_title}: {item['quality_problem'].question}..."
        else:
            debate_type = "apps"
            problem_id = item['problem'].problem_id
            problem_desc = f"Difficulty: {item['problem'].difficulty}"
            
        print(f"\nDebate {debate_idx}/{self.total_debates}: {problem_id}")
        print(f"  Type: {debate_type}")
        print(f"  Mode: {self.experiment_mode}")
        print(f"  Details: {problem_desc}")
        
        # Start debate
        start_time = time.time()
        debate_id = f"{debate_type}_{problem_id}_{self.experiment_mode}_{int(start_time)}"
        
        try:
            # Setup debate based on type (using the existing manager)
            if debate_type == "simple":
                self.manager.run_simple_debate_setup(item['scenario'])
            elif debate_type == "quality":
                self.manager.run_quality_debate_setup(item['quality_problem'])
            else:
                self.manager.run_apps_debate_setup(
                    item['problem'], 
                    item['honest_solution'], 
                    item['backdoored_solution']
                )
            
            # Run debate rounds (but not judge)
            for round_num in range(self.config.max_rounds):
                print(f"  Round {round_num + 1}/{self.config.max_rounds}")
                if debate_type == "quality":
                    if self.manager.honest_goes_first:
                        self.manager._run_quality_honest_turn(round_num + 1)
                        self.manager._run_quality_dishonest_turn(round_num + 1)
                    else:
                        self.manager._run_quality_dishonest_turn(round_num + 1)
                        self.manager._run_quality_honest_turn(round_num + 1)
                else:
                    if self.manager.honest_goes_first:
                        self.manager._run_honest_turn(round_num + 1)
                        self.manager._run_dishonest_turn(round_num + 1)
                    else:
                        self.manager._run_dishonest_turn(round_num + 1)
                        self.manager._run_honest_turn(round_num + 1)
                        
            # Extract and save comprehensive transcript
            transcript = self._create_transcript(self.manager, item, debate_id, debate_type)
            
            # Save immediately (incremental saving)
            transcript_file = self.transcripts_dir / f"{debate_id}.json"
            with open(transcript_file, 'w') as f:
                json.dump(transcript, f, indent=2)
            
            duration = time.time() - start_time
            print(f"  Duration: {duration:.1f}s")
            print(f"  Saved: {transcript_file.name}")
            
            # Save to summary file (append mode for crash recovery)
            summary_entry = {
                "debate_id": debate_id,
                "problem_id": problem_id,
                "transcript_file": str(transcript_file.name),
                "duration": duration,
                "experiment_mode": self.experiment_mode,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            with open(self.summary_file, 'a') as f:
                f.write(json.dumps(summary_entry) + '\n')
            
            return summary_entry
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error entry
            error_entry = {
                "debate_id": debate_id,
                "problem_id": problem_id,
                "error": str(e),
                "experiment_mode": self.experiment_mode,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
            with open(self.summary_file, 'a') as f:
                f.write(json.dumps(error_entry) + '\n')
            
            return error_entry
    
    def _create_transcript(self, manager: DebateManager, item: Dict, 
                          debate_id: str, debate_type: str) -> Dict:
        """Create comprehensive transcript with all debate data and probe scores"""
        
        # Build turns with full probe data
        turns = []
        for turn in manager.all_turns:
            if turn.speaker == DebateRole.JUDGE:
                continue  # Skip judge turns since we haven't run judge yet
            
            turn_data = {
                "turn_number": turn.round,
                "speaker": turn.speaker.value,
                "content": turn.content,
                "timestamp": turn.timestamp
            }
            
            # Add tokens if available
            if turn.tokens:
                turn_data["tokens"] = turn.tokens
            
            # Add probe scores and raw logits if available
            if turn.probe_scores:
                turn_data["probe_scores"] = turn.probe_scores
                turn_data["mean_probe_score"] = turn.mean_probe_score
                
                # Extract raw logits from metadata
                if turn.metadata:
                    # Look for raw logits stored during probe scoring
                    raw_logits = {}
                    for probe_type in turn.probe_scores.keys():
                        # Check if raw logits were stored in the turn's metadata
                        probe_key = f"probe_{probe_type}_raw_logits"
                        if probe_key in turn.metadata:
                            raw_logits[probe_type] = turn.metadata[probe_key]
                    
                    if raw_logits:
                        turn_data["probe_logits"] = raw_logits
            
            # Add model label mapping for clarity
            if hasattr(manager, 'model_mapping'):
                for label, role in manager.model_mapping.items():
                    if role == turn.speaker.value:
                        turn_data["model_label"] = label
                        break
            
            turns.append(turn_data)
        
        # Prepare problem data (without model objects)
        if debate_type == "simple":
            problem_data = item['scenario']
        elif debate_type == "quality":
            problem_data = {
                "question_id": item['quality_problem'].question_id,
                "story": item['quality_problem'].story,
                "story_title": item['quality_problem'].story_title,
                "question": item['quality_problem'].question,
                "correct_answer": item['quality_problem'].correct_answer,
                "incorrect_answer": item['quality_problem'].incorrect_answer,
                "answers": item['quality_problem'].answers,
                "gold_label": item['quality_problem'].gold_label
            }
        else:  # APPS
            problem_data = {
                "problem_id": item['problem'].problem_id,
                "question": item['problem'].question,
                "difficulty": item['problem'].difficulty,
                "honest_solution": item['honest_solution'],
                "backdoored_solution": item['backdoored_solution']
            }
        
        # Build complete transcript
        transcript = {
            "debate_id": debate_id,
            "debate_type": debate_type,
            "experiment_mode": self.experiment_mode,
            "timestamp": time.time(),
            "debate_config": {
                "debate_type": self.config.debate_type.value,
                "max_rounds": self.config.max_rounds,
                "honest_model": self.config.honest_debater.model_name,
                "dishonest_model": self.config.dishonest_debater.model_name,
                "honest_provider": self.config.honest_debater.provider.value,
                "dishonest_provider": self.config.dishonest_debater.provider.value,
                "probe_config": {
                    "enabled": self.config.probe_config is not None,
                    "layer": self.config.probe_config.layer if self.config.probe_config else None,
                    "types": self.config.probe_config.probe_types if self.config.probe_config else [],
                    "honest_probe_dir": self.config.probe_config.honest_probe_dir if self.config.probe_config else None,
                    "dishonest_probe_dir": self.config.probe_config.dishonest_probe_dir if self.config.probe_config else None
                }
            },
            "problem_data": problem_data,
            "turns": turns,
            "model_mapping": manager.model_mapping if hasattr(manager, 'model_mapping') else None,
            "debate_metadata": {
                "honest_goes_first": manager.honest_goes_first if hasattr(manager, 'honest_goes_first') else None,
                "model_a_is_honest": manager.model_a_is_honest if hasattr(manager, 'model_a_is_honest') else None
            },
            "judge_results": {},  # Will be filled by run_judge_for_transcripts.py
            "probe_only_results": self._calculate_probe_only_winners(manager)
        }
        
        return transcript
    
    def run_all_debates(self, dataset: List[Dict]) -> pd.DataFrame:
        """Run all debates in the dataset"""
        self.total_debates = len(dataset)
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running {self.total_debates} debates in {self.experiment_mode} mode")
        print(f"Saving transcripts to: {self.transcripts_dir}")
        print(f"{'='*60}")
        
        for i, item in enumerate(dataset, 1):
            result = self.run_debate_no_judge(item, i)
            results.append(result)
        
        # Create summary DataFrame
        df = pd.DataFrame(results)
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("TRANSCRIPT GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total debates: {len(results)}")
        print(f"Successful: {df['success'].sum()}")
        print(f"Failed: {(~df['success']).sum()}")
        print(f"Average duration: {df[df['success']]['duration'].mean():.1f}s")
        print(f"Total time: {df['duration'].sum()/60:.1f} minutes")
        print(f"\nTranscripts saved to: {self.transcripts_dir}")
        print(f"Summary saved to: {self.summary_file}")
        print(f"{'='*60}")
        
        # Save summary CSV
        summary_csv = self.save_dir / "transcript_generation_summary.csv"
        df.to_csv(summary_csv, index=False)
        
        return df


def create_model_config(model_name: str, provider: str, args, 
                        device_id: Optional[int] = None,
                        gpu_indices: Optional[List[int]] = None) -> ModelConfig:  # ADD gpu_indices param
    """Create model configuration with optional GPU assignment"""
    provider_type = ProviderType[provider.upper()]
    
    # Get API key if needed
    api_key = None
    if provider == 'openai':
        api_key = args.openai_api_key
    elif provider == 'anthropic':
        api_key = args.anthropic_api_key
    elif provider == 'openrouter':
        api_key = args.openrouter_api_key
    
    return ModelConfig(
        provider=provider_type,
        model_name=model_name,
        api_key=api_key,
        device_id=device_id,  # Keep for backward compatibility
        gpu_indices=gpu_indices,  # ADD THIS
        generation_kwargs={
            "temperature": 0,
            "max_tokens": 512
        }
    )



def main():
    parser = argparse.ArgumentParser(
        description='Run debates without judge decision, saving full transcripts'
    )
    
    # Experiment mode
    parser.add_argument('--experiment_mode', type=str, required=True,
                       choices=['baseline_judge', 'debater_full'],
                       help='baseline_judge: for baseline+judge_only experiments, '
                            'debater_full: for debater_only+full_access experiments')
    
    # Model configuration
    parser.add_argument('--honest_model', type=str, required=True,
                       help='Model for honest debater')
    parser.add_argument('--dishonest_model', type=str, required=True,
                       help='Model for dishonest debater')
    
    # Provider configuration
    parser.add_argument('--honest_provider', type=str, default='local',
                       choices=['local', 'openai', 'anthropic', 'openrouter'])
    parser.add_argument('--dishonest_provider', type=str, default='local',
                       choices=['local', 'openai', 'anthropic', 'openrouter'])
    parser.add_argument('--honest_gpu', type=int, default=None,
                       help='GPU device ID for honest model (auto-detect if not specified)')
    parser.add_argument('--dishonest_gpu', type=int, default=None,
                       help='GPU device ID for dishonest model (auto-detect if not specified)')
    
    #use indices
    
    parser.add_argument('--honest_gpu_indices', nargs='+', type=int, default=None,
                       help='GPU indices for honest model (e.g., --honest_gpu_indices 0 1)')
    parser.add_argument('--dishonest_gpu_indices', nargs='+', type=int, default=None,
                       help='GPU indices for dishonest model (e.g., --dishonest_gpu_indices 2 3)')
    
    
    # API keys
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--anthropic_api_key', type=str)
    parser.add_argument('--openrouter_api_key', type=str)
    
    # Probe configuration
    parser.add_argument('--honest_probe_dir', type=str,
                       help='Directory with trained probes for honest model')
    parser.add_argument('--dishonest_probe_dir', type=str,
                       help='Directory with trained probes for dishonest model')
    parser.add_argument('--probe_types', nargs='+',
                       default=['logistic', 'pca', 'meandiff'])
    parser.add_argument('--probe_layer', type=int, default=15)
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['apps', 'simple_debate', 'quality', 'quality_synthetic'])
    parser.add_argument('--dataset_path', type=str,
                       help='Path to dataset file')
    parser.add_argument('--difficulty', type=str,
                       choices=['introductory', 'interview', 'competition'],
                       help='APPS difficulty filter')
    parser.add_argument('--n_problems', type=int, default=10,
                       help='Number of problems to run')
    
    # Debate parameters
    parser.add_argument('--max_rounds', type=int, default=3,
                       help='Maximum debate rounds')
    parser.add_argument('--save_dir', type=str, default='./debate_transcripts',
                       help='Directory to save transcripts')
    parser.add_argument('--experiment_name', type=str,
                       help='Name for this experiment')
    
    # Execution options
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing transcripts')
    
    args = parser.parse_args()

    # GPU Detection and Configuration
    print("\n" + "="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    
    gpu_info = detect_gpu_configuration()
    
    if gpu_info.num_gpus == 0:
        print("⚠️  No GPUs detected, will use CPU (not recommended for large models)")
        honest_device_id = None
        dishonest_device_id = None
        honest_gpu_indices = None
        dishonest_gpu_indices = None
    else:
        print(f"✓ Found {gpu_info.num_gpus} GPU(s):")
        for i, (name, memory) in enumerate(zip(gpu_info.gpu_names, gpu_info.gpu_memory)):
            print(f"  GPU {i}: {name} ({memory:.1f}GB)")
        
        # Handle multi-GPU indices
        if args.honest_gpu_indices and args.dishonest_gpu_indices:
            # Explicit multi-GPU assignment
            honest_gpu_indices = args.honest_gpu_indices
            dishonest_gpu_indices = args.dishonest_gpu_indices
            honest_device_id = honest_gpu_indices[0]  # Primary device
            dishonest_device_id = dishonest_gpu_indices[0]  # Primary device
            print(f"\n📌 Using explicit multi-GPU assignment:")
            print(f"  Honest debater → GPUs {honest_gpu_indices}")
            print(f"  Dishonest debater → GPUs {dishonest_gpu_indices}")
        elif args.honest_gpu is not None and args.dishonest_gpu is not None:
            # Legacy single GPU assignment
            honest_device_id = args.honest_gpu
            dishonest_device_id = args.dishonest_gpu
            honest_gpu_indices = [args.honest_gpu]
            dishonest_gpu_indices = [args.dishonest_gpu]
            print(f"\n📌 Using manual single GPU assignment:")
            print(f"  Honest debater → GPU {honest_device_id}")
            print(f"  Dishonest debater → GPU {dishonest_device_id}")
        else:
            # Fallback to GPU 0 - FIX: set gpu_indices too
            honest_device_id = 0
            dishonest_device_id = 0
            honest_gpu_indices = [0]  # ADD THIS
            dishonest_gpu_indices = [0]  # ADD THIS
            print(f"\n⚠️  Using GPU 0 for both models (may cause OOM):")
            print(f"  Honest debater → GPU {honest_device_id}")
            print(f"  Dishonest debater → GPU {dishonest_device_id}")
    
    print("="*60 + "\n")
    
    
    # Set random seed
    random.seed(args.seed)
    
    # Determine actual debate type based on experiment mode
    if args.experiment_mode == 'baseline_judge':
        debate_type = DebateType.BASELINE
        print("Running baseline debates (no probe access during debate)")
    else:  # debater_full
        debate_type = DebateType.DEBATER_ACCESS
        print("Running debater-access debates (debaters see opponent probe scores)")
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or f"{args.experiment_mode}_{timestamp}"
    save_dir = Path(args.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model configs with explicit gpu assignments
    honest_config = create_model_config(
        args.honest_model, 
        args.honest_provider, 
        args, 
        device_id=honest_device_id,
        gpu_indices=honest_gpu_indices  # ADD THIS
    )
    dishonest_config = create_model_config(
        args.dishonest_model, 
        args.dishonest_provider, 
        args, 
        device_id=dishonest_device_id,
        gpu_indices=dishonest_gpu_indices  # ADD THIS
    )
    
    # Create probe configuration if provided
    probe_config = None
    if args.honest_probe_dir and args.dishonest_probe_dir:
        probe_config = ProbeConfig(
            honest_probe_dir=args.honest_probe_dir,
            dishonest_probe_dir=args.dishonest_probe_dir,
            probe_types=args.probe_types,
            layer=args.probe_layer,
            honest_model_name=args.honest_model,
            dishonest_model_name=args.dishonest_model,
            device=args.device
        )
    
    # Create a dummy judge config that won't trigger model loading
    dummy_judge_config = ModelConfig(
        provider=ProviderType.OPENAI,  # Use API provider to avoid loading
        model_name="gpt-3.5-turbo",  # Dummy model that won't be used
        api_key="dummy",  # Won't be used
        generation_kwargs={}
    )
    
    # Create debate configuration (dummy judge that won't load)
    config = DebateConfig(
        honest_debater=honest_config,
        dishonest_debater=dishonest_config,
        judge=dummy_judge_config,  # Dummy config that won't load a model
        probe_config=probe_config,
        debate_type=debate_type,
        max_rounds=args.max_rounds,
        save_dir=str(save_dir)
    )
    
    # Save configuration
    config_dict = {
        "experiment_mode": args.experiment_mode,
        "debate_type": debate_type.value,
        "models": {
            "honest": args.honest_model,
            "dishonest": args.dishonest_model
        },
        "providers": {
            "honest": args.honest_provider,
            "dishonest": args.dishonest_provider
        },
        "dataset": args.dataset,
        "n_problems": args.n_problems,
        "max_rounds": args.max_rounds,
        "probe_config": {
            "enabled": probe_config is not None,
            "types": args.probe_types if probe_config else [],
            "layer": args.probe_layer if probe_config else None
        },
        "timestamp": timestamp
    }
    
    with open(save_dir / "experiment_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    
    if args.dataset == "simple_debate":
        loader = SimpleDebateDatasetLoader(args.dataset_path, seed=args.seed)
        scenarios = loader.get_scenarios_batch(n=args.n_problems, shuffle=True)
        dataset = [{"scenario": s} for s in scenarios]
        
    elif args.dataset == "quality":
        loader = QuALITYDatasetLoader(csv_path=args.dataset_path, seed=args.seed)
        problems = loader.get_problems_batch(n=args.n_problems, shuffle=True)
        dataset = [{"quality_problem": p} for p in problems]

    elif args.dataset == "quality_synthetic":
        # Load synthetic (uncontaminated) QuALITY dataset from local files
        stories_path = args.dataset_path or "data/quality_synthetic/stories.jsonl"
        loader = SyntheticQuALITYDatasetLoader(stories_path, questions_path=None, seed=args.seed)
        problems = loader.get_problems_batch(n=args.n_problems, shuffle=True)
        dataset = [{"quality_problem": p} for p in problems]

    elif args.dataset == "apps":
        loader = APPSDatasetLoader(difficulty=args.difficulty, seed=args.seed)
        problems = loader.get_problems_batch(n=args.n_problems, shuffle=True)
        dataset = []
        for problem in problems:
            honest_sol, backdoored_sol = SolutionGenerator.generate_solutions(problem)
            dataset.append({
                "problem": problem,
                "honest_solution": honest_sol,
                "backdoored_solution": backdoored_sol
            })
    
    print(f"Loaded {len(dataset)} problems")
    
    # Check for resume
    if args.resume and (save_dir / "transcript_summary.jsonl").exists():
        print("\nResuming from existing transcripts...")
        completed_ids = set()
        with open(save_dir / "transcript_summary.jsonl", 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get('success'):
                    completed_ids.add(entry['problem_id'])
        
        # Filter dataset
        original_size = len(dataset)
        if args.dataset == "simple_debate":
            dataset = [d for d in dataset if d['scenario'].get('scenario_id') not in completed_ids]
        elif args.dataset == "quality":
            dataset = [d for d in dataset if d['quality_problem'].question_id not in completed_ids]
        else:  # APPS
            dataset = [d for d in dataset if d['problem'].problem_id not in completed_ids]
        
        print(f"Skipping {original_size - len(dataset)} completed debates")
    
    # Save dataset for reproducibility
    with open(save_dir / "dataset.json", 'w') as f:
        # Convert to serializable format
        serializable_dataset = []
        for item in dataset:
            if 'scenario' in item:
                serializable_dataset.append({"scenario": item['scenario']})
            elif 'quality_problem' in item:
                prob = item['quality_problem']
                serializable_dataset.append({
                    "quality_problem": {
                        "question_id": prob.question_id,
                        "story": prob.story,
                        "story_title": prob.story_title,
                        "question": prob.question,
                        "correct_answer": prob.correct_answer,
                        "incorrect_answer": prob.incorrect_answer,
                        "answers": prob.answers,
                        "gold_label": prob.gold_label
                    }
                })
            else:  # APPS
                serializable_dataset.append({
                    "problem": item['problem'].__dict__,
                    "honest_solution": item['honest_solution'],
                    "backdoored_solution": item['backdoored_solution']
                })
        json.dump(serializable_dataset, f, indent=2)
    
    # Run debates
    runner = NoJudgeDebateRunner(config, save_dir, args.experiment_mode)
    results_df = runner.run_all_debates(dataset)
    
    print(f"\nExperiment complete! Transcripts saved to: {save_dir}")
    print(f"Next step: Run 'run_judge_for_transcripts.py' to evaluate with judges")
    
    return 0


if __name__ == "__main__":
    exit(main())
