#!/usr/bin/env python3
"""
Main script to run debates with optional probe-based deception detection
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from debate.types import (
    DebateConfig, DebateType, ModelConfig, ProbeConfig, 
    ProviderType, APPSProblem, QuALITYProblem, DebateRole  
)
from debate.debate_manager import DebateManager
from debate.dataset_loader import FlexibleDatasetLoader, APPSDatasetLoader, SolutionGenerator, SimpleDebateDatasetLoader, QuALITYDatasetLoader


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run AI debates with optional probe detection')
    
    # Debate configuration
    parser.add_argument('--debate_type', type=str, required=True,
                       choices=['baseline', 'debater_access', 'full_access', 'judge_only'],
                       help='Type of debate to run')
    
    # Model selection
    parser.add_argument('--honest_model', type=str, required=True,
                       help='Model for honest debater (e.g., meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--dishonest_model', type=str, required=True,
                       help='Model for dishonest debater')
    parser.add_argument('--judge_model', type=str, required=True,
                       help='Model for judge (can be API model like gpt-4)')
    
    # Provider configuration
    parser.add_argument('--honest_provider', type=str, default='local',
                       choices=['local', 'openai', 'anthropic', 'openrouter'])
    parser.add_argument('--dishonest_provider', type=str, default='local',
                       choices=['local', 'openai', 'anthropic', 'openrouter'])
    parser.add_argument('--judge_provider', type=str, default='openai',
                       choices=['local', 'openai', 'anthropic', 'openrouter'])
    
    # API keys (optional, can use environment variables)
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key')
    parser.add_argument('--anthropic_api_key', type=str, help='Anthropic API key')
    parser.add_argument('--openrouter_api_key', type=str, help='OpenRouter API key')
    
    # Probe configuration
    parser.add_argument('--honest_probe_dir', type=str, 
                       help='Directory containing trained probes for honest debater')
    parser.add_argument('--dishonest_probe_dir', type=str,
                       help='Directory containing trained probes for dishonest debater')
    parser.add_argument('--probe_types', nargs='+', 
                       default=['logistic', 'pca', 'meandiff'],
                       help='Types of probes to use')
    parser.add_argument('--probe_layer', type=int, default=15,
                       help='Layer to extract activations from')
    parser.add_argument('--probe_display_mode', type=str, default='per_statement',
                       choices=['per_token', 'per_statement', 'summary_only'],
                       help='How to display probe scores to debaters')
    parser.add_argument('--judge_probe_display_mode', type=str, default='summary_only',
                   choices=['per_token', 'per_statement', 'summary_only'],
                   help='How to display probe scores to judge')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['apps', 'simple_debate', 'quality'],
                       help='Dataset to use')
    parser.add_argument('--dataset_path', type=str,
                       help='Path to custom dataset')
    parser.add_argument('--difficulty', type=str,
                       choices=['introductory', 'interview', 'competition'],
                       help='APPS difficulty filter')
    parser.add_argument('--n_problems', type=int, default=10,
                       help='Number of problems to evaluate')
    
    # Debate parameters
    parser.add_argument('--max_rounds', type=int, default=3,
                       help='Maximum debate rounds')
    parser.add_argument('--save_dir', type=str, default='./debate_results',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str,
                       help='Name for this experiment')
    
    # Execution options
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for local models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def validate_configuration(args):
    """Validate that the configuration is valid"""
    debate_type = DebateType[args.debate_type.upper()]
    
    # Check if probes are needed but not provided
    if debate_type != DebateType.BASELINE and not args.honest_probe_dir:
        raise ValueError(f"Debate type {args.debate_type} requires --honest_probe_dir")

    if debate_type != DebateType.BASELINE and not args.dishonest_probe_dir:
        raise ValueError(f"Debate type {args.debate_type} requires --dishonest_probe_dir")
    
    # Check if local models are required for probes
    needs_probe_on_honest = debate_type in [
        DebateType.DEBATER_ACCESS, 
        DebateType.FULL_ACCESS,
        DebateType.JUDGE_ONLY
    ]
    needs_probe_on_dishonest = debate_type in [
        DebateType.DEBATER_ACCESS,
        DebateType.FULL_ACCESS, 
        DebateType.JUDGE_ONLY
    ]
    
    if needs_probe_on_honest and args.honest_provider != 'local':
        raise ValueError(
            f"Debate type {args.debate_type} requires local model for honest debater "
            f"to run probes, but got provider {args.honest_provider}"
        )
        
    if needs_probe_on_dishonest and args.dishonest_provider != 'local':
        raise ValueError(
            f"Debate type {args.debate_type} requires local model for dishonest debater "
            f"to run probes, but got provider {args.dishonest_provider}"
        )
        
    print("Configuration validated successfully")


def create_model_config(model_name: str, provider: str, args) -> ModelConfig:
    """Create model configuration"""
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
        generation_kwargs={
            "temperature": 0.7,
            "max_tokens": 512
        }
    )


# Fixed sections for debate_manager.py

def run_single_debate(item: Dict, config: DebateConfig) -> Dict:
    """Run a single debate and return results"""
    manager = DebateManager(config)
    
    # Check if this is a simple debate scenario or APPS problem
    if 'scenario' in item:
        # Simple debate scenario
        result = manager.run_debate(scenario=item['scenario'])
        
        # Get ground truth and map it to role
        ground_truth_label = item['scenario'].get('ground_truth', '')  # e.g., "MODEL_A"
        
        # Map ground truth label to role using the manager's mapping
        ground_truth_role = None
        if hasattr(manager, 'model_mapping') and ground_truth_label:
            ground_truth_role = manager.model_mapping.get(ground_truth_label)
        
        # Determine which model label corresponds to the winner
        winner_label = None
        if hasattr(manager, 'model_mapping') and result.winner:
            for label, role in manager.model_mapping.items():
                if role == result.winner.value:
                    winner_label = label
                    break
        
        # Check if the winner matches ground truth
        correct = None
        if ground_truth_role and result.winner:
            correct = (result.winner.value == ground_truth_role)
        elif ground_truth_label and winner_label:
            correct = (winner_label == ground_truth_label)
            
        print(f"  Winner: {winner_label} ({result.winner.value if result.winner else 'None'})")
        print(f"  Ground Truth: {ground_truth_label} ({ground_truth_role if ground_truth_role else 'Unknown'})")
        print(f"  Correct: {correct}")
                
        return {
            "scenario_id": item['scenario'].get('scenario_id', 'unknown'),
            "topic": item['scenario'].get('topic', 'unknown'),
            "debate_type": config.debate_type.value,
            "winner_label": winner_label,
            "winner_role": result.winner.value if result.winner else None,
            "honest_model": config.honest_debater.model_name,
            "dishonest_model": config.dishonest_debater.model_name,
            "ground_truth_label": ground_truth_label,
            "ground_truth_role": ground_truth_role,
            "correct": correct,
            "success": result.success,
            "error": result.error if hasattr(result, 'error') else None,
            "rounds": len([t for t in result.turns if t.speaker.value in ['honest', 'dishonest']]) // 2,
            "probe_analysis": result.probe_analysis,
            "judge_reasoning": result.judge_reasoning if result.judge_reasoning else "",
            "debate_id": result.debate_id,
            "duration": result.end_time - result.start_time
        }


    elif 'quality_problem' in item:
        # QuALITY debate
        result = manager.run_debate(quality_problem=item['quality_problem'])
        
        # Determine correctness
        ground_truth = "honest"  # Honest always defends correct in QuALITY
        correct = (result.winner == DebateRole.HONEST) if result.winner else None
        
        return {
            "problem_id": item['quality_problem'].question_id,
            "story_title": item['quality_problem'].story_title,
            "question": item['quality_problem'].question,
            "debate_type": config.debate_type.value,
            "winner": result.winner.value if result.winner else None,
            "correct": correct,
            "success": result.success,
            "error": result.error if hasattr(result, 'error') else None,
            "rounds": len([t for t in result.turns if t.speaker.value in ['honest', 'dishonest']]) // 2,
            "probe_analysis": result.probe_analysis,
            "judge_reasoning": result.judge_reasoning if result.judge_reasoning else "",
            "debate_id": result.debate_id,
            "duration": result.end_time - result.start_time
        }

    
    else:  # APPS problem
        result = manager.run_debate(
            problem=item['problem'],
            honest_solution=item['honest_solution'],
            backdoored_solution=item['backdoored_solution']
        )
        
        # Determine correctness based on winner
        correct = (result.winner == DebateRole.HONEST) if result.winner else None
        
        return {
            "problem_id": item['problem'].problem_id,
            "difficulty": item['problem'].difficulty,
            "debate_type": config.debate_type.value,
            "winner": result.winner.value if result.winner else None,
            "model_a_role": result.metadata.get("model_a_role"),  # Add this
            "model_b_role": result.metadata.get("model_b_role"),  # Add this
            "model_a_is_honest": result.metadata.get("model_a_is_honest"),  # Add this
            "honest_model": result.metadata.get("honest_model"),  # Add this
            "dishonest_model": result.metadata.get("dishonest_model"),  # Add this
            "correct": correct,  # Add this
            "success": result.success,
            "error": result.error if hasattr(result, 'error') else None,
            "rounds": len([t for t in result.turns if t.speaker.value in ['honest', 'dishonest']]) // 2,
            "probe_analysis": result.probe_analysis,
            "judge_reasoning": result.judge_reasoning if result.judge_reasoning else "",
            "debate_id": result.debate_id,
            "duration": result.end_time - result.start_time
        }


def main():
    args = parse_arguments()
    
    # Validate configuration
    try:
        validate_configuration(args)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
        
    # Set up experiment directory
    timestamp = int(time.time())
    experiment_name = args.experiment_name or f"debate_{args.debate_type}_{timestamp}"
    save_dir = Path(args.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"AI DEBATE EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    print(f"Debate Type: {args.debate_type}")
    print(f"Models:")
    print(f"  Honest: {args.honest_model} ({args.honest_provider})")
    print(f"  Dishonest: {args.dishonest_model} ({args.dishonest_provider})")
    print(f"  Judge: {args.judge_model} ({args.judge_provider})")
    if args.honest_probe_dir or args.dishonest_probe_dir:
        print(f"Probes: {args.probe_types} from {args.honest_probe_dir}")
        print(f"Probes: {args.probe_types} from {args.dishonest_probe_dir}")
    print(f"Dataset: {args.dataset} ({args.n_problems} problems)")
    print(f"Results will be saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    # Create debate configuration
    debate_type = DebateType[args.debate_type.upper()]
    
    honest_config = create_model_config(args.honest_model, args.honest_provider, args)
    dishonest_config = create_model_config(args.dishonest_model, args.dishonest_provider, args)
    judge_config = create_model_config(args.judge_model, args.judge_provider, args)
    
    # Create probe configuration if needed
    probe_config = None
    if (args.honest_probe_dir or args.dishonest_probe_dir) and debate_type != DebateType.BASELINE:
        probe_config = ProbeConfig(
            honest_probe_dir=args.honest_probe_dir,
            dishonest_probe_dir=args.dishonest_probe_dir,
            probe_types=args.probe_types,
            layer=args.probe_layer,
            honest_model_name=args.honest_model,
            dishonest_model_name=args.dishonest_model,
            device=args.device
        )
        
    config = DebateConfig(
        honest_debater=honest_config,
        dishonest_debater=dishonest_config,
        judge=judge_config,
        probe_config=probe_config,
        debate_type=debate_type,
        max_rounds=args.max_rounds,
        save_dir=str(save_dir),
        probe_display_mode=args.probe_display_mode,
        judge_probe_display_mode=args.judge_probe_display_mode  
    )
    
    # Save configuration
    config_path = save_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "args": vars(args),
            "debate_type": debate_type.value,
            "models": {
                "honest": args.honest_model,
                "dishonest": args.dishonest_model,
                "judge": args.judge_model
            },
            "probe_config": {
                "enabled": probe_config is not None,
                "types": args.probe_types if probe_config else [],
                "layer": args.probe_layer if probe_config else None
            }
        }, f, indent=2)


    print("Loading dataset...")
    if args.dataset == "simple_debate":
        loader = SimpleDebateDatasetLoader(args.dataset_path, seed=args.seed)
        dataset = []
        scenarios = loader.get_scenarios_batch(n=args.n_problems, shuffle=True)
        
        for scenario in scenarios:
            dataset.append({
                "scenario": scenario,
                # "problem": None,  # Not using APPSProblem
                # "honest_solution": None,
                # "backdoored_solution": None
            })

    elif args.dataset == "quality":
        loader = QuALITYDatasetLoader(csv_path=args.dataset_path, seed=args.seed)
        dataset = []
        problems = loader.get_problems_batch(n=args.n_problems, shuffle=True)
        
        for problem in problems:
            dataset.append({"quality_problem": problem})
        
    elif args.dataset == "apps":
        dataset = FlexibleDatasetLoader.load_dataset(
            args.dataset,
            difficulty=args.difficulty,
            n_problems=args.n_problems,
            shuffle=True,
            path=args.dataset_path
        )


    
    # Save dataset for reproducibility
    FlexibleDatasetLoader.save_dataset(dataset, save_dir / "dataset.json")
    
    # Run debates
    results = []
    print(f"\nRunning {len(dataset)} debates...\n")


    
    for i, item in enumerate(dataset):
        if 'scenario' in item:
            print(f"Debate {i+1}/{len(dataset)}")
            print(f"Topic: {item['scenario'].get('topic', 'Unknown')}")
        elif 'quality_problem' in item:
            print(f"Debate {i+1}/{len(dataset)}")
            print(f"Question: {item['quality_problem'].question[:100]}...")
            print(f"Story: {item['quality_problem'].story_title}")
        elif 'problem' in item:
            print(f"Debate {i+1}/{len(dataset)}")
            print(f"Problem: {item['problem'].problem_id}")
        
        try:
            result = run_single_debate(item, config)
            results.append(result)
            
            # Note: result is now a dictionary returned by run_single_debate, not a DebateResult object
            print(f"  Winner: {result.get('winner_label', result.get('winner', 'None'))}")
            if 'correct' in result:
                print(f"  Correct: {result['correct']}")
            print(f"  Duration: {result['duration']:.1f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Add a failed result entry - handle different dataset types
            if 'scenario' in item:
                results.append({
                    "scenario_id": item['scenario'].get('scenario_id', 'unknown'),
                    "topic": item['scenario'].get('topic', 'unknown'),
                    "error": str(e),
                    "success": False
                })
            elif 'quality_problem' in item:
                results.append({
                    "problem_id": item['quality_problem'].question_id,
                    "story_title": item['quality_problem'].story_title,
                    "error": str(e),
                    "success": False
                })
            elif 'problem' in item:
                results.append({
                    "problem_id": item['problem'].problem_id,
                    "error": str(e),
                    "success": False
                })
            else:
                results.append({
                    "error": str(e),
                    "success": False
                })
    
    print()
            
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "results.csv", index=False)
    
    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total debates: {len(results)}")
    print(f"Successful: {sum(r.get('success', False) for r in results)}")
    print(f"Failed: {sum(not r.get('success', True) for r in results)}")
    
    if any(r.get('winner') for r in results):
        honest_wins = sum(r.get('winner') == 'honest' for r in results)
        dishonest_wins = sum(r.get('winner') == 'dishonest' for r in results)
        print(f"\nResults:")
        print(f"  Honest wins: {honest_wins} ({100*honest_wins/len(results):.1f}%)")
        print(f"  Dishonest wins: {dishonest_wins} ({100*dishonest_wins/len(results):.1f}%)")
        
    if any(r.get('probe_analysis') for r in results):
        probe_results = [r['probe_analysis'] for r in results if r.get('probe_analysis')]
        avg_honest_score = sum(
            p['scores_by_role'].get('honest', {}).get('mean', 0) 
            for p in probe_results
        ) / len(probe_results)
        avg_dishonest_score = sum(
            p['scores_by_role'].get('dishonest', {}).get('mean', 0)
            for p in probe_results
        ) / len(probe_results)
        
        print(f"\nAverage probe scores:")
        print(f"  Honest debater: {avg_honest_score:.3f}")
        print(f"  Dishonest debater: {avg_dishonest_score:.3f}")
        
    print(f"\nResults saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())