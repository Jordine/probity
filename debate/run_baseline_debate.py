#!/usr/bin/env python3
"""
Simplified baseline debate runner for rapid iteration.
- Uses OpenRouter API (no local GPU needed)
- No probe infrastructure
- Runs debates + optional judging in one pass
- Designed for testing dishonest debater prompting and story quality
"""
import argparse
import json
import time
import random
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from debate.types import (
    DebateConfig, DebateType, ModelConfig, ProbeConfig,
    ProviderType, QuALITYProblem, DebateRole
)
from debate.debate_manager import DebateManager
from debate.dataset_loader import SyntheticQuALITYDatasetLoader


class BaselineDebateRunner:
    """Simplified debate runner for API-based models"""

    def __init__(self, config: DebateConfig, save_dir: Path, dishonesty_level: str = "explicit_v2"):
        self.config = config
        self.save_dir = save_dir
        self.dishonesty_level = dishonesty_level
        self.transcripts_dir = save_dir / "transcripts"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        print("Initializing debate manager...")
        self.manager = DebateManager(config)
        print("Ready!")

    def run_single_debate(self, problem: QuALITYProblem, debate_idx: int, total: int) -> Dict:
        """Run a single debate and return transcript"""

        self.manager.reset_for_new_debate()

        print(f"\n{'='*60}")
        print(f"Debate {debate_idx}/{total}: {problem.question_id}")
        print(f"Question: {problem.question[:80]}...")
        print(f"{'='*60}")

        start_time = time.time()
        debate_id = f"baseline_{problem.question_id}_{int(start_time)}"

        try:
            # Setup debate
            self.manager.run_quality_debate_setup(problem, dishonesty_level=self.dishonesty_level)

            # Run debate rounds
            for round_num in range(self.config.max_rounds):
                print(f"  Round {round_num + 1}/{self.config.max_rounds}")
                if self.manager.honest_goes_first:
                    self.manager._run_quality_honest_turn(round_num + 1)
                    self.manager._run_quality_dishonest_turn(round_num + 1)
                else:
                    self.manager._run_quality_dishonest_turn(round_num + 1)
                    self.manager._run_quality_honest_turn(round_num + 1)

            # Build transcript
            turns = []
            for turn in self.manager.all_turns:
                if turn.speaker == DebateRole.JUDGE:
                    continue
                turns.append({
                    "round": turn.round,
                    "speaker": turn.speaker.value,
                    "content": turn.content
                })

            transcript = {
                "debate_id": debate_id,
                "question_id": problem.question_id,
                "story_title": problem.story_title,
                "question": problem.question,
                "correct_answer": problem.correct_answer,
                "incorrect_answer": problem.incorrect_answer,
                "dishonesty_level": self.dishonesty_level,
                "model_a_is_honest": self.manager.model_a_is_honest,
                "honest_goes_first": self.manager.honest_goes_first,
                "turns": turns,
                "duration": time.time() - start_time
            }

            # Save transcript
            transcript_file = self.transcripts_dir / f"{debate_id}.json"
            with open(transcript_file, 'w') as f:
                json.dump(transcript, f, indent=2)

            print(f"  Duration: {transcript['duration']:.1f}s")
            print(f"  Saved: {transcript_file.name}")

            return {"success": True, "transcript": transcript, "file": str(transcript_file)}

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def run_all(self, problems: List[QuALITYProblem]) -> List[Dict]:
        """Run all debates"""
        results = []
        total = len(problems)

        for i, problem in enumerate(problems, 1):
            result = self.run_single_debate(problem, i, total)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.get("success"))
        print(f"\n{'='*60}")
        print(f"COMPLETE: {successful}/{total} debates successful")
        print(f"Transcripts: {self.transcripts_dir}")
        print(f"{'='*60}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Simplified baseline debate runner')

    # Model configuration
    parser.add_argument('--model', type=str, default='meta-llama/llama-3.3-70b-instruct',
                       help='Model name for both debaters (OpenRouter format)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')

    # Debate configuration
    parser.add_argument('--dishonesty_level', type=str, default='explicit_v2',
                       choices=['symmetric', 'persuasive', 'selective', 'explicit', 'explicit_v2'],
                       help='Dishonesty level for the dishonest debater')
    parser.add_argument('--max_rounds', type=int, default=3,
                       help='Number of debate rounds')

    # Dataset
    parser.add_argument('--dataset_path', type=str,
                       default='data/quality_synthetic/stories.jsonl',
                       help='Path to synthetic stories dataset')
    parser.add_argument('--n_problems', type=int, default=5,
                       help='Number of problems to run')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index in dataset (for resuming)')

    # Output
    parser.add_argument('--save_dir', type=str, default='./debate_baseline_test',
                       help='Directory to save transcripts')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment run')

    # Misc
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: No OpenRouter API key provided.")
        print("Set OPENROUTER_API_KEY env var or use --api_key")
        return 1

    random.seed(args.seed)

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or f"baseline_{args.dishonesty_level}_{timestamp}"
    save_dir = Path(args.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create model configs (same model for both debaters, using OpenRouter)
    model_config = ModelConfig(
        provider=ProviderType.OPENROUTER,
        model_name=args.model,
        api_key=api_key,
        generation_kwargs={
            "temperature": 0,
            "max_tokens": 512
        }
    )

    # Dummy judge config (won't be used in baseline mode)
    dummy_judge = ModelConfig(
        provider=ProviderType.OPENROUTER,
        model_name="openai/gpt-3.5-turbo",
        api_key=api_key,
        generation_kwargs={}
    )

    # Create debate config (no probes for baseline)
    config = DebateConfig(
        honest_debater=model_config,
        dishonest_debater=model_config,
        judge=dummy_judge,
        probe_config=None,  # No probes
        debate_type=DebateType.BASELINE,
        max_rounds=args.max_rounds,
        save_dir=str(save_dir),
        dishonesty_level=args.dishonesty_level
    )

    # Save experiment config
    config_dict = {
        "model": args.model,
        "dishonesty_level": args.dishonesty_level,
        "max_rounds": args.max_rounds,
        "n_problems": args.n_problems,
        "start_idx": args.start_idx,
        "seed": args.seed,
        "timestamp": timestamp
    }
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    loader = SyntheticQuALITYDatasetLoader(args.dataset_path, seed=args.seed)

    # Get problems with offset
    all_problems = loader.get_problems_batch(n=args.start_idx + args.n_problems, shuffle=False)
    problems = all_problems[args.start_idx:args.start_idx + args.n_problems]
    print(f"Loaded {len(problems)} problems (indices {args.start_idx} to {args.start_idx + len(problems) - 1})")

    # Run debates
    runner = BaselineDebateRunner(config, save_dir, args.dishonesty_level)
    results = runner.run_all(problems)

    # Save results summary
    summary = {
        "total": len(results),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "config": config_dict
    }
    with open(save_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
