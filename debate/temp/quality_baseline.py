#!/usr/bin/env python3
"""
Evaluate API models for data contamination on QuALITY dataset.
Tests naive baseline: model sees only question and answers, no story access.

This helps determine if judge models have memorized the QuALITY dataset.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from debate.providers import create_provider
from debate.types import ModelConfig, ProviderType
from debate.dataset_loader import QuALITYDatasetLoader


class QualityContaminationTester:
    """Test models for QuALITY dataset contamination (no story access)"""
    
    def __init__(self, model_config: ModelConfig, verbose: bool = False):
        self.model_config = model_config
        self.provider = create_provider(model_config)
        self.verbose = verbose
        self.results = []
        
    def create_contamination_prompt(self, question: str, answer_a: str, answer_b: str) -> str:
        """
        Create prompt for contamination testing (no story provided).
        This tests if the model has memorized answers from the dataset.
        """
        prompt = f"""You are evaluating a reading comprehension and fact recall question. You must choose between two possible answers, and even though you haven't been given the specific story to read here, you have been trained on them. Pick the best option according to your memory.

Question: {question}

Option A: {answer_a}
Option B: {answer_b}

Think step by step about which answer is more likely to be correct. Then provide your final answer.

You must end your response with exactly one of these formats:
<answer>A</answer>
or
<answer>B</answer>"""
        
        return prompt
    
    def parse_model_response(self, response: str) -> Optional[str]:
        """Parse the model's response to extract A or B"""
        import re
        
        # Look for explicit answer tags
        answer_match = re.search(r'<answer>([AB])</answer>', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Fallback: look for clear statements
        response_lower = response.lower()
        
        # Check various patterns
        patterns = [
            (r'answer:\s*([ab])', 1),
            (r'final answer:\s*([ab])', 1),
            (r'option ([ab]) is correct', 1),
            (r'the answer is ([ab])', 1),
            (r'i choose ([ab])', 1),
            (r'select ([ab])', 1),
        ]
        
        for pattern, group_num in patterns:
            match = re.search(pattern, response_lower)
            if match:
                return match.group(group_num).upper()
        
        # Last resort: count mentions
        a_count = response_lower.count('option a') + response_lower.count('answer a')
        b_count = response_lower.count('option b') + response_lower.count('answer b')
        
        if a_count > b_count:
            return "A"
        elif b_count > a_count:
            return "B"
        
        return None  # Could not determine
    
    def evaluate_single_question(
        self, 
        problem,
        position_swap: bool = False
    ) -> Dict:
        """Evaluate a single QuALITY question without story access"""
        
        # Set up answers with optional position swap to control for position bias
        if position_swap:
            answer_a = problem.incorrect_answer
            answer_b = problem.correct_answer
            correct_label = "B"
        else:
            answer_a = problem.correct_answer
            answer_b = problem.incorrect_answer
            correct_label = "A"
        
        # Create contamination test prompt
        prompt = self.create_contamination_prompt(
            problem.question,
            answer_a,
            answer_b
        )
        
        if self.verbose:
            print(f"\nQuestion: {problem.question[:100]}...")
            print(f"A: {answer_a[:50]}...")
            print(f"B: {answer_b[:50]}...")
        
        # Get model response
        messages = [{"role": "user", "content": prompt}]
        
        try:
            start_time = time.time()
            response, metadata = self.provider.generate(messages)
            latency = time.time() - start_time
            
            # Parse response
            predicted = self.parse_model_response(response)
            
            if predicted is None:
                is_correct = None
                status = "unparseable"
            else:
                is_correct = (predicted == correct_label)
                status = "success"
            
            if self.verbose:
                print(f"Model predicted: {predicted}, Correct: {correct_label}")
                if predicted:
                    print(f"Result: {'✓' if is_correct else '✗'}")
            
            return {
                'question_id': problem.question_id,
                'story_title': problem.story_title,
                'question': problem.question,
                'correct_answer': problem.correct_answer,
                'incorrect_answer': problem.incorrect_answer,
                'position_swap': position_swap,
                'correct_label': correct_label,
                'predicted': predicted,
                'is_correct': is_correct,
                'response': response[:500],  # Truncate for storage
                'full_response': response,
                'latency': latency,
                'status': status,
                'model': self.model_config.model_name
            }
            
        except Exception as e:
            print(f"Error evaluating question: {e}")
            return {
                'question_id': problem.question_id,
                'story_title': problem.story_title,
                'question': problem.question,
                'status': 'error',
                'error': str(e),
                'model': self.model_config.model_name
            }
    
    def evaluate_dataset(
        self, 
        dataset_loader: QuALITYDatasetLoader,
        n_questions: int = 50,
        test_position_bias: bool = True,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Evaluate model on QuALITY questions without story access.
        
        Args:
            dataset_loader: QuALITY dataset loader
            n_questions: Number of questions to evaluate
            test_position_bias: If True, test each question twice with swapped positions
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Get problems
        problems = dataset_loader.get_problems_batch(n=n_questions, shuffle=True)
        
        print(f"\n{'='*60}")
        print(f"Contamination Test: {self.model_config.model_name}")
        print(f"Testing {len(problems)} questions")
        print(f"Position bias testing: {test_position_bias}")
        print(f"{'='*60}")
        
        results = []
        
        for i, problem in enumerate(problems):
            print(f"\nQuestion {i+1}/{len(problems)}")
            
            # Test without position swap
            result = self.evaluate_single_question(problem, position_swap=False)
            results.append(result)
            
            # Optionally test with position swap
            if test_position_bias:
                time.sleep(0.5)  # Small delay to avoid rate limits
                result_swap = self.evaluate_single_question(problem, position_swap=True)
                result_swap['question_id'] = f"{result_swap['question_id']}_swap"
                results.append(result_swap)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate metrics
        valid_df = df[df['status'] == 'success']
        
        if len(valid_df) > 0:
            accuracy = valid_df['is_correct'].mean()
            
            # Position bias analysis if applicable
            if test_position_bias:
                no_swap_df = valid_df[~valid_df['position_swap']]
                swap_df = valid_df[valid_df['position_swap']]
                
                no_swap_acc = no_swap_df['is_correct'].mean() if len(no_swap_df) > 0 else 0
                swap_acc = swap_df['is_correct'].mean() if len(swap_df) > 0 else 0
                position_bias = abs(no_swap_acc - swap_acc)
            else:
                no_swap_acc = accuracy
                swap_acc = None
                position_bias = None
        else:
            accuracy = 0
            no_swap_acc = 0
            swap_acc = None
            position_bias = None
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {self.model_config.model_name}")
        print(f"Total questions: {len(problems)}")
        print(f"Valid responses: {len(valid_df)}")
        print(f"Overall accuracy: {accuracy:.1%}")
        
        if test_position_bias:
            print(f"\nPosition Analysis:")
            print(f"  Correct first (A): {no_swap_acc:.1%}")
            if swap_acc is not None:
                print(f"  Correct second (B): {swap_acc:.1%}")
                print(f"  Position bias: {position_bias:.1%}")
        
        print(f"\nContamination Assessment:")
        if accuracy > 0.65:
            print("  ⚠️ HIGH contamination likely (>65% accuracy)")
        elif accuracy > 0.55:
            print("  ⚠️ MODERATE contamination possible (55-65% accuracy)")
        else:
            print("  ✓ LOW contamination risk (<55% accuracy)")
        
        print(f"  (Random baseline: 50%)")
        print(f"{'='*60}")
        
        # Store summary
        self.summary = {
            'model': self.model_config.model_name,
            'provider': self.model_config.provider.value,
            'n_questions': len(problems),
            'valid_responses': len(valid_df),
            'accuracy': accuracy,
            'no_swap_accuracy': no_swap_acc,
            'swap_accuracy': swap_acc,
            'position_bias': position_bias,
            'contamination_risk': 'high' if accuracy > 0.65 else ('moderate' if accuracy > 0.55 else 'low')
        }
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Test API models for QuALITY dataset contamination'
    )
    
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., gpt-4, claude-3-opus-20240229)')
    parser.add_argument('--provider', type=str, required=True,
                       choices=['openai', 'anthropic', 'openrouter'],
                       help='API provider')
    parser.add_argument('--api_key', type=str,
                       help='API key (or use environment variable)')
    
    # Dataset configuration
    parser.add_argument('--n_questions', type=int, default=50,
                       help='Number of questions to test')
    parser.add_argument('--csv_path', type=str,
                       help='Path to QuALITY CSV for filtering specific questions')
    
    # Testing options
    parser.add_argument('--test_position_bias', action='store_true',
                       help='Test each question twice with swapped answer positions')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Generation temperature (0 for deterministic)')
    parser.add_argument('--max_tokens', type=int, default=256,
                       help='Maximum tokens in response')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./contamination_results',
                       help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model configuration
    provider_type = ProviderType[args.provider.upper()]
    model_config = ModelConfig(
        provider=provider_type,
        model_name=args.model,
        api_key=args.api_key,
        generation_kwargs={
            'temperature': args.temperature,
            'max_tokens': args.max_tokens
        }
    )
    
    # Load dataset
    print("Loading QuALITY dataset...")
    dataset_loader = QuALITYDatasetLoader(csv_path=args.csv_path, seed=args.seed)
    print(f"Loaded {len(dataset_loader.dataset)} total questions")
    
    # Create tester
    tester = QualityContaminationTester(model_config, verbose=args.verbose)
    
    # Run evaluation
    results_df = tester.evaluate_dataset(
        dataset_loader,
        n_questions=args.n_questions,
        test_position_bias=args.test_position_bias,
        seed=args.seed
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_safe = args.model.replace('/', '_')
    
    # Save detailed results
    results_file = output_dir / f"contamination_{model_name_safe}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save summary
    summary_file = output_dir / f"contamination_summary_{model_name_safe}_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(tester.summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    # Create or update master summary
    master_file = output_dir / "contamination_master_summary.jsonl"
    with open(master_file, 'a') as f:
        summary_with_timestamp = {**tester.summary, 'timestamp': timestamp}
        f.write(json.dumps(summary_with_timestamp) + '\n')
    print(f"Added to master summary: {master_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())