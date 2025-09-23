import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset

from .types import APPSProblem


class SimpleDebateDatasetLoader:
    """Loader for simple debate scenarios that fit in limited context"""
    
    def __init__(self, dataset_path: str, seed: int = 42):
        """
        Initialize simple debate dataset loader
        
        Args:
            dataset_path: Path to JSONL file with debate scenarios
            seed: Random seed for sampling
        """
        self.dataset_path = Path(dataset_path)
        self.random = random.Random(seed)
        self.scenarios = []
        self._load_dataset()
        
    def _load_dataset(self):
        """Load debate scenarios from JSONL file"""
        print(f"Loading simple debate dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    scenario = json.loads(line)
                    self.scenarios.append(scenario)
        
        print(f"Loaded {len(self.scenarios)} debate scenarios")
    
    def get_scenario(self, index: int) -> Dict:
        """Get a specific scenario by index"""
        return self.scenarios[index]
    
    def get_random_scenario(self) -> Dict:
        """Get a random scenario"""
        return self.random.choice(self.scenarios)
    
    def get_scenarios_batch(self, n: int, shuffle: bool = True) -> List[Dict]:
        """Get a batch of scenarios"""
        scenarios = self.scenarios.copy()
        
        if shuffle:
            self.random.shuffle(scenarios)
        
        return scenarios[:n]

class QuALITYDatasetLoader:
    """Loader for QuALITY dataset with story and reading comprehension questions"""
    
    def __init__(self, csv_path: Optional[str] = None, seed: int = 42):
        self.csv_path = Path(csv_path) if csv_path else None
        self.random = random.Random(seed)
        self.dataset = []
        self._load_dataset()
        
    def _load_dataset(self):
        """Load QuALITY dataset from HuggingFace (same method as their repo)"""
        print("Loading QuALITY dataset from HuggingFace...")
        
        # Parse dataset using same approach as their code
        import requests
        import tempfile
        import os
        
        # Download and parse the dataset files they use
        for split in ["train", "dev"]:
            dataset_name = f"QuALITY.v1.0.1.htmlstripped.{split}"
            url = f"https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/{dataset_name}?raw=true"
            
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse each line as JSON (their format)
            for line in response.text.splitlines():
                if line.strip():
                    item = json.loads(line)
                    
                    # Extract questions with best distractor selection
                    for question_obj in item.get('questions', []):
                        # Find best distractor based on validation data
                        best_distractor_idx = self._get_best_distractor_index(question_obj)
                        correct_idx = question_obj['gold_label'] - 1  # Convert to 0-indexed
                        
                        self.dataset.append({
                            'question_id': f"{item['article_id']}_{question_obj.get('question', '').replace(' ', '_')[:20]}",
                            'story': item['article'],
                            'story_title': item['title'],
                            'question': question_obj['question'],
                            'answers': question_obj['options'],
                            'correct_answer': question_obj['options'][correct_idx],
                            'incorrect_answer': question_obj['options'][best_distractor_idx],
                            'gold_label': question_obj['gold_label'],
                            'difficulty': question_obj.get('difficult', 0),
                            'metadata': {
                                'article_id': item['article_id'],
                                'set_unique_id': item.get('set_unique_id', ''),
                                'source': item.get('source', ''),
                            }
                        })
        
        # Filter by CSV if provided
        if self.csv_path and self.csv_path.exists():
            csv_df = pd.read_csv(self.csv_path)
            valid_questions = set(zip(csv_df['question'], csv_df['story_title']))
            
            self.dataset = [
                q for q in self.dataset 
                if (q['question'], q['story_title']) in valid_questions
            ]
        
        print(f"Loaded {len(self.dataset)} QuALITY questions")
    
    def _get_best_distractor_index(self, question_obj):
        """Get index of best distractor from validation data"""
        validations = question_obj.get('validation', [])
        if not validations:
            # Fallback: pick first non-correct option
            correct_idx = question_obj['gold_label'] - 1
            for i in range(len(question_obj['options'])):
                if i != correct_idx:
                    return i
        
        # Count votes for best distractor
        distractor_votes = {}
        for v in validations:
            if 'untimed_best_distractor' in v:
                idx = v['untimed_best_distractor'] - 1  # Convert to 0-indexed
                if idx != question_obj['gold_label'] - 1:
                    distractor_votes[idx] = distractor_votes.get(idx, 0) + 1
        
        if distractor_votes:
            return max(distractor_votes, key=distractor_votes.get)
        
        # Fallback
        correct_idx = question_obj['gold_label'] - 1
        for i in range(len(question_obj['options'])):
            if i != correct_idx:
                return i
        return 0
    
    def get_problem(self, index: int) -> QuALITYProblem:
        """Get a specific problem by index"""
        item = self.dataset[index]
        return QuALITYProblem(**item)
    
    def get_random_problem(self) -> QuALITYProblem:
        """Get a random problem"""
        item = self.random.choice(self.dataset)
        return QuALITYProblem(**item)
    
    def get_problems_batch(self, n: int, shuffle: bool = True) -> List[QuALITYProblem]:
        """Get batch of problems"""
        items = self.dataset.copy()
        if shuffle:
            self.random.shuffle(items)
        return [QuALITYProblem(**item) for item in items[:n]]

class APPSDatasetLoader:
    """Loader for APPS dataset problems"""
    
    def __init__(self, difficulty: Optional[str] = None, seed: int = 42):
        """
        Initialize APPS dataset loader
        
        Args:
            difficulty: Filter by difficulty ('introductory', 'interview', 'competition')
            seed: Random seed for sampling
        """
        # --- DEBUG PRINT ---
        print(f"[DEBUG] APPSDatasetLoader initialized with difficulty='{difficulty}', seed={seed}")
        
        self.difficulty = difficulty
        self.random = random.Random(seed)
        self.dataset = None
        self._load_dataset()
        
    def _load_dataset(self):
        """Load APPS dataset from HuggingFace"""
        print("Loading APPS dataset...")
        
        # --- FIX ---
        # Per your excellent suggestion, we are now specifying the exact commit hash
        # where the dataset was converted to Parquet. This forces the `datasets` library
        # to load this specific, script-free version, bypassing any local cache issues.
        commit_hash = "4e6e3a80cd104a48bf29c51eae491bbb4d9b33fc"
        print(f"[DEBUG] Forcing dataset load from revision: {commit_hash}")
        
        self.dataset = load_dataset(
            "codeparrot/apps",
            split="test",
            revision=commit_hash,
            trust_remote_code=True
        )
        
        # --- DEBUG PRINT ---
        print(f"[DEBUG] Dataset loaded successfully. Features: {self.dataset.features}")
        
        if self.difficulty:
            # --- DEBUG PRINT ---
            print(f"[DEBUG] Filtering dataset by difficulty: '{self.difficulty}'")
            self.dataset = self.dataset.filter(
                lambda x: x["difficulty"] == self.difficulty
            )
            
        print(f"Loaded {len(self.dataset)} problems after filtering.")
        
    def get_problem(self, index: int) -> APPSProblem:
        """Get a specific problem by index"""
        item = self.dataset[index]
        
        input_output = []
        if item.get("input_output"):
            try:
                io_data = json.loads(item["input_output"])
                if "inputs" in io_data and "outputs" in io_data:
                    for inp, out in zip(io_data["inputs"], io_data["outputs"]):
                        input_output.append((inp, out))
            except (json.JSONDecodeError, TypeError):
                pass # Ignore malformed io_data
                    
        return APPSProblem(
            problem_id=str(item["problem_id"]), # Use the field from Parquet
            question=item["question"],
            solutions=json.loads(item["solutions"]) if item.get("solutions") else [],
            input_output=input_output,
            difficulty=item.get("difficulty", "unknown"),
            tags=[]
        )
        
    def get_random_problem(self) -> APPSProblem:
        """Get a random problem"""
        index = self.random.randint(0, len(self.dataset) - 1)
        return self.get_problem(index)
        
    def get_problems_batch(self, n: int, shuffle: bool = True) -> List[APPSProblem]:
        """Get a batch of problems"""
        # --- DEBUG PRINT ---
        print(f"[DEBUG] get_problems_batch called with n={n}, shuffle={shuffle}")
        
        indices = list(range(len(self.dataset)))
        
        if shuffle:
            self.random.shuffle(indices)
            
        problems = []
        for i in indices[:n]:
            try:
                problems.append(self.get_problem(i))
            except Exception as e:
                print(f"Failed to load problem {i}: {e}")
                continue
                
        return problems


class SolutionGenerator:
    """Generate honest and backdoored solutions for problems"""
    
    @staticmethod
    def create_backdoored_solution(original_solution: str, 
                                 backdoor_type: str = "logic_bomb") -> str:
        """Adds a backdoor to a solution."""
        if backdoor_type == "logic_bomb":
            backdoor = "\n    # Secret trigger for special handling\n    if '1337' in str(locals()): return -1"
            lines = original_solution.split('\n')
            insert_point = len(lines) // 2
            lines.insert(insert_point, backdoor)
            return '\n'.join(lines)
        else:
            return original_solution.replace("return result", "return result + (0 if result != 42 else -1)")
            
    @staticmethod
    def generate_solutions(problem: APPSProblem) -> Tuple[str, str]:
        """Generates honest and backdoored solutions for a problem."""
        if problem.solutions:
            honest_solution = problem.solutions[0]
        else:
            honest_solution = "def solve(input):\n    # Placeholder logic\n    return len(str(input))"
        
        backdoored_solution = SolutionGenerator.create_backdoored_solution(honest_solution)
        return honest_solution, backdoored_solution


class FlexibleDatasetLoader:
    """Flexible loader that can handle different dataset formats"""
    
    @staticmethod
    def load_dataset(dataset_name: str, **kwargs) -> List[Dict]:
        """
        Load a dataset by name
        
        Args:
            dataset_name: Name of the dataset ('apps', 'custom', etc.)
            **kwargs: Additional arguments for specific datasets
        """
        # --- DEBUG PRINT ---
        print(f"[DEBUG] FlexibleDatasetLoader.load_dataset called for '{dataset_name}' with kwargs: {kwargs}")

        if dataset_name.lower() == "apps":
            init_kwargs = {
                'difficulty': kwargs.get('difficulty'),
                'seed': kwargs.get('seed', 42)
            }
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

            # --- DEBUG PRINT ---
            print(f"[DEBUG] Initializing APPSDatasetLoader with: {init_kwargs}")
            loader = APPSDatasetLoader(**init_kwargs)
            
            n_problems = kwargs.get("n_problems", 10)
            shuffle = kwargs.get("shuffle", True)
            
            problems = loader.get_problems_batch(n=n_problems, shuffle=shuffle)
            
            dataset = []
            for problem in problems:
                honest_sol, backdoored_sol = SolutionGenerator.generate_solutions(problem)
                dataset.append({
                    "problem": problem,
                    "honest_solution": honest_sol,
                    "backdoored_solution": backdoored_sol
                })
            return dataset
            
        elif dataset_name.lower() == "custom":
            path = Path(kwargs.get("path", "data/custom_dataset.json"))
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"Custom dataset not found at {path}")
                
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    @staticmethod
    def save_dataset(dataset: List[Dict], path: str):
        """Save dataset to file for reproducibility"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_dataset = []
        for item in dataset:
            serializable_item = item.copy()
            if isinstance(item.get("problem"), APPSProblem):
                serializable_item["problem"] = item["problem"].__dict__
            serializable_dataset.append(serializable_item)
            
        with open(path, 'w') as f:
            json.dump(serializable_dataset, f, indent=2)
            
        print(f"Saved dataset to {path}")