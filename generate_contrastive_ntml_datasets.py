#!/usr/bin/env python3
"""
Generate Contrastive NTML Datasets

This script generates contrastive NTML datasets from statement/prompt pairs,
creating conversations with specified truth/lie ratios where each category
appears at most once per conversation.

Usage:
    python generate_contrastive_ntml_datasets.py --input statements.json --ratio "4T1L" --samples 300
    python generate_contrastive_ntml_datasets.py --input statements.json --ratios "2T1L,4T1L,3T2L" --samples 100
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def parse_ratio(ratio_str: str) -> Tuple[int, int]:
    """Parse ratio string like '4T1L' into (truth_count, lie_count)."""
    match = re.match(r'(\d+)T(\d+)L', ratio_str)
    if not match:
        raise ValueError(f"Invalid ratio format: {ratio_str}. Expected format like '4T1L'")
    return int(match.group(1)), int(match.group(2))


def load_statements(input_path: str) -> List[Dict]:
    """Load statements from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        statements = json.load(f)
    
    # Validate format
    required_fields = ['id', 'statement', 'truthful_system_prompt', 'deceptive_system_prompt', 'category']
    for i, stmt in enumerate(statements):
        for field in required_fields:
            if field not in stmt:
                raise ValueError(f"Missing field '{field}' in statement {i}")
    
    print(f"Loaded {len(statements)} statements")
    
    # Group by category
    by_category = defaultdict(list)
    for stmt in statements:
        by_category[stmt['category']].append(stmt)
    
    print(f"Found {len(by_category)} categories:")
    for category, stmts in by_category.items():
        print(f"  • {category}: {len(stmts)} statements")
    
    return statements


def select_statements_for_conversation(statements: List[Dict], truth_count: int, lie_count: int, 
                                     used_categories: Set[str] = None) -> Tuple[List[Dict], List[int]]:
    """
    Select statements for a single conversation ensuring:
    1. Each category appears at most once
    2. We get the right number of truths and lies
    
    Returns:
        statements_selected: List of selected statements
        lie_positions: List of indices where lies will be placed
    """
    if used_categories is None:
        used_categories = set()
    
    total_needed = truth_count + lie_count
    
    # Group statements by category, excluding already used categories
    available_by_category = defaultdict(list)
    for stmt in statements:
        if stmt['category'] not in used_categories:
            available_by_category[stmt['category']].append(stmt)
    
    available_categories = list(available_by_category.keys())
    
    if len(available_categories) < total_needed:
        raise ValueError(f"Not enough categories available. Need {total_needed}, have {len(available_categories)}")
    
    # Randomly select categories for this conversation
    selected_categories = random.sample(available_categories, total_needed)
    
    # Select one statement from each chosen category
    selected_statements = []
    for category in selected_categories:
        stmt = random.choice(available_by_category[category])
        selected_statements.append(stmt)
    
    # Randomly choose which positions will be lies
    lie_positions = random.sample(range(total_needed), lie_count)
    lie_positions.sort()  # Keep them in order
    
    return selected_statements, lie_positions


def create_conversation(statements: List[Dict], lie_positions: List[int], 
                       conversation_id: str) -> Dict:
    """Create a conversation from selected statements."""
    
    # Build the lie version (some statements use deceptive prompts)
    lie_system_parts = []
    lie_model_parts = []
    
    # Build the truth version (all statements use truthful prompts)  
    truth_system_parts = []
    truth_model_parts = []
    
    lie_statements = []
    original_ids = []
    
    for i, stmt in enumerate(statements):
        original_ids.append(stmt['id'])
        
        if i in lie_positions:
            # This position should be a lie
            lie_statements.append(stmt['statement'])
            lie_system_parts.append(stmt['deceptive_system_prompt'])
            lie_model_parts.append(stmt['statement'])
            
            # Truth version still uses truthful prompt for this statement
            truth_system_parts.append(stmt['truthful_system_prompt'])
            truth_model_parts.append(stmt['statement'])
        else:
            # This position should be truthful in both versions
            lie_system_parts.append(stmt['truthful_system_prompt'])
            lie_model_parts.append(stmt['statement'])
            
            truth_system_parts.append(stmt['truthful_system_prompt'])
            truth_model_parts.append(stmt['statement'])
    
    # Join the parts
    lie_system = " ".join(lie_system_parts)
    lie_model = " ".join(lie_model_parts)
    
    truth_system = " ".join(truth_system_parts)
    truth_model = " ".join(truth_model_parts)
    
    return {
        "id": conversation_id,
        "original_ids": original_ids,
        "lie_statements": lie_statements,
        "lie_ids": lie_positions,
        "lie_version": {
            "system": lie_system,
            "model": lie_model
        },
        "truth_version": {
            "system": truth_system,
            "model": truth_model
        }
    }


def generate_conversations(statements: List[Dict], ratio: str, num_samples: int, 
                          output_path: str, seed: int = None):
    """Generate conversations for a specific ratio."""
    
    if seed is not None:
        random.seed(seed)
    
    truth_count, lie_count = parse_ratio(ratio)
    total_statements = truth_count + lie_count
    
    print(f"\nGenerating {num_samples} conversations with ratio {ratio}")
    print(f"  • {truth_count} truths, {lie_count} lies per conversation")
    print(f"  • {total_statements} total statements per conversation")
    
    # Check if we have enough categories
    categories = set(stmt['category'] for stmt in statements)
    if len(categories) < total_statements:
        raise ValueError(f"Not enough categories for ratio {ratio}. "
                        f"Need {total_statements}, have {len(categories)}")
    
    conversations = []
    
    for i in range(num_samples):
        conversation_id = f"{ratio}_{i:03d}"
        
        try:
            selected_statements, lie_positions = select_statements_for_conversation(
                statements, truth_count, lie_count
            )
            
            conversation = create_conversation(selected_statements, lie_positions, conversation_id)
            conversations.append(conversation)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_samples} conversations")
                
        except ValueError as e:
            print(f"Warning: Could not generate conversation {i}: {e}")
            continue
    
    # Save conversations
    output_file = Path(output_path) / f"{ratio}_{num_samples}samples.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(conversations)} conversations to {output_file}")
    
    # Print some statistics
    if conversations:
        avg_lie_statements = sum(len(c['lie_statements']) for c in conversations) / len(conversations)
        categories_used = set()
        for c in conversations:
            for original_id in c['original_ids']:
                stmt = next(s for s in statements if s['id'] == original_id)
                categories_used.add(stmt['category'])
        
        print(f"  • Average lie statements per conversation: {avg_lie_statements:.1f}")
        print(f"  • Categories used: {len(categories_used)}/{len(categories)}")
    
    return output_file


def validate_conversations(conversations: List[Dict], statements: List[Dict]):
    """Validate generated conversations."""
    print("\n🔍 Validating conversations...")
    
    errors = []
    statement_lookup = {stmt['id']: stmt for stmt in statements}
    
    for i, conv in enumerate(conversations):
        conv_id = conv['id']
        
        # Check required fields
        required_fields = ['id', 'original_ids', 'lie_statements', 'lie_ids', 'lie_version', 'truth_version']
        for field in required_fields:
            if field not in conv:
                errors.append(f"Conversation {conv_id}: Missing field '{field}'")
                continue
        
        # Check that original_ids are valid
        for orig_id in conv['original_ids']:
            if orig_id not in statement_lookup:
                errors.append(f"Conversation {conv_id}: Invalid original_id {orig_id}")
        
        # Check that lie_ids are valid indices
        max_index = len(conv['original_ids']) - 1
        for lie_id in conv['lie_ids']:
            if lie_id < 0 or lie_id > max_index:
                errors.append(f"Conversation {conv_id}: lie_id {lie_id} out of range [0, {max_index}]")
        
        # Check that categories don't repeat
        categories = []
        for orig_id in conv['original_ids']:
            if orig_id in statement_lookup:
                categories.append(statement_lookup[orig_id]['category'])
        
        if len(categories) != len(set(categories)):
            duplicates = [cat for cat in categories if categories.count(cat) > 1]
            errors.append(f"Conversation {conv_id}: Duplicate categories {set(duplicates)}")
        
        # Check that lie_statements match the statements at lie_ids
        expected_lie_statements = []
        for lie_id in conv['lie_ids']:
            if lie_id < len(conv['original_ids']):
                orig_id = conv['original_ids'][lie_id]
                if orig_id in statement_lookup:
                    expected_lie_statements.append(statement_lookup[orig_id]['statement'])
        
        if set(conv['lie_statements']) != set(expected_lie_statements):
            errors.append(f"Conversation {conv_id}: lie_statements don't match lie_ids positions")
    
    if errors:
        print(f"❌ Found {len(errors)} validation errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  • {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        print(f"✅ All {len(conversations)} conversations are valid")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive NTML datasets from statement/prompt pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_contrastive_ntml_datasets.py --input statements.json --ratio "4T1L" --samples 300
    python generate_contrastive_ntml_datasets.py --input statements.json --ratios "2T1L,4T1L,3T2L" --samples 100
    python generate_contrastive_ntml_datasets.py --input statements.json --ratio "5T2L" --samples 200 --output ./datasets
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input JSON file containing statements and prompts"
    )
    
    # Ratio arguments (mutually exclusive)
    ratio_group = parser.add_mutually_exclusive_group(required=True)
    ratio_group.add_argument(
        "--ratio",
        type=str,
        help="Single ratio to generate (e.g., '4T1L')"
    )
    ratio_group.add_argument(
        "--ratios", 
        type=str,
        help="Comma-separated list of ratios (e.g., '2T1L,4T1L,3T2L')"
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=100,
        help="Number of conversations to generate per ratio (default: 100)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/NTML-datasets/contrastive",
        help="Output directory (default: ./data/NTML-datasets/contrastive)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated conversations"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("🎯 Contrastive NTML Dataset Generation")
    print("=" * 50)
    
    # Load statements
    try:
        statements = load_statements(args.input)
    except Exception as e:
        print(f"❌ Error loading statements: {e}")
        return 1
    
    # Determine ratios to generate
    if args.ratio:
        ratios = [args.ratio]
    else:
        ratios = [r.strip() for r in args.ratios.split(',')]
    
    print(f"\nGenerating datasets for ratios: {ratios}")
    print(f"Samples per ratio: {args.samples}")
    print(f"Output directory: {args.output}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    
    # Generate datasets
    generated_files = []
    for ratio in ratios:
        try:
            output_file = generate_conversations(
                statements, ratio, args.samples, args.output, args.seed
            )
            generated_files.append(output_file)
        except Exception as e:
            print(f"❌ Error generating {ratio}: {e}")
            continue
    
    # Validate if requested
    if args.validate and generated_files:
        print(f"\n{'='*50}")
        print("🔍 Validation Results")
        print("=" * 50)
        
        all_valid = True
        for output_file in generated_files:
            print(f"\nValidating {output_file.name}...")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                valid = validate_conversations(conversations, statements)
                if not valid:
                    all_valid = False
            except Exception as e:
                print(f"❌ Error validating {output_file.name}: {e}")
                all_valid = False
        
        if all_valid:
            print(f"\n✅ All generated datasets are valid!")
        else:
            print(f"\n❌ Some validation errors found. Please check the output above.")
    
    # Summary
    print(f"\n{'='*50}")
    print("🏁 Generation Complete")
    print("=" * 50)
    print(f"Generated {len(generated_files)} dataset files:")
    for output_file in generated_files:
        print(f"  • {output_file}")
    
    if generated_files:
        print(f"\n📋 Next Steps:")
        print(f"  • Train probes: python train_contrastive_ntml_probes.py --dataset {generated_files[0].stem}")
        print(f"  • Validate datasets: python generate_contrastive_ntml_datasets.py --input {args.input} --ratio {ratios[0]} --samples 5 --validate")
        return 0
    else:
        print("❌ No datasets generated successfully")
        return 1


if __name__ == "__main__":
    exit(main())