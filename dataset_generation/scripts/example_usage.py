#!/usr/bin/env python3
"""Example script showing how to use the generated dataset."""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_dataset(jsonl_path: str):
    """Load dataset from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of prompt dictionaries
    """
    prompts = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    return prompts


def organize_by_scenario(prompts):
    """Organize prompts by scenario ID.
    
    Args:
        prompts: List of prompt dictionaries
        
    Returns:
        Dictionary mapping scenario_id to its 4 prompts
    """
    scenarios = defaultdict(dict)
    
    for prompt in prompts:
        scenario_id = prompt['scenario_id']
        prompt_type = prompt['type']
        scenarios[scenario_id][prompt_type] = prompt['text']
    
    return dict(scenarios)


def print_scenario(scenario_id: int, scenario_data: dict):
    """Pretty print a single scenario.
    
    Args:
        scenario_id: The scenario ID
        scenario_data: Dictionary with the 4 prompt types
    """
    print(f"\n{'='*80}")
    print(f"Scenario {scenario_id}")
    print('='*80)
    
    print("\n[SINGLE-TOKEN PROMPT (Safe)]")
    print(scenario_data.get('single_token_prompt', 'N/A'))
    
    print("\n[SINGLE-TOKEN COUNTERFACTUAL (Harmful)]")
    print(scenario_data.get('single_token_counterfactual', 'N/A'))
    
    print("\n[MULTI-TOKEN PROMPT (Safe)]")
    print(scenario_data.get('multi_token_prompt', 'N/A'))
    
    print("\n[MULTI-TOKEN COUNTERFACTUAL (Harmful)]")
    print(scenario_data.get('multi_token_counterfactual', 'N/A'))


def main():
    """Main function to demonstrate dataset usage."""
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path_to_dataset.jsonl>")
        print("\nExample: python example_usage.py dataset.jsonl")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    print(f"Loading dataset from: {dataset_path}")
    prompts = load_dataset(dataset_path)
    
    print(f"Total prompts loaded: {len(prompts)}")
    
    # Organize by scenario
    scenarios = organize_by_scenario(prompts)
    num_scenarios = len(scenarios)
    
    print(f"Total scenarios: {num_scenarios}")
    
    # Show statistics
    types = defaultdict(int)
    for prompt in prompts:
        types[prompt['type']] += 1
    
    print("\nPrompt type distribution:")
    for prompt_type, count in sorted(types.items()):
        print(f"  {prompt_type}: {count}")
    
    # Display first 3 scenarios as examples
    print("\n" + "="*80)
    print("EXAMPLE SCENARIOS (First 3)")
    print("="*80)
    
    for i in range(min(3, num_scenarios)):
        if i in scenarios:
            print_scenario(i, scenarios[i])
    
    print(f"\n{'='*80}")
    print("Dataset loaded successfully!")
    print("="*80)
    
    # Example: Filter by type
    print("\n--- Example: Filter single-token prompts only ---")
    single_token_prompts = [p for p in prompts if p['type'] == 'single_token_prompt']
    print(f"Number of single-token prompts: {len(single_token_prompts)}")
    
    # Example: Get a specific scenario
    print("\n--- Example: Access specific scenario (ID=5) ---")
    if 5 in scenarios:
        print(f"Single-token prompt: {scenarios[5]['single_token_prompt'][:100]}...")
    else:
        print("Scenario 5 not found in dataset")


if __name__ == "__main__":
    main()

