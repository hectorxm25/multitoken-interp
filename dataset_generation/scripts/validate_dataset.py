#!/usr/bin/env python3
"""Script to validate an existing dataset for token constraints."""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validator import TokenValidator
from src.utils import setup_logging
import logging

logger = logging.getLogger(__name__)


def load_dataset(jsonl_path: str):
    """Load dataset from JSONL file."""
    prompts = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    return prompts


def organize_by_scenario(prompts):
    """Organize prompts by scenario ID."""
    scenarios = defaultdict(dict)
    
    for prompt in prompts:
        scenario_id = prompt['scenario_id']
        prompt_type = prompt['type']
        scenarios[scenario_id][prompt_type] = prompt['text']
    
    return dict(scenarios)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a dataset for token-level constraints"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset JSONL file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Show detailed information about failed scenarios",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("=" * 80)
    logger.info("Dataset Validation Script")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info("=" * 80)
    
    # Check if file exists
    if not Path(args.dataset).exists():
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Load dataset
    logger.info("Loading dataset...")
    prompts = load_dataset(args.dataset)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Organize by scenario
    scenarios = organize_by_scenario(prompts)
    num_scenarios = len(scenarios)
    logger.info(f"Found {num_scenarios} scenarios")
    
    # Initialize validator
    logger.info("Initializing token validator...")
    validator = TokenValidator()
    
    # Validate each scenario
    logger.info("Validating scenarios...")
    
    valid_scenarios = 0
    failed_scenarios = []
    single_token_failures = 0
    multi_token_failures = 0
    
    for scenario_id in sorted(scenarios.keys()):
        scenario = scenarios[scenario_id]
        
        # Check if all 4 prompts are present
        required_types = ['single_token_prompt', 'single_token_counterfactual', 
                         'multi_token_prompt', 'multi_token_counterfactual']
        
        if not all(t in scenario for t in required_types):
            logger.warning(f"Scenario {scenario_id}: Missing prompt types")
            failed_scenarios.append((scenario_id, "missing_prompts", {}))
            continue
        
        # Validate single-token pair
        single_valid, single_info = validator.validate_pair(
            scenario['single_token_prompt'],
            scenario['single_token_counterfactual']
        )
        
        # Validate multi-token pair
        multi_valid, multi_info = validator.validate_pair(
            scenario['multi_token_prompt'],
            scenario['multi_token_counterfactual']
        )
        
        if single_valid and multi_valid:
            valid_scenarios += 1
        else:
            if not single_valid:
                single_token_failures += 1
            if not multi_valid:
                multi_token_failures += 1
            
            failed_scenarios.append((
                scenario_id,
                "validation_failed",
                {
                    "single_token": single_info,
                    "multi_token": multi_info,
                    "scenario": scenario
                }
            ))
    
    # Print results
    logger.info("=" * 80)
    logger.info("Validation Results")
    logger.info("=" * 80)
    logger.info(f"Total scenarios: {num_scenarios}")
    logger.info(f"Valid scenarios: {valid_scenarios}")
    logger.info(f"Failed scenarios: {len(failed_scenarios)}")
    logger.info(f"  - Single-token failures: {single_token_failures}")
    logger.info(f"  - Multi-token failures: {multi_token_failures}")
    logger.info(f"Success rate: {valid_scenarios / num_scenarios * 100:.2f}%")
    logger.info("=" * 80)
    
    # Show failure details if requested
    if args.show_failures and failed_scenarios:
        logger.info("\nFailure Details:")
        logger.info("=" * 80)
        
        for scenario_id, reason, details in failed_scenarios[:10]:  # Show first 10
            logger.info(f"\nScenario {scenario_id}: {reason}")
            
            if reason == "validation_failed":
                if 'single_token' in details:
                    logger.info("  Single-token validation:")
                    logger.info(f"    Equal counts: {details['single_token']['equal_token_counts']}")
                    logger.info(f"    One diff: {details['single_token']['one_token_difference']}")
                    logger.info(f"    Diff counts: {details['single_token']['diff_counts']}")
                
                if 'multi_token' in details:
                    logger.info("  Multi-token validation:")
                    logger.info(f"    Equal counts: {details['multi_token']['equal_token_counts']}")
                    logger.info(f"    One diff: {details['multi_token']['one_token_difference']}")
                    logger.info(f"    Diff counts: {details['multi_token']['diff_counts']}")
        
        if len(failed_scenarios) > 10:
            logger.info(f"\n... and {len(failed_scenarios) - 10} more failures")
    
    # Exit with appropriate code
    if valid_scenarios == num_scenarios:
        logger.info("\n✓ All scenarios passed validation!")
        sys.exit(0)
    else:
        logger.warning(f"\n✗ {len(failed_scenarios)} scenarios failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()

