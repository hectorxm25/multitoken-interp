#!/usr/bin/env python3
"""Main script for generating the interpretability dataset."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import DatasetGenerator
from src.utils import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    write_prompts_to_jsonl,
    append_prompts_to_jsonl,
)
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate interpretability dataset with token-level validation"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="refusal",
        help="Task name (default: refusal)",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=1500,
        help="Number of scenarios to generate (default: 1500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.jsonl",
        help="Output JSONL file path (default: dataset.jsonl)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.json",
        help="Checkpoint file path (default: checkpoint.json)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Configuration directory (default: ../config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint/resume functionality",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for tokenizers (default: .cache in project root)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Set config directory
    if args.config_dir is None:
        script_dir = Path(__file__).parent
        args.config_dir = str(script_dir.parent / "config")
    
    logger.info("=" * 80)
    logger.info("Dataset Generation Script")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Target scenarios: {args.num_scenarios}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Config directory: {args.config_dir}")
    logger.info("=" * 80)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Initialize generator
    logger.info("Initializing dataset generator...")
    generator = DatasetGenerator(
        config_dir=args.config_dir,
        task_name=args.task,
        model=args.model,
        cache_dir=args.cache_dir,
    )
    
    # Load checkpoint if exists
    start_scenario_id = 0
    generated_scenarios = []
    
    if not args.no_checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint:
            start_scenario_id = checkpoint["scenario_id"]
            generated_scenarios = checkpoint["generated_scenarios"]
            logger.info(f"Resuming from checkpoint: {start_scenario_id} scenarios already generated")
    
    remaining_scenarios = args.num_scenarios - start_scenario_id
    
    if remaining_scenarios <= 0:
        logger.info("All scenarios already generated!")
    else:
        logger.info(f"Generating {remaining_scenarios} more scenarios...")
        
        # Generate scenarios in batches
        batch_size = 50  # Save checkpoint every 50 scenarios
        scenarios_generated = 0
        
        while scenarios_generated < remaining_scenarios:
            batch_count = min(batch_size, remaining_scenarios - scenarios_generated)
            
            logger.info(f"Generating batch of {batch_count} scenarios...")
            batch_scenarios = generator.generate_batch_with_validation(batch_count)
            
            # Convert to prompts
            batch_prompts = generator.scenarios_to_prompts(
                batch_scenarios,
                start_id=start_scenario_id + scenarios_generated
            )
            
            # Append to output file
            if scenarios_generated == 0 and start_scenario_id == 0:
                # First batch, create new file
                write_prompts_to_jsonl(args.output, batch_prompts)
            else:
                # Subsequent batches, append
                append_prompts_to_jsonl(args.output, batch_prompts)
            
            # Update progress
            generated_scenarios.extend(batch_scenarios)
            scenarios_generated += len(batch_scenarios)
            
            # Save checkpoint
            if not args.no_checkpoint:
                save_checkpoint(
                    args.checkpoint,
                    start_scenario_id + scenarios_generated,
                    generated_scenarios
                )
            
            logger.info(f"Progress: {scenarios_generated}/{remaining_scenarios} scenarios generated")
    
    # Final statistics
    stats = generator.get_stats()
    total_scenarios = start_scenario_id + len(generated_scenarios)
    total_prompts = total_scenarios * 4  # Each scenario produces 4 prompts
    
    logger.info("=" * 80)
    logger.info("Generation Complete!")
    logger.info("=" * 80)
    logger.info(f"Total scenarios: {total_scenarios}")
    logger.info(f"Total prompts: {total_prompts}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"API tokens used: {stats['total_tokens']}")
    logger.info(f"Total API cost: ${stats['total_cost']:.2f}")
    logger.info("=" * 80)
    
    # Clean up checkpoint if complete
    if not args.no_checkpoint and os.path.exists(args.checkpoint):
        if total_scenarios >= args.num_scenarios:
            logger.info(f"Removing checkpoint file: {args.checkpoint}")
            os.remove(args.checkpoint)


if __name__ == "__main__":
    main()

