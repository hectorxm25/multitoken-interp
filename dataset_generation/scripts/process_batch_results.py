#!/usr/bin/env python3
"""Process batch results and validate with tokenizers."""

import argparse
import json
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validator import TokenValidator
from src.task_loader import TaskLoader
from src.utils import setup_logging, write_prompts_to_jsonl
import logging

logger = logging.getLogger(__name__)


def parse_batch_response(response_content: str) -> dict:
    """Parse a batch API response.
    
    The response may be plain text or JSON. Try to extract the pairs.
    
    Args:
        response_content: The response content from GPT-4o
        
    Returns:
        Dictionary with 'safe' and 'harmful' or list of pairs
    """
    try:
        # Try to parse as JSON first
        data = json.loads(response_content)
        
        # Handle various response formats
        if isinstance(data, dict):
            if "pairs" in data:
                return data["pairs"]
            elif "scenarios" in data:
                return data["scenarios"]
            elif "safe" in data and "harmful" in data:
                return [data]
        elif isinstance(data, list):
            return data
        
        return []
    except json.JSONDecodeError:
        # If not JSON, try to extract from text (fallback)
        logger.debug(f"Could not parse as JSON: {response_content[:100]}...")
        return []


def process_batch_results(
    input_dir: str,
    task: str,
    config_dir: str,
    output_file: str,
    target_scenarios: int = 1500,
    cache_dir: str = None
):
    """Process batch results and validate scenarios.
    
    Args:
        input_dir: Directory containing batch output files
        task: Task name
        config_dir: Configuration directory
        output_file: Output JSONL file for validated prompts
        target_scenarios: Target number of valid scenarios
        cache_dir: Cache directory for tokenizers
    """
    # Load task configuration
    task_loader = TaskLoader(config_dir)
    task_config = task_loader.load_task(task)
    
    # Initialize validator
    logger.info("Initializing token validator...")
    validator = TokenValidator(cache_dir=cache_dir)
    
    # Get templates
    single_token_prefix = task_config["templates"]["single_token"]["prefix"]
    single_token_suffix = task_config["templates"]["single_token"]["suffix"]
    multi_token_prefix = task_config["templates"]["multi_token"]["prefix"]
    multi_token_suffix = task_config["templates"]["multi_token"]["suffix"]
    
    # Find all batch output files
    input_path = Path(input_dir) / task
    batch_files = sorted(input_path.glob("output_batch*.jsonl"))
    
    if not batch_files:
        logger.error(f"No batch output files found in {input_path}")
        logger.error("Run download_batch_results.py first!")
        sys.exit(1)
    
    logger.info(f"Found {len(batch_files)} batch output files")
    logger.info(f"Target: {target_scenarios} valid scenarios")
    logger.info("")
    
    valid_scenarios = []
    total_generated = 0
    total_valid = 0
    
    # Statistics
    stats = {
        "total_responses": 0,
        "json_parse_errors": 0,
        "malformed_pairs": 0,
        "validation_failures": 0,
        "valid_scenarios": 0,
    }
    
    # Process each batch file
    for batch_file in batch_files:
        logger.info(f"Processing {batch_file.name}...")
        
        with open(batch_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    response_obj = json.loads(line)
                    stats["total_responses"] += 1
                    
                    # Extract the response content
                    if "response" not in response_obj:
                        continue
                    
                    response = response_obj["response"]
                    if "body" not in response or "choices" not in response["body"]:
                        continue
                    
                    content = response["body"]["choices"][0]["message"]["content"]
                    
                    # Parse the content
                    pairs = parse_batch_response(content)
                    
                    if not pairs:
                        stats["json_parse_errors"] += 1
                        continue
                    
                    # Process each pair
                    for pair in pairs:
                        if not isinstance(pair, dict) or "safe" not in pair or "harmful" not in pair:
                            stats["malformed_pairs"] += 1
                            continue
                        
                        safe_task = pair["safe"]
                        harmful_task = pair["harmful"]
                        total_generated += 1
                        
                        # Validate the scenario
                        is_valid, validation_details = validator.validate_scenario(
                            safe_task=safe_task,
                            harmful_task=harmful_task,
                            single_token_prefix=single_token_prefix,
                            single_token_suffix=single_token_suffix,
                            multi_token_prefix=multi_token_prefix,
                            multi_token_suffix=multi_token_suffix,
                        )
                        
                        if is_valid:
                            valid_scenarios.append({
                                "safe_task": safe_task,
                                "harmful_task": harmful_task,
                                "prompts": validation_details["prompts"],
                            })
                            total_valid += 1
                            stats["valid_scenarios"] += 1
                            
                            # Stop if we've reached our target
                            if total_valid >= target_scenarios:
                                logger.info(f"✓ Reached target of {target_scenarios} valid scenarios!")
                                break
                        else:
                            stats["validation_failures"] += 1
                            logger.debug(
                                f"Validation failed: {safe_task} / {harmful_task}"
                            )
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse response line: {e}")
                    stats["json_parse_errors"] += 1
                    continue
                except Exception as e:
                    logger.warning(f"Error processing response: {e}")
                    continue
                
                # Check if we've reached target
                if total_valid >= target_scenarios:
                    break
        
        logger.info(f"  Valid scenarios so far: {total_valid}")
        
        # Check if we've reached target
        if total_valid >= target_scenarios:
            break
    
    # Convert scenarios to prompts
    logger.info("")
    logger.info("Converting scenarios to prompts...")
    all_prompts = []
    
    for i, scenario in enumerate(valid_scenarios):
        # Single token prompt
        all_prompts.append({
            "scenario_id": i,
            "type": "single_token_prompt",
            "task": task,
            "text": scenario["prompts"]["single_prompt"],
        })
        
        # Single token counterfactual
        all_prompts.append({
            "scenario_id": i,
            "type": "single_token_counterfactual",
            "task": task,
            "text": scenario["prompts"]["single_counterfactual"],
        })
        
        # Multi-token prompt
        all_prompts.append({
            "scenario_id": i,
            "type": "multi_token_prompt",
            "task": task,
            "text": scenario["prompts"]["multi_prompt"],
        })
        
        # Multi-token counterfactual
        all_prompts.append({
            "scenario_id": i,
            "type": "multi_token_counterfactual",
            "task": task,
            "text": scenario["prompts"]["multi_counterfactual"],
        })
    
    # Write to file
    write_prompts_to_jsonl(output_file, all_prompts)
    
    # Final statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("Processing Complete")
    logger.info("=" * 80)
    logger.info(f"Total API responses processed: {stats['total_responses']}")
    logger.info(f"Total pairs generated: {total_generated}")
    logger.info(f"Valid scenarios: {stats['valid_scenarios']}")
    logger.info(f"Validation failures: {stats['validation_failures']}")
    logger.info(f"JSON parse errors: {stats['json_parse_errors']}")
    logger.info(f"Malformed pairs: {stats['malformed_pairs']}")
    logger.info(f"Success rate: {(stats['valid_scenarios']/total_generated*100):.1f}%")
    logger.info("")
    logger.info(f"Total prompts written: {len(all_prompts)}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Process batch results and validate with tokenizers"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="refusal",
        help="Task name (default: refusal)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="batch_outputs",
        help="Directory containing batch output files (default: batch_outputs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.jsonl",
        help="Output JSONL file (default: dataset.jsonl)",
    )
    parser.add_argument(
        "--target-scenarios",
        type=int,
        default=1500,
        help="Target number of valid scenarios (default: 1500)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Configuration directory (default: ../config)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for tokenizers (default: .cache in project root)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Set config directory
    if args.config_dir is None:
        script_dir = Path(__file__).parent
        args.config_dir = str(script_dir.parent / "config")
    
    logger.info("=" * 80)
    logger.info("Batch Results Processor")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Target scenarios: {args.target_scenarios}")
    logger.info("=" * 80)
    
    process_batch_results(
        input_dir=args.input_dir,
        task=args.task,
        config_dir=args.config_dir,
        output_file=args.output,
        target_scenarios=args.target_scenarios,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()

