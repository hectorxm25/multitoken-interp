#!/usr/bin/env python3
"""Check status of submitted batch jobs."""

import argparse
import json
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.utils import setup_logging
import logging

logger = logging.getLogger(__name__)


def check_batch_status(request_dir: str, task: str, metadata_file: str = "batch_metadata.json"):
    """Check status of all submitted batches.
    
    Args:
        request_dir: Directory containing batch metadata
        task: Task name
        metadata_file: Metadata file name
    """
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set!")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Load metadata
    metadata_path = Path(request_dir) / task / metadata_file
    
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        logger.error("Run submit_batches.py first!")
        sys.exit(1)
    
    with open(metadata_path, 'r') as f:
        batch_metadata = json.load(f)
    
    logger.info(f"Checking status of {len(batch_metadata)} batches...")
    logger.info("")
    
    status_counts = {}
    updated_metadata = []
    
    for batch_info in batch_metadata:
        batch_id = batch_info["batch_id"]
        batch_file = Path(batch_info["batch_file"]).name
        
        try:
            batch = client.batches.retrieve(batch_id)
            
            status = batch.status
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Update metadata
            batch_info["status"] = status
            batch_info["request_counts"] = {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed,
            }
            
            if hasattr(batch, 'output_file_id') and batch.output_file_id:
                batch_info["output_file_id"] = batch.output_file_id
            
            updated_metadata.append(batch_info)
            
            # Log status
            logger.info(f"{batch_file}: {status}")
            if status == "completed":
                logger.info(f"  ✓ Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
                if batch.request_counts.failed > 0:
                    logger.info(f"  ✗ Failed: {batch.request_counts.failed}")
            elif status == "failed":
                logger.info(f"  ✗ Batch failed")
            elif status in ["validating", "in_progress", "finalizing"]:
                logger.info(f"  ⏳ Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
            
        except Exception as e:
            logger.error(f"  Error checking {batch_file}: {e}")
            updated_metadata.append(batch_info)
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(updated_metadata, f, indent=2)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Status Summary")
    logger.info("=" * 80)
    for status, count in sorted(status_counts.items()):
        logger.info(f"{status}: {count}")
    logger.info("=" * 80)
    
    if status_counts.get("completed", 0) > 0:
        logger.info("")
        logger.info("✓ Some batches completed! Download results:")
        logger.info("  python scripts/download_batch_results.py")


def main():
    parser = argparse.ArgumentParser(
        description="Check status of submitted batch jobs"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="refusal",
        help="Task name (default: refusal)",
    )
    parser.add_argument(
        "--request-dir",
        type=str,
        default="requests",
        help="Directory containing batch metadata (default: requests)",
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
    
    check_batch_status(
        request_dir=args.request_dir,
        task=args.task
    )


if __name__ == "__main__":
    main()

