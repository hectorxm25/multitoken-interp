#!/usr/bin/env python3
"""Submit batch request files to OpenAI Batch API."""

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


def submit_batches(request_dir: str, task: str, metadata_file: str = "batch_metadata.json"):
    """Submit all batch request files in a directory.
    
    Args:
        request_dir: Directory containing batch request JSONL files
        task: Task name
        metadata_file: File to save batch metadata
    """
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Find all batch files
    request_path = Path(request_dir) / task
    batch_files = sorted(request_path.glob("request_batch*.jsonl"))
    
    if not batch_files:
        logger.error(f"No batch files found in {request_path}")
        logger.error("Run create_batch_requests.py first!")
        sys.exit(1)
    
    logger.info(f"Found {len(batch_files)} batch files to submit")
    
    batch_metadata = []
    
    for batch_file in batch_files:
        logger.info(f"\nSubmitting: {batch_file.name}")
        
        try:
            # Upload file
            logger.info("  Uploading file...")
            with open(batch_file, "rb") as f:
                file_obj = client.files.create(file=f, purpose="batch")
            
            logger.info(f"  File ID: {file_obj.id}")
            
            # Create batch
            logger.info("  Creating batch job...")
            batch_obj = client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"{task} dataset generation",
                    "batch_file": batch_file.name
                }
            )
            
            logger.info(f"  Batch ID: {batch_obj.id}")
            logger.info(f"  Status: {batch_obj.status}")
            
            # Save metadata
            batch_info = {
                "batch_file": str(batch_file),
                "batch_id": batch_obj.id,
                "file_id": file_obj.id,
                "status": batch_obj.status,
                "created_at": batch_obj.created_at,
            }
            batch_metadata.append(batch_info)
            
        except Exception as e:
            logger.error(f"  Error submitting {batch_file.name}: {e}")
            continue
    
    # Save metadata to file
    metadata_path = request_path / metadata_file
    with open(metadata_path, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("Batch Submission Complete")
    logger.info("=" * 80)
    logger.info(f"Submitted {len(batch_metadata)} batches")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Check status: python scripts/check_batch_status.py")
    logger.info("2. Wait for completion (batches typically complete in 1-24 hours)")
    logger.info("3. Download results: python scripts/download_batch_results.py")
    logger.info("4. Process and validate: python scripts/process_batch_results.py")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Submit batch request files to OpenAI Batch API"
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
        help="Directory containing batch request files (default: requests)",
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
    
    logger.info("=" * 80)
    logger.info("Batch Submission Script")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Request directory: {args.request_dir}")
    logger.info("=" * 80)
    
    submit_batches(
        request_dir=args.request_dir,
        task=args.task
    )


if __name__ == "__main__":
    main()

