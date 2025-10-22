#!/usr/bin/env python3
"""Download completed batch results from OpenAI."""

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


def download_batch_results(request_dir: str, task: str, output_dir: str = "batch_outputs", metadata_file: str = "batch_metadata.json"):
    """Download all completed batch results.
    
    Args:
        request_dir: Directory containing batch metadata
        task: Task name
        output_dir: Directory to save downloaded results
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
        sys.exit(1)
    
    with open(metadata_path, 'r') as f:
        batch_metadata = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir) / task
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading batch results to: {output_path}")
    logger.info("")
    
    downloaded = 0
    skipped = 0
    
    for batch_info in batch_metadata:
        batch_id = batch_info["batch_id"]
        batch_file = Path(batch_info["batch_file"]).name
        status = batch_info.get("status", "unknown")
        
        # Extract batch number from filename (e.g., request_batch0.jsonl -> 0)
        batch_num = batch_file.replace("request_batch", "").replace(".jsonl", "")
        output_file = output_path / f"output_batch{batch_num}.jsonl"
        
        if output_file.exists():
            logger.info(f"{batch_file}: Already downloaded → {output_file.name}")
            skipped += 1
            continue
        
        if status != "completed":
            logger.info(f"{batch_file}: Status={status}, skipping")
            skipped += 1
            continue
        
        try:
            # Get batch details
            batch = client.batches.retrieve(batch_id)
            
            if not batch.output_file_id:
                logger.warning(f"{batch_file}: No output file ID, skipping")
                skipped += 1
                continue
            
            # Download output file
            logger.info(f"{batch_file}: Downloading...")
            file_response = client.files.content(batch.output_file_id)
            
            # Save to file
            with open(output_file, "wb") as f:
                f.write(file_response.text.encode("utf-8"))
            
            logger.info(f"  ✓ Saved to {output_file.name}")
            downloaded += 1
            
        except Exception as e:
            logger.error(f"  ✗ Error downloading {batch_file}: {e}")
            skipped += 1
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Download Summary")
    logger.info("=" * 80)
    logger.info(f"Downloaded: {downloaded}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 80)
    
    if downloaded > 0:
        logger.info("")
        logger.info("Next step: Process and validate results")
        logger.info("  python scripts/process_batch_results.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download completed batch results from OpenAI"
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
        "--output-dir",
        type=str,
        default="batch_outputs",
        help="Directory to save downloaded results (default: batch_outputs)",
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
    logger.info("Batch Results Downloader")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info("=" * 80)
    
    download_batch_results(
        request_dir=args.request_dir,
        task=args.task,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

