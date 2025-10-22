"""Utility functions for dataset generation."""

import json
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(checkpoint_path: str, scenario_id: int, generated_scenarios: List[Dict]):
    """Save a checkpoint of the generation progress.
    
    Args:
        checkpoint_path: Path to save the checkpoint
        scenario_id: Current scenario ID
        generated_scenarios: List of generated scenarios so far
    """
    checkpoint_data = {
        "scenario_id": scenario_id,
        "generated_scenarios": generated_scenarios,
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    logger.info(f"Checkpoint saved: {scenario_id} scenarios")


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load a checkpoint if it exists.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Checkpoint data or None if no checkpoint exists
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)
    
    logger.info(f"Checkpoint loaded: resuming from scenario {checkpoint_data['scenario_id']}")
    return checkpoint_data


def write_prompts_to_jsonl(output_path: str, prompts: List[Dict]):
    """Write prompts to a JSONL file.
    
    Args:
        output_path: Path to the output JSONL file
        prompts: List of prompt dictionaries
    """
    with open(output_path, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
    
    logger.info(f"Wrote {len(prompts)} prompts to {output_path}")


def append_prompts_to_jsonl(output_path: str, prompts: List[Dict]):
    """Append prompts to a JSONL file.
    
    Args:
        output_path: Path to the output JSONL file
        prompts: List of prompt dictionaries
    """
    with open(output_path, 'a') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
    
    logger.debug(f"Appended {len(prompts)} prompts to {output_path}")


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

