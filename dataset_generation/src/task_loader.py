"""Task configuration loader."""

import yaml
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TaskLoader:
    """Loads task configurations from YAML files."""
    
    def __init__(self, config_dir: str):
        """Initialize the task loader.
        
        Args:
            config_dir: Directory containing task configuration files
        """
        self.config_dir = config_dir
    
    def load_task(self, task_name: str) -> Dict:
        """Load a task configuration by name.
        
        Args:
            task_name: Name of the task (e.g., 'refusal')
            
        Returns:
            Task configuration dictionary
        """
        task_path = os.path.join(self.config_dir, "tasks", f"{task_name}.yaml")
        
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task configuration not found: {task_path}")
        
        logger.info(f"Loading task configuration from: {task_path}")
        
        with open(task_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ["task_name", "templates", "examples", "generation_instructions"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in task config: {field}")
        
        logger.info(f"Successfully loaded task: {config['task_name']}")
        return config

