"""Core dataset generation logic with validation and retry."""

import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

from .validator import TokenValidator
from .api_client import OpenAIClient
from .task_loader import TaskLoader

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generates dataset with token-level validation."""
    
    def __init__(self, config_dir: str, task_name: str, model: str = "gpt-4o", cache_dir: str = None):
        """Initialize the dataset generator.
        
        Args:
            config_dir: Directory containing task configurations
            task_name: Name of the task to generate
            model: OpenAI model to use
            cache_dir: Directory to cache tokenizers (optional)
        """
        self.task_loader = TaskLoader(config_dir)
        self.task_config = self.task_loader.load_task(task_name)
        self.task_name = task_name
        
        self.validator = TokenValidator(cache_dir=cache_dir)
        self.api_client = OpenAIClient(model=model)
        
        # Extract templates
        self.single_token_prefix = self.task_config["templates"]["single_token"]["prefix"]
        self.single_token_suffix = self.task_config["templates"]["single_token"]["suffix"]
        self.multi_token_prefix = self.task_config["templates"]["multi_token"]["prefix"]
        self.multi_token_suffix = self.task_config["templates"]["multi_token"]["suffix"]
        
        self.batch_size = self.task_config.get("batch_size", 10)
    
    def generate_and_validate_scenario(self, max_retries: int = 5) -> Tuple[bool, Dict]:
        """Generate a single scenario and validate it.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (success, scenario_data or None)
        """
        for attempt in range(max_retries):
            try:
                # Generate scenarios from API
                scenarios = self.api_client.generate_scenarios(self.task_config, num_scenarios=1)
                
                if not scenarios:
                    logger.warning(f"No scenarios returned (attempt {attempt + 1}/{max_retries})")
                    continue
                
                scenario = scenarios[0]
                safe_task = scenario["safe"]
                harmful_task = scenario["harmful"]
                
                # Validate the scenario
                is_valid, validation_details = self.validator.validate_scenario(
                    safe_task=safe_task,
                    harmful_task=harmful_task,
                    single_token_prefix=self.single_token_prefix,
                    single_token_suffix=self.single_token_suffix,
                    multi_token_prefix=self.multi_token_prefix,
                    multi_token_suffix=self.multi_token_suffix,
                )
                
                if is_valid:
                    logger.debug(f"Scenario validated successfully: {safe_task} / {harmful_task}")
                    return True, {
                        "safe_task": safe_task,
                        "harmful_task": harmful_task,
                        "prompts": validation_details["prompts"],
                    }
                else:
                    logger.debug(
                        f"Scenario failed validation (attempt {attempt + 1}/{max_retries}): "
                        f"{safe_task} / {harmful_task}"
                    )
                    logger.debug(f"Validation details: {validation_details}")
            
            except Exception as e:
                logger.warning(f"Error generating scenario (attempt {attempt + 1}/{max_retries}): {e}")
        
        logger.error(f"Failed to generate valid scenario after {max_retries} attempts")
        return False, None
    
    def generate_batch_with_validation(self, num_scenarios: int) -> List[Dict]:
        """Generate a batch of scenarios with validation and retry.
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of validated scenarios
        """
        validated_scenarios = []
        
        with tqdm(total=num_scenarios, desc="Generating scenarios") as pbar:
            while len(validated_scenarios) < num_scenarios:
                success, scenario_data = self.generate_and_validate_scenario()
                
                if success:
                    validated_scenarios.append(scenario_data)
                    pbar.update(1)
                else:
                    logger.warning("Failed to generate a valid scenario, continuing...")
        
        return validated_scenarios
    
    def scenarios_to_prompts(self, scenarios: List[Dict], start_id: int = 0) -> List[Dict]:
        """Convert scenarios to individual prompts in JSONL format.
        
        Args:
            scenarios: List of scenario dictionaries
            start_id: Starting scenario ID
            
        Returns:
            List of prompt dictionaries ready for JSONL output
        """
        prompts = []
        
        for i, scenario in enumerate(scenarios):
            scenario_id = start_id + i
            
            # Single token prompt
            prompts.append({
                "scenario_id": scenario_id,
                "type": "single_token_prompt",
                "task": self.task_name,
                "text": scenario["prompts"]["single_prompt"],
            })
            
            # Single token counterfactual
            prompts.append({
                "scenario_id": scenario_id,
                "type": "single_token_counterfactual",
                "task": self.task_name,
                "text": scenario["prompts"]["single_counterfactual"],
            })
            
            # Multi-token prompt
            prompts.append({
                "scenario_id": scenario_id,
                "type": "multi_token_prompt",
                "task": self.task_name,
                "text": scenario["prompts"]["multi_prompt"],
            })
            
            # Multi-token counterfactual
            prompts.append({
                "scenario_id": scenario_id,
                "type": "multi_token_counterfactual",
                "task": self.task_name,
                "text": scenario["prompts"]["multi_counterfactual"],
            })
        
        return prompts
    
    def get_stats(self) -> Dict:
        """Get generation statistics.
        
        Returns:
            Dictionary with API usage stats
        """
        return self.api_client.get_stats()

