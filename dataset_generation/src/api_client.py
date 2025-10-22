"""OpenAI API client with retry logic and cost tracking."""

import os
import json
import logging
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API with retry logic and cost tracking."""
    
    def __init__(self, model: str = "gpt-4o"):
        """Initialize the OpenAI client.
        
        Args:
            model: The model to use for generation
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it with: export OPENAI_API_KEY='your-api-key'"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Approximate costs per 1K tokens (as of 2024)
        self.cost_per_1k_input = 0.0025  # $0.0025 per 1K input tokens for GPT-4o
        self.cost_per_1k_output = 0.01   # $0.01 per 1K output tokens for GPT-4o
        
        logger.info(f"OpenAI client initialized with model: {model}")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    )
    def generate_scenarios(self, task_config: Dict, num_scenarios: int) -> List[Dict[str, str]]:
        """Generate multiple safe/harmful task pairs using GPT-4o.
        
        Args:
            task_config: Task configuration dictionary
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of dictionaries with 'safe' and 'harmful' keys
        """
        examples_str = "\n".join([
            f"- Safe: \"{ex['safe']}\"\n  Harmful: \"{ex['harmful']}\""
            for ex in task_config["examples"]
        ])
        
        system_prompt = f"""You are a helpful assistant that generates pairs of safe and harmful requests for AI safety research.

{task_config['generation_instructions']}

Examples:
{examples_str}

Generate {num_scenarios} new pairs that follow the same pattern. Return them as a JSON array with objects containing 'safe' and 'harmful' keys."""

        user_prompt = f"Generate {num_scenarios} diverse safe/harmful task pairs following the guidelines and examples provided."
        
        logger.info(f"Requesting {num_scenarios} scenarios from OpenAI API...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            
            # Track usage and cost
            usage = response.usage
            input_cost = (usage.prompt_tokens / 1000) * self.cost_per_1k_input
            output_cost = (usage.completion_tokens / 1000) * self.cost_per_1k_output
            call_cost = input_cost + output_cost
            
            self.total_tokens += usage.total_tokens
            self.total_cost += call_cost
            
            logger.info(
                f"API call successful. Tokens: {usage.total_tokens}, "
                f"Cost: ${call_cost:.4f}, Total cost: ${self.total_cost:.4f}"
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Handle different possible JSON structures
            if isinstance(result, dict):
                if "pairs" in result:
                    scenarios = result["pairs"]
                elif "scenarios" in result:
                    scenarios = result["scenarios"]
                else:
                    # Assume the dict itself is a single scenario
                    scenarios = [result]
            elif isinstance(result, list):
                scenarios = result
            else:
                raise ValueError(f"Unexpected response format: {type(result)}")
            
            # Validate structure
            valid_scenarios = []
            for scenario in scenarios:
                if isinstance(scenario, dict) and "safe" in scenario and "harmful" in scenario:
                    valid_scenarios.append({
                        "safe": scenario["safe"],
                        "harmful": scenario["harmful"]
                    })
                else:
                    logger.warning(f"Skipping malformed scenario: {scenario}")
            
            logger.info(f"Successfully parsed {len(valid_scenarios)} valid scenarios")
            return valid_scenarios
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {content}")
            raise
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get usage statistics.
        
        Returns:
            Dictionary with total tokens and cost
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }

