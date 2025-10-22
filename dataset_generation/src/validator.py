"""Token validation module for ensuring prompt/counterfactual token constraints."""

from typing import List, Tuple, Dict
from transformers import AutoTokenizer
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenValidator:
    """Validates token-level constraints across multiple tokenizers."""
    
    def __init__(self, cache_dir: str = None):
        """Initialize tokenizers for Qwen2, Llama-3, and SOLAR.
        
        Args:
            cache_dir: Directory to cache downloaded tokenizers. If None, uses
                      a local .cache directory in the project.
        """
        logger.info("Loading tokenizers...")
        
        # Set cache directory to avoid permission issues with default HF cache
        if cache_dir is None:
            # Use local cache directory in the project
            project_root = Path(__file__).parent.parent
            cache_dir = str(project_root / ".cache")
            os.makedirs(cache_dir, exist_ok=True)
        
        # Set HuggingFace environment variables
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        logger.info(f"Using cache directory: {cache_dir}")
        
        # Load the three required tokenizers
        self.tokenizers = {
            "qwen2": AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat", cache_dir=cache_dir),
            "llama3": AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", cache_dir=cache_dir),
            "solar": AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-v1.0", cache_dir=cache_dir),
        }
        
        logger.info("Tokenizers loaded successfully")
    
    def tokenize_all(self, text: str) -> Dict[str, List[int]]:
        """Tokenize text with all tokenizers.
        
        Args:
            text: The text to tokenize
            
        Returns:
            Dictionary mapping tokenizer name to token IDs
        """
        return {
            name: tokenizer.encode(text, add_special_tokens=False)
            for name, tokenizer in self.tokenizers.items()
        }
    
    def validate_equal_token_counts(self, text1: str, text2: str) -> Tuple[bool, Dict[str, Tuple[int, int]]]:
        """Validate that two texts have equal token counts across all tokenizers.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Tuple of (is_valid, token_counts_dict) where token_counts_dict maps
            tokenizer name to (text1_count, text2_count)
        """
        tokens1 = self.tokenize_all(text1)
        tokens2 = self.tokenize_all(text2)
        
        token_counts = {}
        is_valid = True
        
        for name in self.tokenizers.keys():
            count1 = len(tokens1[name])
            count2 = len(tokens2[name])
            token_counts[name] = (count1, count2)
            
            if count1 != count2:
                is_valid = False
        
        return is_valid, token_counts
    
    def validate_one_token_difference(self, text1: str, text2: str) -> Tuple[bool, Dict[str, int]]:
        """Validate that two texts differ in exactly one token across all tokenizers.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Tuple of (is_valid, diff_counts_dict) where diff_counts_dict maps
            tokenizer name to number of differing tokens
        """
        tokens1 = self.tokenize_all(text1)
        tokens2 = self.tokenize_all(text2)
        
        diff_counts = {}
        is_valid = True
        
        for name in self.tokenizers.keys():
            t1 = tokens1[name]
            t2 = tokens2[name]
            
            # Count differing tokens
            if len(t1) != len(t2):
                # If lengths differ, we can't have exactly one token difference
                diff_counts[name] = abs(len(t1) - len(t2))
                is_valid = False
            else:
                diff_count = sum(1 for a, b in zip(t1, t2) if a != b)
                diff_counts[name] = diff_count
                
                if diff_count != 1:
                    is_valid = False
        
        return is_valid, diff_counts
    
    def validate_pair(self, text1: str, text2: str) -> Tuple[bool, Dict[str, any]]:
        """Validate both constraints: equal token counts and exactly one token difference.
        
        Args:
            text1: First text (prompt)
            text2: Second text (counterfactual)
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        # Check equal token counts
        equal_counts, token_counts = self.validate_equal_token_counts(text1, text2)
        
        # Check one token difference
        one_diff, diff_counts = self.validate_one_token_difference(text1, text2)
        
        is_valid = equal_counts and one_diff
        
        validation_info = {
            "equal_token_counts": equal_counts,
            "one_token_difference": one_diff,
            "token_counts": token_counts,
            "diff_counts": diff_counts,
        }
        
        if not is_valid:
            logger.debug(f"Validation failed: {validation_info}")
        
        return is_valid, validation_info
    
    def validate_scenario(self, safe_task: str, harmful_task: str, 
                         single_token_prefix: str, single_token_suffix: str,
                         multi_token_prefix: str, multi_token_suffix: str) -> Tuple[bool, Dict]:
        """Validate an entire scenario (both single-token and multi-token variants).
        
        Args:
            safe_task: The safe task text
            harmful_task: The harmful task text
            single_token_prefix: Prefix for single-token prompts
            single_token_suffix: Suffix for single-token prompts
            multi_token_prefix: Prefix for multi-token prompts
            multi_token_suffix: Suffix for multi-token prompts
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        # Create single-token variants
        single_prompt = f"{single_token_prefix}{safe_task}.{single_token_suffix}"
        single_counterfactual = f"{single_token_prefix}{harmful_task}.{single_token_suffix}"
        
        # Create multi-token variants
        multi_prompt = f"{multi_token_prefix}{safe_task}.{multi_token_suffix}"
        multi_counterfactual = f"{multi_token_prefix}{harmful_task}.{multi_token_suffix}"
        
        # Validate single-token pair
        single_valid, single_info = self.validate_pair(single_prompt, single_counterfactual)
        
        # Validate multi-token pair
        multi_valid, multi_info = self.validate_pair(multi_prompt, multi_counterfactual)
        
        is_valid = single_valid and multi_valid
        
        validation_details = {
            "single_token_valid": single_valid,
            "multi_token_valid": multi_valid,
            "single_token_info": single_info,
            "multi_token_info": multi_info,
            "prompts": {
                "single_prompt": single_prompt,
                "single_counterfactual": single_counterfactual,
                "multi_prompt": multi_prompt,
                "multi_counterfactual": multi_counterfactual,
            }
        }
        
        return is_valid, validation_details

