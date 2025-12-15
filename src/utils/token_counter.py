# -*- coding: utf-8 -*-
"""
Token counting and cost estimation for LLM API calls.

Provides token counting using tiktoken and cost estimation for different Claude
models with budget tracking for API calls. Supports input/output token tracking
and per-model pricing calculation.

Example:
    counter = TokenCounter(model='claude-sonnet')
    tokens = counter.count_tokens("example prompt")
    cost = counter.estimate_cost(input_tokens=1000, output_tokens=500)
"""

# Standard library
import sys
from pathlib import Path
from typing import Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import tiktoken

# Local
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# PRICING CONSTANTS (USD per 1M tokens)
# ============================================================================

# Anthropic Claude pricing (as of Dec 2024)
PRICING = {
    'claude-opus': {
        'input': 15.00,    # $15 per 1M input tokens
        'output': 75.00    # $75 per 1M output tokens
    },
    'claude-sonnet': {
        'input': 3.00,     # $3 per 1M input tokens
        'output': 15.00    # $15 per 1M output tokens
    },
    'claude-haiku': {
        'input': 0.80,     # $0.80 per 1M input tokens
        'output': 4.00     # $4 per 1M output tokens
    }
}


# ============================================================================
# TOKEN COUNTING FUNCTIONS
# ============================================================================

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for.
        model: Encoding model (default: cl100k_base for GPT-4/Claude).
        
    Returns:
        Number of tokens.
        
    Note:
        Claude uses a similar tokenizer to GPT-4, so cl100k_base
        provides reasonable approximations. Actual token counts
        may vary slightly.
    """
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning("Token counting failed: %s. Using character estimate.", str(e))
        # Fallback: ~4 chars per token
        return len(text) // 4


# ============================================================================
# TOKEN COUNTER CLASS
# ============================================================================

class TokenCounter:
    """
    Track token usage and estimate costs for LLM API calls.
    
    Provides:
    - Token counting for prompts
    - Cost estimation for different models
    - Budget tracking across multiple calls
    """
    
    def __init__(self):
        """Initialize token counter."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info("TokenCounter initialized")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count.
            
        Returns:
            Number of tokens.
        """
        return count_tokens(text)
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-haiku"
    ) -> float:
        """
        Estimate cost for a single API call.
        
        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name (claude-opus, claude-sonnet, claude-haiku).
            
        Returns:
            Estimated cost in USD.
        """
        if model not in PRICING:
            logger.warning("Unknown model '%s', using claude-haiku pricing", model)
            model = "claude-haiku"
        
        pricing = PRICING[model]
        
        # Cost per million tokens
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        
        total = input_cost + output_cost
        
        return total
    
    def track_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-haiku"
    ) -> dict:
        """
        Track a single API call and update totals.
        
        Args:
            input_tokens: Input tokens used.
            output_tokens: Output tokens used.
            model: Model used.
            
        Returns:
            Dict with call cost and running totals.
        """
        cost = self.estimate_cost(input_tokens, output_tokens, model)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        
        logger.info(
            "Call tracked: %d in + %d out = $%.4f (Total: $%.4f)",
            input_tokens, output_tokens, cost, self.total_cost
        )
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'call_cost': cost,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost': self.total_cost
        }
    
    def get_totals(self) -> dict:
        """
        Get total token usage and cost.
        
        Returns:
            Dict with total_input_tokens, total_output_tokens, total_cost.
        """
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost': self.total_cost
        }
    
    def reset(self):
        """Reset all counters to zero."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        logger.info("TokenCounter reset")
    
    def estimate_prompt_cost(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        model: str = "claude-haiku"
    ) -> dict:
        """
        Estimate cost for a prompt before making API call.
        
        Args:
            system_prompt: System prompt text.
            user_prompt: User prompt text.
            max_output_tokens: Expected output tokens.
            model: Model to use.
            
        Returns:
            Dict with estimated tokens and cost.
        """
        input_tokens = (
            self.count_tokens(system_prompt) +
            self.count_tokens(user_prompt)
        )
        
        cost = self.estimate_cost(input_tokens, max_output_tokens, model)
        
        logger.debug(
            "Prompt estimate: %d in + %d out = $%.4f",
            input_tokens, max_output_tokens, cost
        )
        
        return {
            'estimated_input_tokens': input_tokens,
            'estimated_output_tokens': max_output_tokens,
            'estimated_cost': cost,
            'model': model
        }