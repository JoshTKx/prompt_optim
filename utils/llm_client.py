"""Unified LLM client interface via OpenRouter."""
import json
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
from config.llm_config import LLMConfig
from utils.logging_utils import setup_logging, get_correlation_id
from utils.metrics import get_metrics_collector
from utils.error_handling import handle_errors, ErrorSeverity, retry_with_backoff, RetryConfig

logger = setup_logging()
metrics = get_metrics_collector()


class LLMClient:
    """Unified interface for calling LLM providers via OpenRouter."""
    
    def __init__(self):
        """Initialize OpenRouter client."""
        if not LLMConfig.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        # OpenRouter uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=LLMConfig.OPENROUTER_API_KEY,
            base_url=LLMConfig.OPENROUTER_BASE_URL,
            default_headers=LLMConfig.get_headers()
        )
        
        # Track total cost
        self.total_cost: float = 0.0
    
    @handle_errors(severity=ErrorSeverity.HIGH, log_error=True, reraise=True)
    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=60.0,
            retryable_exceptions=(Exception,)
        )
    )
    def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False
    ) -> str:
        """
        Generate completion from LLM.
        
        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022", "deepseek-chat")
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Whether to force JSON output (OpenAI only)
        
        Returns:
            Generated text
        """
        start_time = time.time()
        corr_id = get_correlation_id()
        
        logger.debug(
            "LLM completion request",
            model=model,
            prompt_length=len(prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            correlation_id=corr_id
        )
        metrics.increment("llm.requests", tags={"model": model})
        
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # JSON mode (OpenRouter supports this for compatible models)
            if json_mode:
                # Check if model supports JSON mode (OpenAI models typically do)
                if "openai" in model.lower() or "gpt" in model.lower():
                    kwargs["response_format"] = {"type": "json_object"}
            
            # Make request via OpenRouter
            response = self.client.chat.completions.create(**kwargs)
            
            # Extract response
            if not response or not response.choices or len(response.choices) == 0:
                raise ValueError(f"No response from model {model}")
            
            result = response.choices[0].message.content
            if result is None:
                raise ValueError(f"Empty response from model {model}")
            
            # Log usage if available and calculate cost
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                cost = self._calculate_cost(usage, model)
                self.total_cost += cost
                
                logger.debug(
                    "LLM usage",
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=round(cost, 4),
                    total_cost_usd=round(self.total_cost, 4)
                )
                metrics.record("llm.tokens.prompt", usage.prompt_tokens, tags={"model": model})
                metrics.record("llm.tokens.completion", usage.completion_tokens, tags={"model": model})
                metrics.record("llm.cost", cost, tags={"model": model})
            
            return result
        
        finally:
            duration = time.time() - start_time
            metrics.histogram("llm.duration", duration, tags={"model": model})
            logger.debug("LLM completion finished", model=model, duration_seconds=duration)
    
    def _calculate_cost(self, usage, model: str) -> float:
        """
        Calculate actual cost based on usage from API response.
        
        Args:
            usage: Usage object from API response (with prompt_tokens, completion_tokens)
            model: Model identifier
        
        Returns:
            Cost in USD
        """
        prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
        completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
        
        return self.estimate_cost(model, prompt_tokens, completion_tokens)
    
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost in USD for a completion via OpenRouter.
        
        Pricing (OpenRouter pricing, approximate as of 2024):
        - openai/gpt-4o-mini: ~$0.15/$0.60 per 1M tokens (input/output)
        - anthropic/claude-3.5-sonnet: ~$3/$15 per 1M tokens
        - deepseek/deepseek-chat: ~$0.14/$0.56 per 1M tokens
        
        Note: OpenRouter pricing may vary. Check https://openrouter.ai/models for current rates.
        """
        # OpenRouter model pricing (provider/model format)
        pricing = {
            "openai/gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
            "openai/gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
            "anthropic/claude-3.5-sonnet": (3.0 / 1_000_000, 15.0 / 1_000_000),
            "anthropic/claude-sonnet-4.5": (3.0 / 1_000_000, 15.0 / 1_000_000),  # Approximate
            "anthropic/claude-3-opus": (15.0 / 1_000_000, 75.0 / 1_000_000),
            "deepseek/deepseek-chat": (0.14 / 1_000_000, 0.56 / 1_000_000),
            "deepseek/deepseek-v3": (0.55 / 1_000_000, 2.19 / 1_000_000),
            "deepseek/deepseek-v3.2": (0.55 / 1_000_000, 2.19 / 1_000_000),  # Approximate
            "google/gemini-pro": (0.50 / 1_000_000, 1.50 / 1_000_000),  # Approximate
            "google/gemini-flash": (0.075 / 1_000_000, 0.30 / 1_000_000),  # Approximate
            "google/gemini-flash-1.5": (0.075 / 1_000_000, 0.30 / 1_000_000),  # Approximate
            "google/gemini-3-flash-preview": (0.075 / 1_000_000, 0.30 / 1_000_000),  # Approximate
        }
        
        # Find matching model
        price_tuple = pricing.get(model)
        
        if price_tuple is None:
            # Try to match by provider/model pattern (more flexible matching)
            for model_key, (in_p, out_p) in pricing.items():
                # Match if model starts with the key, or key is a prefix of model
                if model.startswith(model_key) or model_key in model:
                    price_tuple = (in_p, out_p)
                    break
        
        if price_tuple is None:
            # Default to conservative estimate (GPT-4o-mini pricing)
            logger.warning(f"Unknown model pricing for {model}, using default")
            price_tuple = (0.15 / 1_000_000, 0.60 / 1_000_000)
        
        input_price, output_price = price_tuple
        cost = (prompt_tokens * input_price) + (completion_tokens * output_price)
        return cost
