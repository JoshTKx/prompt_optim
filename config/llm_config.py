"""LLM configuration and API key management."""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """Configuration for LLM providers and models via OpenRouter."""
    
    # OpenRouter API Key (single key for all models)
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    
    # OpenRouter Base URL
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Model Selection (OpenRouter format: provider/model-name)
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "deepseek/deepseek-v3.2")
    REVISER_MODEL: str = os.getenv("REVISER_MODEL", "anthropic/claude-4.5-sonnet")
    TARGET_MODEL: str = os.getenv("TARGET_MODEL", "google/gemini-3-flash-preview")
    CHECKER_MODEL: str = os.getenv("CHECKER_MODEL", "deepseek/deepseek-v3.2")
    
    # Optional headers for OpenRouter
    HTTP_REFERER: Optional[str] = os.getenv("HTTP_REFERER")  # Your site URL
    X_TITLE: Optional[str] = os.getenv("X_TITLE")  # Your site name
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are present."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required")
        return True
    
    @classmethod
    def get_headers(cls) -> dict:
        """Get headers for OpenRouter API requests."""
        headers = {
            "Authorization": f"Bearer {cls.OPENROUTER_API_KEY}",
        }
        if cls.HTTP_REFERER:
            headers["HTTP-Referer"] = cls.HTTP_REFERER
        if cls.X_TITLE:
            headers["X-Title"] = cls.X_TITLE
        return headers
