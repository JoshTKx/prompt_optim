"""System prompt versioning and management."""
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from utils.prompt_versioning import PromptVersionManager


class SystemPromptManager:
    """Manages versioning of system prompts."""
    
    def __init__(self, version_manager: Optional[PromptVersionManager] = None):
        """
        Initialize system prompt manager.
        
        Args:
            version_manager: Optional PromptVersionManager instance
        """
        self.version_manager = version_manager or PromptVersionManager()
        self._prompt_cache: Dict[str, str] = {}
        self._prompt_hashes: Dict[str, str] = {}
    
    def load_prompt(
        self,
        prompt_name: str,
        prompt_path: Path,
        version: Optional[int] = None
    ) -> tuple[str, int, str]:
        """
        Load a system prompt with versioning.
        
        Args:
            prompt_name: Name identifier for the prompt
            prompt_path: Path to prompt file
            version: Optional specific version to load (defaults to latest)
        
        Returns:
            Tuple of (prompt_text, version_number, prompt_hash)
        """
        # Read prompt file
        with open(prompt_path, 'r') as f:
            prompt_text = f.read()
        
        # Compute hash
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
        
        # Check if this is a new version
        latest = self.version_manager.get_latest(prompt_name)
        
        if latest is None or latest.prompt_hash != prompt_hash:
            # New version detected
            new_version = self.version_manager.create_version(
                prompt_id=prompt_name,
                prompt_text=prompt_text,
                metadata={
                    "file_path": str(prompt_path),
                    "file_mtime": datetime.fromtimestamp(prompt_path.stat().st_mtime).isoformat()
                },
                change_summary="System prompt file updated"
            )
            version_num = new_version.version
        else:
            # Existing version
            version_num = latest.version
        
        # Cache
        self._prompt_cache[prompt_name] = prompt_text
        self._prompt_hashes[prompt_name] = prompt_hash
        
        return prompt_text, version_num, prompt_hash
    
    def get_prompt_version(self, prompt_name: str) -> Optional[int]:
        """Get current version number for a prompt."""
        latest = self.version_manager.get_latest(prompt_name)
        return latest.version if latest else None
    
    def get_prompt_history(self, prompt_name: str) -> list:
        """Get version history for a prompt."""
        return self.version_manager.get_history(prompt_name)
