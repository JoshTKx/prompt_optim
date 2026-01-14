"""Prompt versioning and change tracking."""
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import difflib


@dataclass
class PromptVersion:
    """A versioned prompt with metadata."""
    version: int
    prompt_text: str
    prompt_hash: str
    created_at: datetime
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = None
    parent_version: Optional[int] = None
    change_summary: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.prompt_hash is None:
            self.prompt_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of prompt."""
        return hashlib.sha256(self.prompt_text.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptVersion':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class PromptVersionManager:
    """Manages prompt versions and tracks changes."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize version manager.
        
        Args:
            storage_path: Path to store version history (defaults to outputs/prompt_versions)
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "outputs" / "prompt_versions"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._versions: Dict[str, List[PromptVersion]] = {}  # prompt_id -> versions
    
    def create_version(
        self,
        prompt_id: str,
        prompt_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
        change_summary: Optional[str] = None
    ) -> PromptVersion:
        """
        Create a new prompt version.
        
        Args:
            prompt_id: Unique identifier for the prompt
            prompt_text: The prompt text
            metadata: Optional metadata
            created_by: Optional creator identifier
            change_summary: Optional summary of changes
        
        Returns:
            PromptVersion instance
        """
        # Get existing versions
        versions = self._get_versions(prompt_id)
        
        # Determine version number
        version_num = len(versions) + 1
        
        # Get parent version
        parent_version = versions[-1].version if versions else None
        
        # Create new version
        version = PromptVersion(
            version=version_num,
            prompt_text=prompt_text,
            prompt_hash=None,  # Will be computed in __post_init__
            created_at=datetime.utcnow(),
            created_by=created_by,
            metadata=metadata or {},
            parent_version=parent_version,
            change_summary=change_summary
        )
        
        # Add to versions
        versions.append(version)
        self._versions[prompt_id] = versions
        
        # Persist
        self._save_versions(prompt_id, versions)
        
        return version
    
    def get_version(self, prompt_id: str, version: int) -> Optional[PromptVersion]:
        """Get specific version of a prompt."""
        versions = self._get_versions(prompt_id)
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def get_latest(self, prompt_id: str) -> Optional[PromptVersion]:
        """Get latest version of a prompt."""
        versions = self._get_versions(prompt_id)
        return versions[-1] if versions else None
    
    def get_history(self, prompt_id: str) -> List[PromptVersion]:
        """Get all versions of a prompt."""
        return self._get_versions(prompt_id)
    
    def diff_versions(
        self,
        prompt_id: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """
        Get diff between two versions.
        
        Returns:
            Dictionary with diff information
        """
        v1 = self.get_version(prompt_id, version1)
        v2 = self.get_version(prompt_id, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        lines1 = v1.prompt_text.splitlines(keepends=True)
        lines2 = v2.prompt_text.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            lines1, lines2,
            fromfile=f"v{version1}",
            tofile=f"v{version2}",
            lineterm=''
        ))
        
        return {
            "from_version": version1,
            "to_version": version2,
            "from_hash": v1.prompt_hash,
            "to_hash": v2.prompt_hash,
            "diff": diff,
            "lines_changed": len([l for l in diff if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]),
            "created_at_diff": (v2.created_at - v1.created_at).total_seconds()
        }
    
    def _get_versions(self, prompt_id: str) -> List[PromptVersion]:
        """Load versions from storage."""
        if prompt_id in self._versions:
            return self._versions[prompt_id]
        
        # Load from file
        version_file = self.storage_path / f"{prompt_id}.json"
        if version_file.exists():
            with open(version_file, 'r') as f:
                data = json.load(f)
                versions = [PromptVersion.from_dict(v) for v in data.get('versions', [])]
                self._versions[prompt_id] = versions
                return versions
        
        return []
    
    def _save_versions(self, prompt_id: str, versions: List[PromptVersion]):
        """Save versions to storage."""
        version_file = self.storage_path / f"{prompt_id}.json"
        data = {
            "prompt_id": prompt_id,
            "versions": [v.to_dict() for v in versions],
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(version_file, 'w') as f:
            json.dump(data, f, indent=2)
