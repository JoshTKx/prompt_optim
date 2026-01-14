"""Dynamic Golden Set - automatically capture and use successful examples."""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
from utils.logging_utils import setup_logging
from utils.metrics import get_metrics_collector

logger = setup_logging()
metrics = get_metrics_collector()


@dataclass
class GoldenExample:
    """A golden example from a successful test case."""
    example_id: str
    test_case_id: str
    prompt: str
    input: str
    output: str
    score: float
    captured_at: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['captured_at'] = self.captured_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldenExample':
        """Create from dictionary."""
        data['captured_at'] = datetime.fromisoformat(data['captured_at'])
        return cls(**data)
    
    def similarity_hash(self) -> str:
        """Compute hash for similarity checking."""
        content = f"{self.test_case_id}:{self.input}:{self.output}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class GoldenSetManager:
    """Manages dynamic golden set of successful examples."""
    
    def __init__(self, storage_path: Optional[Path] = None, min_score: float = 85.0):
        """
        Initialize golden set manager.
        
        Args:
            storage_path: Path to store golden examples
            min_score: Minimum score to capture (default: 85.0)
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "outputs" / "golden_sets"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.min_score = min_score
        self._examples: Dict[str, List[GoldenExample]] = {}  # test_case_id -> examples
        self._load_existing()
    
    def should_capture(
        self,
        test_case_id: str,
        score: float,
        output: str,
        novelty_threshold: float = 0.3
    ) -> bool:
        """
        Determine if an example should be captured.
        
        Criteria:
        1. Score >= min_score (excellent performance)
        2. Not too similar to existing examples (novel)
        3. Not already in golden set (avoid duplicates)
        
        Args:
            test_case_id: Test case identifier
            score: Output score
            output: Output text
            novelty_threshold: Minimum similarity difference to consider novel
        
        Returns:
            True if should capture
        """
        # Must meet minimum score
        if score < self.min_score:
            return False
        
        # Check for novelty
        existing = self._examples.get(test_case_id, [])
        if not existing:
            return True  # First example for this test case
        
        # Check similarity to existing examples
        output_hash = hashlib.sha256(output.encode()).hexdigest()[:16]
        for ex in existing:
            if ex.similarity_hash() == output_hash:
                return False  # Too similar, skip
        
        # Limit examples per test case (prevent bloat)
        if len(existing) >= 5:
            # Only capture if significantly better than worst existing
            worst_score = min(ex.score for ex in existing)
            if score <= worst_score:
                return False
        
        return True
    
    def capture(
        self,
        test_case_id: str,
        prompt: str,
        input_text: str,
        output: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[GoldenExample]:
        """
        Capture a golden example if it meets criteria.
        
        Returns:
            GoldenExample if captured, None otherwise
        """
        if not self.should_capture(test_case_id, score, output):
            return None
        
        example_id = f"{test_case_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        example = GoldenExample(
            example_id=example_id,
            test_case_id=test_case_id,
            prompt=prompt,
            input=input_text,
            output=output,
            score=score,
            captured_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Add to collection
        if test_case_id not in self._examples:
            self._examples[test_case_id] = []
        self._examples[test_case_id].append(example)
        
        # Persist
        self._save_examples(test_case_id)
        
        logger.info(
            "Golden example captured",
            example_id=example_id,
            test_case_id=test_case_id,
            score=score
        )
        metrics.increment("golden_set.captured", tags={"test_case": test_case_id})
        
        return example
    
    def get_examples(
        self,
        test_case_id: str,
        limit: int = 3,
        min_score: Optional[float] = None
    ) -> List[GoldenExample]:
        """
        Get golden examples for a test case.
        
        Args:
            test_case_id: Test case identifier
            limit: Maximum number of examples
            min_score: Optional minimum score filter
        
        Returns:
            List of golden examples (sorted by score, descending)
        """
        examples = self._examples.get(test_case_id, [])
        
        if min_score:
            examples = [ex for ex in examples if ex.score >= min_score]
        
        # Sort by score descending
        examples.sort(key=lambda x: x.score, reverse=True)
        
        return examples[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get golden set statistics."""
        total_examples = sum(len(examples) for examples in self._examples.values())
        avg_score = 0.0
        if total_examples > 0:
            all_scores = [
                ex.score
                for examples in self._examples.values()
                for ex in examples
            ]
            avg_score = sum(all_scores) / len(all_scores)
        
        return {
            "total_examples": total_examples,
            "test_cases_covered": len(self._examples),
            "average_score": round(avg_score, 2),
            "min_score_threshold": self.min_score,
            "by_test_case": {
                tc_id: len(examples)
                for tc_id, examples in self._examples.items()
            }
        }
    
    def _load_existing(self):
        """Load existing golden examples from storage."""
        for json_file in self.storage_path.glob("*.json"):
            test_case_id = json_file.stem
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    examples = [GoldenExample.from_dict(ex) for ex in data.get('examples', [])]
                    self._examples[test_case_id] = examples
            except Exception as e:
                logger.error(f"Error loading golden set {json_file}: {e}")
    
    def _save_examples(self, test_case_id: str):
        """Save examples for a test case."""
        examples = self._examples.get(test_case_id, [])
        json_file = self.storage_path / f"{test_case_id}.json"
        data = {
            "test_case_id": test_case_id,
            "examples": [ex.to_dict() for ex in examples],
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
