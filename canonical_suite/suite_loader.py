"""Loader for canonical test suite."""
import json
from pathlib import Path
from typing import List, Dict, Any
from models.test_case import CanonicalTestCase
from utils.logging_utils import setup_logging

logger = setup_logging()


class CanonicalTestSuite:
    """Manages the canonical test suite."""
    
    def __init__(self, test_cases: List[CanonicalTestCase]):
        self.test_cases = test_cases
    
    @classmethod
    def load(cls, test_dir: str) -> "CanonicalTestSuite":
        """
        Load test cases from directory.
        
        Args:
            test_dir: Path to test cases directory
        
        Returns:
            CanonicalTestSuite instance
        """
        test_path = Path(test_dir)
        test_cases = []
        
        # Load all JSON files recursively
        for json_file in test_path.rglob("*.json"):
            # Skip manifest files
            if "MANIFEST" in json_file.name.upper():
                logger.debug(f"Skipping manifest file: {json_file}")
                continue
            
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    
                    # Skip if this looks like a manifest (has 'total_tests' or 'tests' but not 'id')
                    if ("total_tests" in data or "tests" in data) and "id" not in data:
                        logger.debug(f"Skipping manifest-like file: {json_file}")
                        continue
                    
                    test_case = CanonicalTestCase(**data)
                    test_cases.append(test_case)
                    logger.debug(f"Loaded test case: {test_case.id}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return cls(test_cases)
    
    def get_by_category(self, category: str) -> List[CanonicalTestCase]:
        """Get test cases by category."""
        return [tc for tc in self.test_cases if tc.category == category]
    
    def get_by_archetype(self, archetype: str) -> List[CanonicalTestCase]:
        """Get test cases by archetype."""
        return [tc for tc in self.test_cases if tc.archetype.startswith(archetype)]
    
    def get_by_difficulty(self, difficulty: str) -> List[CanonicalTestCase]:
        """Get test cases by difficulty."""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __iter__(self):
        return iter(self.test_cases)
