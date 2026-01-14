"""Golden Set - stores high-scoring test cases with best prompts for regression prevention."""
import json
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from utils.logging_utils import setup_logging
from utils.metrics import get_metrics_collector
from config.optimization_config import OptimizationConfig

logger = setup_logging()
metrics = get_metrics_collector()


@dataclass
class GoldenSetEntry:
    """A golden set entry storing the best prompt for a test case."""
    test_case_id: str
    test_input: Dict[str, Any]  # Original test input
    expected_output: Dict[str, Any]  # Output that scored >= threshold
    best_prompt: str  # Complete prompt text that achieved this score
    score: float  # Score achieved by best_prompt
    threshold: float  # Threshold used (default 85.0)
    timestamp: str  # ISO format timestamp
    optimization_run_id: str  # ID of the optimization run that produced this
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldenSetEntry':
        """Create from dictionary."""
        return cls(**data)


class GoldenSetManager:
    """Manages golden set of high-scoring test cases with best prompts."""
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        threshold: float = None,
        regression_enabled: bool = None,
        regression_pass_rate: float = None
    ):
        """
        Initialize golden set manager.
        
        Args:
            storage_path: Path to store golden set JSON file
            threshold: Minimum score to add to golden set (defaults to config)
            regression_enabled: Enable regression testing (defaults to config)
            regression_pass_rate: Minimum pass rate for regression (defaults to config)
        """
        # Use config defaults if not provided
        self.threshold = threshold or OptimizationConfig.GOLDEN_SET_THRESHOLD
        self.regression_enabled = regression_enabled if regression_enabled is not None else OptimizationConfig.REGRESSION_CHECK_ENABLED
        self.regression_pass_rate = regression_pass_rate or OptimizationConfig.REGRESSION_PASS_RATE_THRESHOLD
        
        # Set up storage path
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / OptimizationConfig.GOLDEN_SET_PATH
        else:
            storage_path = Path(storage_path)
        
        # Ensure directory exists
        self.storage_path = storage_path.parent if storage_path.suffix == '.json' else storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Set file path
        if storage_path.suffix == '.json':
            self.golden_set_file = storage_path
        else:
            self.golden_set_file = self.storage_path / "golden_set.json"
        
        # Load existing golden set
        self._entries: Dict[str, GoldenSetEntry] = {}  # test_case_id -> entry
        self._load_existing()
        
        logger.info(
            "GoldenSetManager initialized",
            threshold=self.threshold,
            regression_enabled=self.regression_enabled,
            total_entries=len(self._entries),
            storage_path=str(self.golden_set_file)
        )
    
    def _load_existing(self):
        """Load existing golden set from disk."""
        if not self.golden_set_file.exists():
            logger.debug("Golden set file does not exist, starting fresh", path=str(self.golden_set_file))
            return
        
        try:
            with open(self.golden_set_file, 'r') as f:
                data = json.load(f)
            
            # Load entries
            entries_data = data.get('entries', [])
            for entry_data in entries_data:
                entry = GoldenSetEntry.from_dict(entry_data)
                self._entries[entry.test_case_id] = entry
            
            logger.info(
                "Golden set loaded",
                total_entries=len(self._entries),
                version=data.get('version', 'unknown'),
                last_updated=data.get('last_updated', 'unknown')
            )
        except Exception as e:
            logger.error(f"Error loading golden set: {e}", path=str(self.golden_set_file))
            self._entries = {}
    
    def _save_to_disk(self):
        """Save golden set to disk with backup."""
        try:
            # Create backup before saving
            if self.golden_set_file.exists():
                backup_path = self.storage_path / f"golden_set_backup_{datetime.now().strftime('%Y%m%d')}.json"
                shutil.copy2(self.golden_set_file, backup_path)
                logger.debug("Created golden set backup", backup_path=str(backup_path))
            
            # Save current state
            data = {
                "version": "1.0",
                "total_entries": len(self._entries),
                "threshold": self.threshold,
                "last_updated": datetime.now().isoformat(),
                "entries": [entry.to_dict() for entry in self._entries.values()]
            }
            
            with open(self.golden_set_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Golden set saved", path=str(self.golden_set_file), entries=len(self._entries))
        except Exception as e:
            logger.error(f"Error saving golden set: {e}", path=str(self.golden_set_file))
    
    def add_or_update(
        self,
        test_case_id: str,
        test_input: Dict[str, Any],
        expected_output: Dict[str, Any],
        best_prompt: str,
        score: float,
        optimization_run_id: str
    ) -> str:
        """
        Add or update a golden set entry.
        
        Only adds if score >= threshold.
        If entry exists, updates only if new score > existing score.
        
        Args:
            test_case_id: Test case identifier
            test_input: Original test input
            expected_output: Output that achieved the score
            best_prompt: Prompt that achieved this score
            score: Score achieved
            optimization_run_id: ID of the optimization run
        
        Returns:
            "added", "updated", or "skipped" (with reason)
        """
        # Check threshold
        if score < self.threshold:
            logger.debug(
                "Golden set: Skipped (below threshold)",
                test_case_id=test_case_id,
                score=score,
                threshold=self.threshold
            )
            return f"skipped (score {score:.1f} < threshold {self.threshold:.1f})"
        
        # Check if entry exists
        existing = self._entries.get(test_case_id)
        
        if existing is None:
            # New entry
            entry = GoldenSetEntry(
                test_case_id=test_case_id,
                test_input=test_input,
                expected_output=expected_output,
                best_prompt=best_prompt,
                score=score,
                threshold=self.threshold,
                timestamp=datetime.now().isoformat(),
                optimization_run_id=optimization_run_id
            )
            self._entries[test_case_id] = entry
            self._save_to_disk()
            
            logger.info(
                "Golden set: Added",
                test_case_id=test_case_id,
                score=score
            )
            metrics.increment("golden_set.added", tags={"test_case": test_case_id})
            return "added"
        
        elif score > existing.score:
            # Update with better score
            entry = GoldenSetEntry(
                test_case_id=test_case_id,
                test_input=test_input,
                expected_output=expected_output,
                best_prompt=best_prompt,
                score=score,
                threshold=self.threshold,
                timestamp=datetime.now().isoformat(),
                optimization_run_id=optimization_run_id
            )
            old_score = existing.score
            self._entries[test_case_id] = entry
            self._save_to_disk()
            
            logger.info(
                "Golden set: Updated",
                test_case_id=test_case_id,
                old_score=old_score,
                new_score=score
            )
            metrics.increment("golden_set.updated", tags={"test_case": test_case_id})
            return "updated"
        
        else:
            # Keep existing (higher or equal score)
            logger.debug(
                "Golden set: Kept existing",
                test_case_id=test_case_id,
                existing_score=existing.score,
                new_score=score
            )
            return f"skipped (existing score {existing.score:.1f} >= new score {score:.1f})"
    
    def get_entry(self, test_case_id: str) -> Optional[GoldenSetEntry]:
        """Get golden set entry for a test case."""
        return self._entries.get(test_case_id)
    
    def get_all_entries(self) -> List[GoldenSetEntry]:
        """Get all golden set entries."""
        return list(self._entries.values())
    
    def test_regression(
        self,
        prompt: str,
        llm_client,
        judge,
        specification,
        cost_tracker,
        target_model: str = None
    ) -> Dict[str, Any]:
        """
        Test a prompt against golden set for regression.
        
        Args:
            prompt: Prompt to test
            llm_client: LLM client for testing
            judge: Judge instance
            specification: Output specification
            cost_tracker: Cost tracker instance
        
        Returns:
            Dict with passing_golden, total_golden, regression_pass_rate, results
        """
        if not self.regression_enabled or len(self._entries) == 0:
            return {
                "enabled": False,
                "passing_golden": 0,
                "total_golden": 0,
                "regression_pass_rate": 1.0,
                "results": []
            }
        
        results = []
        passing = 0
        total = len(self._entries)
        
        logger.info(
            "Running regression test on golden set",
            prompt_hash=hash(prompt) % 10000,
            total_cases=total
        )
        
        for entry in self._entries.values():
            try:
                # Extract test input from entry
                # test_input is stored as a dict with "input" key
                if isinstance(entry.test_input, dict):
                    test_input_str = entry.test_input.get("input", "")
                else:
                    # Fallback for old format
                    test_input_str = str(entry.test_input)
                
                if not test_input_str:
                    logger.warning(
                        "Golden set entry missing test input",
                        test_case_id=entry.test_case_id
                    )
                    results.append({
                        "test_case_id": entry.test_case_id,
                        "score": 0.0,
                        "passed": False,
                        "error": "Missing test input"
                    })
                    continue
                
                # Construct full prompt with test input
                full_prompt = f"{prompt}\n\nInput: {test_input_str}"
                
                # Generate output using LLM client directly
                model_to_use = target_model or (llm_client.target_model if hasattr(llm_client, 'target_model') else "openai/gpt-4o-mini")
                output = llm_client.complete(
                    model=model_to_use,
                    prompt=full_prompt,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Create a test result structure for the judge
                test_result = [{
                    "input": test_input_str,
                    "output": output,
                    "expected": entry.expected_output,
                    "success": True
                }]
                
                # Judge the result
                feedback = judge.evaluate_batch(test_result, specification)
                score = feedback.average_score
                
                passed = score >= self.threshold
                if passed:
                    passing += 1
                
                results.append({
                    "test_case_id": entry.test_case_id,
                    "score": score,
                    "passed": passed,
                    "threshold": self.threshold
                })
                
            except Exception as e:
                logger.warning(
                    "Error testing golden set entry",
                    test_case_id=entry.test_case_id,
                    error=str(e)
                )
                results.append({
                    "test_case_id": entry.test_case_id,
                    "score": 0.0,
                    "passed": False,
                    "error": str(e)
                })
        
        regression_pass_rate = passing / total if total > 0 else 1.0
        
        logger.info(
            "Regression check: {}/{} golden cases passed ({:.1%})".format(
                passing, total, regression_pass_rate
            ),
            passing=passing,
            total=total,
            pass_rate=regression_pass_rate
        )
        metrics.gauge("golden_set.regression_pass_rate", regression_pass_rate)
        
        return {
            "enabled": True,
            "passing_golden": passing,
            "total_golden": total,
            "regression_pass_rate": regression_pass_rate,
            "results": results
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get golden set statistics."""
        if len(self._entries) == 0:
            return {
                "total_entries": 0,
                "threshold": self.threshold,
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0
            }
        
        scores = [entry.score for entry in self._entries.values()]
        
        return {
            "total_entries": len(self._entries),
            "threshold": self.threshold,
            "average_score": round(sum(scores) / len(scores), 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
            "regression_enabled": self.regression_enabled,
            "regression_pass_rate_threshold": self.regression_pass_rate
        }
