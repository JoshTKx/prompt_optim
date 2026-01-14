"""Multi-Turn Testing - test prompts across conversation turns."""
from typing import List, Dict, Any, Optional
from utils.llm_client import LLMClient
from utils.cost_tracker import CostTracker
from models.test_case import CanonicalTestCase
from utils.logging_utils import setup_logging, get_correlation_id
from utils.metrics import get_metrics_collector

logger = setup_logging()
metrics = get_metrics_collector()


class MultiTurnTester:
    """Tests prompts across multiple conversation turns."""
    
    def __init__(self, llm_client: LLMClient, cost_tracker: CostTracker):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
    
    def test_multi_turn(
        self,
        prompt: str,
        test_case: CanonicalTestCase,
        turns: int = 3,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test prompt across multiple conversation turns.
        
        Args:
            prompt: System prompt to test
            test_case: Test case with initial input
            turns: Number of conversation turns
            context: Optional context
        
        Returns:
            {
                "turns": List[Dict],
                "constraints_maintained": bool,
                "degradation_detected": bool,
                "final_score": float
            }
        """
        conversation_history = []
        target_model = test_case.target_model
        
        # Initial turn
        initial_input = test_case.test_cases[0].input if test_case.test_cases else "Hello"
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\n{prompt}"
        
        user_message = f"{full_prompt}\n\nUser: {initial_input}"
        
        logger.info(
            "Starting multi-turn test",
            turns=turns,
            test_case_id=test_case.id,
            correlation_id=get_correlation_id()
        )
        metrics.increment("multi_turn.tests_started", tags={"test_case": test_case.id})
        
        for turn_num in range(1, turns + 1):
            try:
                # Generate response
                response = self.llm_client.complete(
                    model=target_model,
                    prompt=user_message,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Store turn
                turn_data = {
                    "turn": turn_num,
                    "input": user_message.split("User: ")[-1] if "User: " in user_message else user_message,
                    "output": response,
                    "success": True
                }
                conversation_history.append(turn_data)
                
                # Build next turn (add response to history)
                conversation_history_text = self._format_history(conversation_history)
                user_message = f"{full_prompt}\n\n{conversation_history_text}\n\nUser: {self._generate_followup(turn_num, turns)}"
                
                logger.debug(f"Turn {turn_num} complete", response_length=len(response))
            
            except Exception as e:
                logger.error(f"Error in turn {turn_num}: {e}")
                conversation_history.append({
                    "turn": turn_num,
                    "input": user_message,
                    "output": None,
                    "success": False,
                    "error": str(e)
                })
                break
        
        # Check constraints maintained
        constraints_maintained = self._check_constraints(conversation_history, test_case)
        degradation_detected = self._detect_degradation(conversation_history)
        
        result = {
            "turns": conversation_history,
            "constraints_maintained": constraints_maintained,
            "degradation_detected": degradation_detected,
            "final_score": self._score_conversation(conversation_history, test_case)
        }
        
        metrics.record(
            "multi_turn.final_score",
            result["final_score"],
            tags={"test_case": test_case.id, "turns": str(turns)}
        )
        
        return result
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for next turn."""
        formatted = []
        for turn in history:
            if turn.get("success"):
                formatted.append(f"User: {turn['input']}")
                formatted.append(f"Assistant: {turn['output']}")
        return "\n".join(formatted)
    
    def _generate_followup(self, turn_num: int, total_turns: int) -> str:
        """Generate follow-up question for next turn."""
        followups = [
            "Can you provide more details?",
            "What about edge cases?",
            "Can you give an example?",
            "Is there anything else I should know?",
            "Can you clarify that?",
        ]
        
        # Vary followups to test different aspects
        idx = (turn_num - 1) % len(followups)
        return followups[idx]
    
    def _check_constraints(
        self,
        history: List[Dict[str, Any]],
        test_case: CanonicalTestCase
    ) -> bool:
        """Check if constraints are maintained across turns."""
        if not history:
            return False
        
        # Check each turn's output against expected constraints
        for turn in history:
            if not turn.get("success"):
                return False
            
            output = turn["output"]
            expected = test_case.test_cases[0].expected_output if test_case.test_cases else None
            
            if expected:
                # Check format constraints
                if expected.format == "json":
                    import json
                    try:
                        # Remove markdown if present
                        clean_output = output.strip()
                        if clean_output.startswith("```"):
                            return False  # Markdown violation
                        json.loads(clean_output)
                    except:
                        return False  # Invalid JSON
        
        return True
    
    def _detect_degradation(self, history: List[Dict[str, Any]]) -> bool:
        """Detect if output quality degrades over turns."""
        if len(history) < 2:
            return False
        
        # Simple heuristic: check if later responses are significantly shorter
        # or contain error indicators
        lengths = [len(turn.get("output", "")) for turn in history if turn.get("success")]
        if len(lengths) < 2:
            return False
        
        # Check for significant length drop
        avg_early = sum(lengths[:len(lengths)//2]) / (len(lengths)//2)
        avg_late = sum(lengths[len(lengths)//2:]) / (len(lengths) - len(lengths)//2)
        
        if avg_late < avg_early * 0.5:  # 50% drop
            return True
        
        # Check for error indicators
        error_indicators = ["error", "sorry", "cannot", "unable", "failed"]
        for turn in history[-2:]:  # Check last 2 turns
            output_lower = turn.get("output", "").lower()
            if any(indicator in output_lower for indicator in error_indicators):
                return True
        
        return False
    
    def _score_conversation(
        self,
        history: List[Dict[str, Any]],
        test_case: CanonicalTestCase
    ) -> float:
        """Score the multi-turn conversation."""
        if not history:
            return 0.0
        
        successful_turns = sum(1 for turn in history if turn.get("success"))
        total_turns = len(history)
        
        base_score = (successful_turns / total_turns) * 100
        
        # Penalties
        if not self._check_constraints(history, test_case):
            base_score *= 0.7  # 30% penalty
        
        if self._detect_degradation(history):
            base_score *= 0.8  # 20% penalty
        
        return round(base_score, 2)
