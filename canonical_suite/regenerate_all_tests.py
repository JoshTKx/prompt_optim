"""
Regenerate all test cases in the canonical test suite with more challenging and diverse versions.

This script uses an LLM to regenerate existing test cases, making them more challenging
and diverse while maintaining the same structure and format. Tests are processed in
parallel for faster execution.

Usage:
    python canonical_suite/regenerate_all_tests.py
    python canonical_suite/regenerate_all_tests.py --backup
    python canonical_suite/regenerate_all_tests.py --model anthropic/claude-4.5-sonnet
    python canonical_suite/regenerate_all_tests.py --max-workers 10
"""

import json
import os
import sys
import argparse
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LLMClient
from config.llm_config import LLMConfig

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

load_dotenv()

# Category mappings
CATEGORY_PREFIXES = {
    "instruction_following": "IF",
    "reasoning": "RS",
    "gaps": "GAP",
    "security": "SEC",
    "edge": "EDGE"
}

CATEGORY_DIRS = {
    "instruction_following": "category_1_instruction",
    "reasoning": "category_2_reasoning",
    "gaps": "category_3_gaps",
    "security": "category_4_security",
    "edge": "category_5_edge"
}


class TestRegenerator:
    def __init__(self, model: str = None):
        """Initialize test regenerator with LLM client."""
        LLMConfig.validate()
        self.client = LLMClient()
        # Use REVISER_MODEL for generation (typically Claude) or specified model
        self.model = model or LLMConfig.REVISER_MODEL
    
    def regenerate_test(self, existing_test: Dict, test_file_path: Path) -> Dict:
        """Regenerate a test case with more challenging and diverse content."""
        
        category = existing_test.get('category', 'instruction_following')
        archetype = existing_test.get('archetype', 'A1_simple')
        test_id = existing_test.get('id', 'UNKNOWN')
        
        # Build prompt for regeneration
        prompt = f"""You are an expert test case designer. Regenerate this test case to be MORE CHALLENGING and DIVERSE while maintaining the exact same structure.

EXISTING TEST CASE:
{json.dumps(existing_test, indent=2)}

REQUIREMENTS FOR REGENERATION:

1. CHALLENGE LEVEL:
   - Make the test HARDER (initial expected score 30-60 without optimization)
   - Increase complexity of constraints, reasoning, or edge cases
   - Add more subtle failure modes that require specific prompt improvements

2. DIVERSITY (CRITICAL):
   - Use COMPLETELY DIFFERENT domains than the original test
   - AVOID pharmaceutical, clinical trial, medical, or healthcare domains unless the original test was specifically about those
   - Choose from diverse domains like: finance, e-commerce, logistics, education, legal, technology, manufacturing, agriculture, entertainment, sports, real estate, energy, transportation, etc.
   - If original was pharmaceutical/medical, change to finance, legal, or technology
   - If original was finance, change to logistics, education, or legal
   - Ensure test cases within the same test also cover different sub-domains or scenarios
   - Make the domain shift obvious and meaningful

3. STRUCTURE PRESERVATION:
   - Keep the EXACT same JSON structure
   - Maintain the same test ID: {test_id}
   - Keep the same category: {category}
   - Keep the same archetype: {archetype}
   - Preserve all required fields and their types

4. IMPROVEMENTS:
   - Make initial_prompt even more inadequate (too simple/vague)
   - Add more complex constraints or requirements
   - Include edge cases that are harder to handle
   - Ensure test_cases are comprehensive and varied
   - Make optimization_challenge more specific about what must be learned

5. QUALITY:
   - Ensure all test cases are realistic and meaningful
   - Make sure validators and constraints are appropriate
   - Keep evaluation method consistent with test type

Generate the regenerated test case following this EXACT structure (match the format precisely):

{{
  "id": "{test_id}",
  "name": "<new descriptive name - different from original>",
  "category": "{category}",
  "archetype": "{archetype}",
  "difficulty": "hard",
  "task": {{
    "description": "<new description - more challenging>",
    "initial_prompt": "<even simpler/more vague prompt that will fail harder>",
    "target_model": "gpt-4o-mini",
    "context": null
  }},
  "test_cases": [
    {{
      "input": "<new diverse test input>",
      "expected_output": {{
        "type": "<structured|unstructured>",
        "format": "<json|text>",
        "validation": "<validator_name>",
        "constraints": ["<constraint1>", "<constraint2>", "<more challenging constraints>"]
      }}
    }}
    // Add 2-4 diverse test cases total
  ],
  "evaluation": {{
    "method": "<deterministic|llm_judge|hybrid>",
    "validators": ["<validator1>", "<validator2>"],
    "rubric": "five_point_standard",
    "pass_criteria": {{
      "minimum_score": 4,
      "required_validators": ["<validator1>"],
      // Add other criteria as needed
    }}
  }},
  "optimization_challenge": "<more specific challenge about what optimizer must learn>",
  "metadata": {{
    "source": "regenerated",
    "tags": ["<new diverse tags>"],
    "archetype_class": "<A_directive|B_contextual|C_reasoning|D_adversarial>",
    "domain": "<new diverse domain>"
  }}
}}

CRITICAL DOMAIN DIVERSITY REQUIREMENTS:
- If the original test uses pharmaceutical/clinical trial/medical domains, you MUST change to a completely different domain (e.g., finance, legal, e-commerce, logistics, education, technology)
- If the original test uses finance, change to a different domain (e.g., legal, logistics, real estate, energy)
- NEVER repeat the same domain as the original unless it's a security test specifically about that domain
- Choose domains that are realistic but different: e-commerce, supply chain, legal contracts, financial planning, real estate, education, manufacturing, agriculture, entertainment, sports analytics, etc.
- Make the domain change obvious in the name, description, and test cases

OTHER CRITICAL REQUIREMENTS:
- Return ONLY valid JSON, no markdown, no explanation
- The JSON must be complete and properly closed (all braces and brackets matched)
- All strings must be properly escaped and terminated
- All property names must be in double quotes
- No trailing commas
- Make it MORE challenging than the original
- Use completely different examples/scenarios/domains
- Ensure test cases are varied and comprehensive
- Keep the exact same structure and ID

IMPORTANT: Generate the COMPLETE JSON object. Do not truncate or leave it incomplete. Ensure all strings are properly closed with double quotes."""

        # Try up to 2 times with different temperatures
        for attempt in range(2):
            temperature = 0.9 if attempt == 0 else 0.7  # Lower temp on retry for more structured output
            max_tokens = 4000 if attempt == 0 else 5000  # More tokens on retry
            
            try:
                response = self.client.complete(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=False
                )
                
                # Parse response
                content = response.strip()
                
                # Remove markdown if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                # Remove leading/trailing whitespace and newlines
                content = content.strip()
                
                # Try to find and extract JSON object if response is truncated
                # Look for the last complete JSON object
                if not content.startswith('{'):
                    # Find first {
                    start_idx = content.find('{')
                    if start_idx >= 0:
                        content = content[start_idx:]
                
                # Try to fix common JSON issues
                content = self._repair_json(content)
                
                try:
                    regenerated = json.loads(content)
                    
                    # Ensure ID and category match original
                    regenerated['id'] = test_id
                    regenerated['category'] = category
                    regenerated['archetype'] = archetype
                    
                    return regenerated
                    
                except json.JSONDecodeError as e:
                    if attempt == 0:
                        # First attempt failed, try again with lower temperature
                        continue
                    else:
                        # Both attempts failed
                        print(f"  ✗ JSON parse error (attempt {attempt + 1}): {e}")
                        print(f"  Error at line {e.lineno}, column {e.colno}")
                        print(f"  Response preview (first 1000 chars): {content[:1000]}")
                        if len(content) > 1000:
                            print(f"  ... (truncated, total length: {len(content)})")
                        raise
                        
            except Exception as e:
                if attempt == 0:
                    # Retry on first attempt
                    continue
                else:
                    raise
    
    def _repair_json(self, content: str) -> str:
        """Attempt to repair common JSON issues."""
        # Fix trailing commas in objects and arrays (but not inside strings)
        # This regex is safe because it only matches commas followed by closing brackets/braces
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Try to find the complete JSON object by tracking brace/bracket depth
        # This handles truncated responses
        if not content.rstrip().endswith('}'):
            depth = 0
            last_valid_pos = -1
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content):
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            last_valid_pos = i
                            # Found complete JSON object
                            break
                        elif depth < 0:
                            # Mismatched closing, stop here
                            break
            
            # If we found a valid closing position, use that
            if last_valid_pos > 0 and depth == 0:
                content = content[:last_valid_pos + 1]
        
        return content


def find_all_test_files(base_dir: Path) -> List[Path]:
    """Find all test JSON files in the test directory structure."""
    test_files = []
    
    for category_dir in base_dir.iterdir():
        if category_dir.is_dir() and category_dir.name.startswith("category_"):
            for test_file in category_dir.glob("*.json"):
                # Skip manifest file
                if test_file.name != "TEST_MANIFEST.json":
                    test_files.append(test_file)
    
    return sorted(test_files)


def backup_test_file(test_file: Path, backup_dir: Path):
    """Create a backup of the original test file."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    relative_path = test_file.relative_to(test_file.parent.parent.parent)
    backup_path = backup_dir / relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_file, backup_path)


def process_single_test(
    test_file: Path,
    model: str,
    backup_dir: Optional[Path]
) -> Tuple[Path, Optional[Dict], Optional[Exception]]:
    """
    Process a single test file (regenerate it).
    
    Each worker creates its own TestRegenerator instance for thread safety.
    
    Returns:
        Tuple of (test_file, regenerated_test_dict, error)
    """
    try:
        # Create regenerator instance for this thread
        regenerator = TestRegenerator(model=model)
        
        # Load existing test
        with open(test_file, 'r') as f:
            existing_test = json.load(f)
        
        # Create backup if requested
        if backup_dir:
            backup_test_file(test_file, backup_dir)
        
        # Regenerate test
        regenerated_test = regenerator.regenerate_test(existing_test, test_file)
        
        # Save regenerated test
        with open(test_file, 'w') as f:
            json.dump(regenerated_test, f, indent=2)
        
        return (test_file, regenerated_test, None)
        
    except Exception as e:
        return (test_file, None, e)


def main():
    parser = argparse.ArgumentParser(description="Regenerate all test cases with more challenging and diverse versions")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="canonical_suite/tests",
        help="Base directory containing test cases (default: canonical_suite/tests)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for generation (default: REVISER_MODEL from config)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original test files before regeneration"
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="canonical_suite/tests_backup",
        help="Directory to store backups (default: canonical_suite/tests_backup)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be regenerated without actually doing it"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers (default: 5)"
    )
    args = parser.parse_args()
    
    # Setup paths
    base_test_dir = Path(args.test_dir)
    if not base_test_dir.exists():
        print(f"Error: Test directory not found: {base_test_dir}")
        sys.exit(1)
    
    backup_dir = Path(args.backup_dir) if args.backup else None
    
    # Find all test files
    test_files = find_all_test_files(base_test_dir)
    
    if not test_files:
        print(f"No test files found in {base_test_dir}")
        sys.exit(1)
    
    print(f"Found {len(test_files)} test files to regenerate")
    print("=" * 60)
    
    if args.dry_run:
        print("\nDRY RUN MODE - No files will be modified")
        print("\nTest files that would be regenerated:")
        for test_file in test_files:
            print(f"  - {test_file}")
        return
    
    # Get model to use
    model = args.model or LLMConfig.REVISER_MODEL
    
    # Track results
    successful = []
    failed = []
    regenerated_tests = {}  # Map test_file -> regenerated_test_dict
    total_cost = 0.0
    
    print(f"Using {args.max_workers} parallel workers")
    print("=" * 60)
    
    # Process test files in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_test, test_file, model, backup_dir): test_file
            for test_file in test_files
        }
        
        # Progress tracking
        if HAS_TQDM:
            pbar = tqdm(total=len(test_files), desc="Regenerating tests", unit="test")
        else:
            print(f"\nProcessing {len(test_files)} test files in parallel...")
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            test_file = future_to_file[future]
            
            try:
                test_file_path, regenerated_test, error = future.result()
                
                if error:
                    print(f"\n✗ Error regenerating {test_file.name}: {error}")
                    import traceback
                    traceback.print_exc()
                    failed.append(test_file)
                else:
                    test_name = regenerated_test.get('name', 'Unknown') if regenerated_test else 'Unknown'
                    if HAS_TQDM:
                        pbar.set_postfix_str(f"✓ {test_file.name}")
                    else:
                        print(f"✓ {test_file.name}: {test_name}")
                    successful.append(test_file)
                    regenerated_tests[test_file] = regenerated_test
                    
                    # Note: Cost tracking per thread would require shared state
                    # For now, we'll estimate or track separately if needed
                
            except Exception as e:
                print(f"\n✗ Unexpected error processing {test_file.name}: {e}")
                import traceback
                traceback.print_exc()
                failed.append(test_file)
            
            if HAS_TQDM:
                pbar.update(1)
        
        if HAS_TQDM:
            pbar.close()
    
    # Update manifest
    print("\n" + "=" * 60)
    print("Updating TEST_MANIFEST.json...")
    
    manifest_path = base_test_dir / "TEST_MANIFEST.json"
    test_manifest = []
    
    for test_file in successful:
        try:
            # Use cached regenerated test if available, otherwise read from file
            if test_file in regenerated_tests:
                test_data = regenerated_tests[test_file]
            else:
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
            
            test_manifest.append({
                "id": test_data.get('id', 'UNKNOWN'),
                "name": test_data.get('name', 'Unknown'),
                "category": test_data.get('category', 'unknown'),
                "difficulty": test_data.get('difficulty', 'hard'),
                "file": str(test_file.relative_to(base_test_dir))
            })
        except Exception as e:
            print(f"  ✗ Error reading {test_file} for manifest: {e}")
    
    with open(manifest_path, 'w') as f:
        json.dump({
            "total_tests": len(test_manifest),
            "generated_at": str(Path(__file__).stat().st_mtime),
            "difficulty": "hard",
            "regenerated": True,
            "tests": test_manifest
        }, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Successfully regenerated: {len(successful)}/{len(test_files)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
        for test_file in failed:
            print(f"  - {test_file}")
    print(f"✓ Manifest updated: {manifest_path}")
    print(f"\nNote: Cost tracking for parallel execution is approximate.")
    print(f"Each test file required 1 LLM API call for regeneration.")
    if args.backup:
        print(f"✓ Backups saved to: {backup_dir}")


if __name__ == "__main__":
    main()
