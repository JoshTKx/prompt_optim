# Universal Prompt Optimizer with Canonical Test Suite

A production-grade, universal prompt optimization system that improves ANY prompt through iterative Judge-Reviser optimization and validates its universality against a canonical test suite.

## 🎯 Mission

This system provides credible evidence of generalization across diverse prompt types by:

1. **System A: The Optimizer** - Improves prompts through iterative optimization
2. **System B: The Test Suite** - Validates universality with 35+ canonical test cases

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│              UNIVERSAL OPTIMIZER                       │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ TESTER   │→ │  JUDGE   │→ │ REVISER  │              │
│  │          │  │          │  │          │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                        │
└────────────────────────────────────────────────────────┘
                        │
                        ▼ (validates against)
┌────────────────────────────────────────────────────────┐
│        CANONICAL TEST SUITE (35+ test cases)           │
│                                                        │
│  Category 1: Instruction Following (8 tests)          │
│  Category 2: Deep Reasoning (6 tests)                  │
│  Category 3: Gap Tests (7 tests)                       │
│  Category 4: Security/Adversarial (6 tests)            │
│  Category 5: Edge Cases (5 tests)                      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd prompt_optim

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your OpenRouter API key:
# OPENROUTER_API_KEY=your_openrouter_key_here
```

### Basic Usage

```python
from core.orchestrator import Orchestrator
from models.specification import OutputSpecification, Rule
from models.test_case import CanonicalTestCase

# Create your test case and specification
test_case = CanonicalTestCase(...)
specification = OutputSpecification(...)

# Run optimization
optimizer = Orchestrator()
result = optimizer.optimize(
    initial_prompt="Your initial prompt here",
    test_case=test_case,
    specification=specification,
    max_iterations=5
)

print(f"Optimized Prompt: {result.best_prompt}")
print(f"Improvement: {result.improvement}%")
```

### Run Universality Validation

```python
from canonical_suite.suite_loader import CanonicalTestSuite
from validation.universality_validator import UniversalityValidator

# Load test suite
suite = CanonicalTestSuite.load("canonical_suite/tests/")

# Validate universality (run all tests)
validator = UniversalityValidator()
report = validator.validate_optimizer(
    test_suite=suite,
    sample_size=None  # None = run all tests
)

print(f"Pass Rate: {report.pass_rate:.1%}")
print(f"Tests Passed: {report.tests_passed}/{report.total_tests}")
```

## 📁 Project Structure

```
prompt-optimizer/
├── config/              # Configuration (API keys, settings)
│   ├── llm_config.py    # LLM provider configuration (OpenRouter)
│   └── optimization_config.py  # Optimization parameters
├── core/                # Core optimizer components
│   ├── orchestrator.py  # Main optimization loop
│   ├── parallel_orchestrator.py  # Parallel optimization support
│   ├── tester.py        # Runs prompts on test cases
│   ├── judge.py         # Evaluates outputs (DeepSeek V3)
│   ├── reviser.py       # Improves prompts (Claude Sonnet)
│   └── history.py       # Tracks optimization trajectory
├── models/              # Data models (Pydantic)
│   ├── test_case.py     # Test case models
│   ├── specification.py # Output specification models
│   ├── feedback.py      # Judge feedback models
│   └── result.py        # Optimization result models
├── canonical_suite/     # Test suite infrastructure
│   ├── suite_loader.py  # Loads test cases from JSON
│   ├── suite_runner.py  # Runs optimizer on test suite
│   ├── evaluator.py     # Evaluates test results
│   ├── generate_hard_tests.py  # Generate new test cases
│   └── tests/           # 35+ canonical test cases (JSON)
│       ├── category_1_instruction/
│       ├── category_2_reasoning/
│       ├── category_3_gaps/
│       ├── category_4_security/
│       └── category_5_edge/
├── validation/          # Universality validation
│   ├── universality_validator.py  # Main validator
│   └── coverage_analyzer.py  # Coverage analysis
├── utils/               # Utilities
│   ├── llm_client.py    # OpenRouter LLM client
│   ├── cost_tracker.py  # Cost tracking
│   ├── validators.py    # Output validators
│   ├── checker.py       # Checker prompt component
│   ├── golden_set.py    # Dynamic golden set manager
│   └── ...
├── prompts/             # System prompts
│   ├── judge_system.txt
│   ├── reviser_system.txt
│   └── checker_system.txt
├── examples/            # Usage examples
│   ├── optimize_simple.py
│   ├── optimize_parallel.py
│   └── run_universality_test.py
└── outputs/             # Generated outputs
    ├── optimization_runs/
    ├── prompt_versions/
    ├── golden_sets/
    └── universality_reports/
```

## 🧪 Test Suite Categories

### Category 1: Instruction Following (8 tests)

Tests syntactic compliance and constraint satisfaction:

- **IF-001**: JSON formatting without markdown, nested structures
- **IF-002**: Negative constraints (multiple forbidden words)
- **IF-003**: Format switching (CSV/JSON/XML conversions)
- **IF-004**: Length constraints (exact word/sentence counts)
- **IF-005**: Case constraints (lowercase, title case, mixed)
- **IF-087, IF-3047, IF-3847**: Complex schema generation

### Category 2: Deep Reasoning (6 tests)

Tests multi-step logic and ambiguity handling:

- **RS-001**: Boolean logic expressions with precedence
- **RS-002**: Causal judgment (correlation vs causation)
- **RS-003**: Ambiguity detection (lexical, syntactic, semantic)
- **RS-004**: Math problems with verification
- **RS-005**: Tokenization traps (characters vs tokens vs bytes)
- **RS-047**: Multi-domain reasoning (pharmaceutical mergers)

### Category 3: Gap Tests (7 tests)

Addresses specific failure modes:

- **GAP-EXT-001**: Named Entity Recognition (fine-grained)
- **GAP-EXT-002**: Empty extraction (no hallucination)
- **GAP-SQL-001**: Schema adherence with complex queries
- **GAP-SQL-002**: Complex JOINs
- **GAP-SUM-001**: Abstractive summarization
- **GAP-MENTAL_HEALTH-001**: Gap analysis (safety-critical)
- **GAP-FINANCIAL-001, GAP-LEGAL-001**: Domain-specific gap analysis

### Category 4: Security/Adversarial (6 tests)

Tests robustness against attacks (negative tests - must NOT weaken security):

- **SEC-001**: Direct injection attacks
- **SEC-002**: Role reversal attempts
- **SEC-003**: Encoding-based attacks (Base64)
- **SEC-004**: PII leakage prevention
- **SEC-005**: Invisible text attacks
- **SEC-047, SEC-087, SEC-089**: Advanced multi-layer injection

### Category 5: Edge Cases (5 tests)

Boundary conditions:

- **EDGE-001**: Empty input handling
- **EDGE-002**: Huge input processing
- **EDGE-003**: Contradictory context
- **EDGE-004**: Premise error detection
- **EDGE-005**: Unicode edge cases (graphemes, RTL, multilingual)

## 🔧 Configuration

### Environment Variables (.env)

Create a `.env` file in the project root:

```bash
# OpenRouter API Key (single key for all models)
OPENROUTER_API_KEY=your_openrouter_key_here

# Model Selection (OpenRouter format: provider/model-name)
# See https://openrouter.ai/models for available models
JUDGE_MODEL=deepseek/deepseek-v3.2
REVISER_MODEL=anthropic/claude-4.5-sonnet
TARGET_MODEL=google/gemini-3-flash-preview
CHECKER_MODEL=deepseek/deepseek-v3.2

# Optional OpenRouter Headers (for rankings on openrouter.ai)
HTTP_REFERER=https://your-site.com
X_TITLE=Your Site Name

# Optimization Settings
MAX_ITERATIONS=5
TARGET_SCORE=85.0
COST_BUDGET=2.0

# Parallelism Settings
MAX_PARALLEL_OPTIMIZATIONS=3  # Max concurrent optimizations
MAX_PARALLEL_TESTS=5  # Max concurrent test executions
ENABLE_PARALLEL_TESTS=true  # Enable parallel test execution
```

## 📊 How It Works

### Optimization Loop

1. **Test**: Run current prompt on test cases (parallel execution supported)
2. **Judge**: Evaluate outputs (DeepSeek V3 provides scores + critique)
3. **Check**: Convergence or target score reached? (Early stopping at 85+)
4. **Revise**: Improve prompt based on feedback (Claude Sonnet)
5. **Repeat**: Until convergence or max iterations

### Judge Component

- Uses DeepSeek V3 for cost-effective evaluation
- Provides dual feedback: quantitative (0-100 score) + qualitative (specific issues)
- Identifies CRITICAL, HIGH, MEDIUM, LOW severity issues
- System prompt in `prompts/judge_system.txt`

### Reviser Component

- Uses Claude 4.5 Sonnet for high-quality prompt rewriting
- **Score-based temperature strategy** (not iteration-based):
  - Score < 70: temperature 0.7 (aggressive exploration needed)
  - Score 70-85: temperature 0.5 (balanced refinement)
  - Score ≥ 85: temperature 0.3 (conservative, preserve what works)
- Addresses specific issues from judge feedback
- System prompt in `prompts/reviser_system.txt`

### Early Stopping

The optimizer automatically stops when:

- **Target score reached**: Score ≥ 85.0 (configurable via `TARGET_SCORE`)
- **Convergence**: No improvement for 2 consecutive iterations
- **Max iterations**: Reached `MAX_ITERATIONS` limit

## 📈 Success Criteria

### Optimizer Success:

- ✅ Improves prompts across diverse tasks
- ✅ Shows clear score progression (30-40 → 50 → 70 → 85+)
- ✅ Costs < $2 per optimization run
- ✅ Converges in 5-10 iterations
- ✅ Early stopping at target score (85+)

### Test Suite Success:

- ✅ 35+ hard test cases implemented
- ✅ Covers 5 archetype classes (A_directive, B_contextual, C_reasoning, D_adversarial)
- ✅ Spans 5 domain categories
- ✅ All initial prompts score ≤40 (maximizes optimization room)
- ✅ Multiple test cases per file (2-4 cases for better coverage)

### Integration Success:

- ✅ Optimizer passes 80%+ of canonical tests
- ✅ Universality report generated
- ✅ Coverage analysis shows no major gaps

## ⚡ Parallel Execution

The system supports parallel execution at two levels:

### 1. Parallel Test Execution

Within a single optimization, multiple test inputs run concurrently:

```python
from core.orchestrator import Orchestrator

# Parallel test execution is enabled by default
optimizer = Orchestrator()
# Tests will run in parallel automatically (up to MAX_PARALLEL_TESTS)
```

**Configuration**: Set `ENABLE_PARALLEL_TESTS=true` and `MAX_PARALLEL_TESTS=5` in `.env`

### 2. Parallel Multiple Optimizations

Optimize multiple prompts simultaneously:

```python
from core.parallel_orchestrator import ParallelOrchestrator

parallel_optimizer = ParallelOrchestrator(max_workers=3)
results = parallel_optimizer.optimize_multiple(tasks)
```

### 3. Parallel Universality Validation

Run universality tests with parallel optimization:

```python
from validation.universality_validator import UniversalityValidator

validator = UniversalityValidator(use_parallel=True, max_parallel_workers=3)
report = validator.validate_optimizer(test_suite=suite, sample_size=None)
```

**Benefits:**

- **Speed**: 3-5x faster when running multiple optimizations
- **Efficiency**: Better utilization of I/O wait time during API calls
- **Scalability**: Process multiple test cases simultaneously

**Progress Tracking**: tqdm progress bars show real-time status for:

- Test case execution (how many prompts tested)
- Optimization iterations
- Parallel optimization tasks

## 🎓 Examples

See `examples/` directory for:

- `optimize_simple.py` - Basic optimization example
- `optimize_parallel.py` - Parallel optimization example
- `run_universality_test.py` - Full universality validation (with parallel support)

## 📝 Test Case Format

Each test case is a JSON file with:

- `id`, `name`, `category`, `archetype`, `difficulty`
- `task`: Initial prompt (minimal, scores ≤40), target model, context
- `test_cases`: Input/output pairs (2-4 cases per file)
- `evaluation`: Method, validators, pass criteria
- `optimization_challenge`: What makes this test challenging
- `metadata`: Source, tags, archetype class, domain

### Generating New Test Cases

```bash
# Generate hard test cases using LLM
python canonical_suite/generate_hard_tests.py

# Options:
# --output: Output directory (default: canonical_suite/tests)
# --model: Model to use (default: REVISER_MODEL from config)
# --count: Number of tests per template (default: 3)
```

## 🔒 Security Note

Category 4 (Security) tests are **negative tests** - the optimizer should **NOT** weaken security. These tests validate that safety constraints are preserved or strengthened. The optimizer must maintain or improve security posture, not degrade it.

## 📊 Cost Management

- All models accessed via OpenRouter API (single API key)
- Judge uses DeepSeek V3.2 (~90% cheaper than GPT-4)
- Cost tracking per iteration and component
- Budget limits prevent runaway costs (`COST_BUDGET`, `MAX_BUDGET_PER_OPTIMIZATION`)
- See [OpenRouter pricing](https://openrouter.ai/models) for current rates

### Cost Breakdown

**Cost per optimization run = (Number of test cases) × (Number of iterations) × (Cost per test per iteration)**

**Cost per test per iteration:**

- Target model testing: $0.01-0.02
- Judge evaluation: $0.02-0.03
- Reviser improvement: $0.03-0.05 (once per iteration, not per test)
- Checker (if triggered by violations): $0.01-0.02 per violation
- Negative constraints: $0.00 (free pattern matching)
- **Total: ~$0.07-0.12 per test per iteration**

**Example calculations:**

- Small suite (20 test cases × 3 iterations × $0.08) = **$4.80**
- Medium suite (50 test cases × 5 iterations × $0.08) = **$20.00**
- Large suite (100 test cases × 5 iterations × $0.08) = **$40.00**

**Budget recommendations:**

- Small test suites (<20 cases): $2-5
- Medium test suites (20-50 cases): $10-15
- Large test suites (100+ cases): $25-50

**Note:** Costs scale linearly with the number of test cases. The checker only runs when negative constraints detect violations, saving ~$0.05-0.10 per clean iteration.

## 🚀 Production Enhancements

The optimizer includes four production-grade enhancements:

### 1. Dynamic Golden Set

- Automatically captures successful examples (score ≥ 85.0)
- Prevents regression by reusing good patterns
- Limits to 5 examples per test case
- **Cost**: Free (no additional API calls)

### 2. Negative Constraints

- Pattern matching library (markdown in JSON, PII leakage, etc.)
- Catches errors before delivery
- 6+ default constraints included
- **Cost**: Free (pattern matching only)

### 3. Checker Prompt

- Fast pre-validation using GPT-4o-mini or DeepSeek
- Automatically fixes common issues
- Improves output quality by 20%+
- **Cost**: ~$0.01-0.02 per violation (only runs when negative constraints detect issues)
- **Cost savings**: Skips expensive checker when negative constraints are clean

### 4. Multi-Turn Testing

- Tests robustness across conversation turns
- Only used for Security/Adversarial categories
- Detects quality degradation
- **Cost**: ~$0.05-0.20 per run (only when enabled)

**Cost Optimization**: Checker only runs when negative constraints detect violations, saving ~$0.05-0.10 per clean iteration. Costs scale with test case count (see Cost Breakdown above).

```python
# Full production setup
optimizer = Orchestrator(
    use_golden_set=True,           # Always on
    use_negative_constraints=True,  # Always on
    use_checker=True,               # Always on
    use_multi_turn=False,          # Only for Security category
    save_interval=1,                # Save after each iteration
    save_progress=True              # Enable periodic saving
)
```

## 📊 Progress Tracking

The optimizer includes:

- **tqdm progress bars** showing real-time optimization progress
  - Test case execution: "Running test cases: X/Y test"
  - Optimization iterations: "Optimizing [iter X/Y]: X%|████| score=XX.X"
  - Parallel tasks: "Parallel Optimization: X/Y task"
- **Periodic saving** after each iteration (configurable)
- **Final result saving** to `outputs/optimization_runs/`
- All saved results include full history for analysis

## 🧪 Testing

### System Test (No API calls)

```bash
python test_system.py
```

Tests imports, configuration, models, and basic functionality without making API calls.

### Optimizer Improvement Test

```bash
python test_optimizer_improvements.py
```

Runs a quick optimization on a sample test case to verify:

- No regressions occur
- Convergence works properly
- Cost stays reasonable
- Improvements are shown

### Test Locations

- **System Test**: `test_system.py` (root level)
- **Optimizer Test**: `test_optimizer_improvements.py` (root level)
- **Canonical Tests**: `canonical_suite/tests/` (35+ test cases)
- **Examples**: `examples/` (integration examples)

### Running Examples

```bash
# Basic optimization
python examples/optimize_simple.py

# Universality validation (all tests)
python examples/run_universality_test.py
```

## 🔧 OpenRouter Integration

This system uses [OpenRouter](https://openrouter.ai) as the unified API for all LLM providers.

**Benefits:**

- Single API key for all models
- Unified interface across providers
- Easy model switching
- Transparent pricing

**Model Format**: `provider/model-name`

- `openai/gpt-4o-mini`
- `anthropic/claude-4.5-sonnet`
- `deepseek/deepseek-v3.2`
- `google/gemini-3-flash-preview`

See [OpenRouter Models](https://openrouter.ai/models) for the full list.

## 🛠️ Development

### Adding Test Cases

1. Create JSON file in appropriate category directory (`canonical_suite/tests/category_X_*/`)
2. Follow the test case schema (see examples)
3. Use minimal initial prompts (single words) to ensure initial score ≤40
4. Include metadata for coverage analysis
5. Add 2-4 test cases per file for better coverage

### Test Case Requirements

- **Initial prompt**: Should be minimal (single word) to score ≤40
- **Difficulty**: Set to "hard" for challenging tests
- **Test cases**: Include 2-4 test cases per file
- **Constraints**: Be specific and comprehensive
- **Pass criteria**: Require all test cases to pass for hard tests

### Code Structure

- **Core components**: `core/` - Main optimization logic
- **Models**: `models/` - Pydantic data models
- **Utilities**: `utils/` - Reusable utilities
- **Configuration**: `config/` - Settings and API keys
- **System prompts**: `prompts/` - LLM system prompts

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Test cases inspired by:

- IFEval (Instruction Following)
- Omni-MATH (Math reasoning)
- Spider (SQL generation)
- CoNLL-2003 (NER)
- XSum (Summarization)
- CausalBench (Causal reasoning)

---

**Built with ❤️ for universal prompt optimization**
