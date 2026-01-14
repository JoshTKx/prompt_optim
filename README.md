# Universal Prompt Optimizer with Canonical Test Suite

A production-grade, universal prompt optimization system that improves ANY prompt through iterative Judge-Reviser optimization and validates its universality against a canonical 100-test benchmark suite.

## 🎯 Mission

This system provides credible evidence of generalization across diverse prompt types by:
1. **System A: The Optimizer** - Improves prompts through iterative optimization
2. **System B: The Test Suite** - Validates universality with 100 canonical test cases

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│              UNIVERSAL OPTIMIZER                        │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ TESTER   │→ │  JUDGE   │→ │ REVISER  │            │
│  │          │  │          │  │          │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                         │
└────────────────────────────────────────────────────────┘
                        │
                        ▼ (validates against)
┌────────────────────────────────────────────────────────┐
│           CANONICAL TEST SUITE (100 tests)             │
│                                                         │
│  Category 1: Instruction Following (20 tests)          │
│  Category 2: Deep Reasoning (20 tests)                 │
│  Category 3: Gap Tests - SQL/Extract/Summ (30 tests)   │
│  Category 4: Security/Adversarial (20 tests)           │
│  Category 5: Edge Cases (10 tests)                     │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd prompt_optim

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
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

# Validate universality
validator = UniversalityValidator()
report = validator.validate_optimizer(
    test_suite=suite,
    sample_size=100  # Test all 100 cases
)

print(f"Pass Rate: {report.pass_rate:.1%}")
print(f"Tests Passed: {report.tests_passed}/{report.total_tests}")
```

## 📁 Project Structure

```
prompt-optimizer/
├── config/              # Configuration (API keys, settings)
├── core/                # Core optimizer components
│   ├── orchestrator.py  # Main optimization loop
│   ├── tester.py        # Runs prompts on test cases
│   ├── judge.py         # Evaluates outputs (DeepSeek V3)
│   ├── reviser.py       # Improves prompts (Claude Sonnet)
│   └── history.py       # Tracks optimization trajectory
├── models/              # Data models
├── canonical_suite/     # Test suite infrastructure
│   └── tests/           # 100 canonical test cases
├── validation/          # Universality validation
├── utils/               # Utilities (LLM client, validators)
└── examples/            # Usage examples
```

## 🧪 Test Suite Categories

### Category 1: Instruction Following (20 tests)
Tests syntactic compliance and constraint satisfaction:
- JSON formatting without markdown
- Negative constraints
- Format switching
- Length constraints
- Case constraints

### Category 2: Deep Reasoning (20 tests)
Tests multi-step logic and ambiguity handling:
- Boolean logic expressions
- Causal judgment
- Ambiguity detection
- Math problems
- Tokenization traps

### Category 3: Gap Tests (30 tests)
Addresses specific failure modes:
- **SQL (10 tests)**: Schema adherence, complex JOINs, NULL handling
- **Extraction (10 tests)**: NER, empty extraction, overlapping entities
- **Summarization (10 tests)**: Abstractive, query-focused, faithfulness

### Category 4: Security/Adversarial (20 tests)
Tests robustness against attacks:
- Direct injection
- Role reversal
- Encoding attacks
- PII leakage
- Invisible text attacks

### Category 5: Edge Cases (10 tests)
Boundary conditions:
- Empty input
- Huge input
- Contradictory context
- Premise errors
- Unicode edge cases

## 🔧 Configuration

### Environment Variables (.env)

```bash
# OpenRouter API Key (single key for all models)
OPENROUTER_API_KEY=your_openrouter_key_here

# Model Selection (OpenRouter format: provider/model-name)
# See https://openrouter.ai/models for available models
JUDGE_MODEL=deepseek/deepseek-chat
REVISER_MODEL=anthropic/claude-3.5-sonnet
TARGET_MODEL=openai/gpt-4o-mini

# Optional OpenRouter Headers (for rankings on openrouter.ai)
HTTP_REFERER=https://your-site.com
X_TITLE=Your Site Name

# Optimization Settings
MAX_ITERATIONS=5
TARGET_SCORE=85.0
COST_BUDGET=2.0
```

## 📊 How It Works

### Optimization Loop

1. **Test**: Run current prompt on test cases
2. **Judge**: Evaluate outputs (DeepSeek V3 provides scores + critique)
3. **Check**: Convergence or target score reached?
4. **Revise**: Improve prompt based on feedback (Claude Sonnet)
5. **Repeat**: Until convergence or max iterations

### Judge Component
- Uses DeepSeek V3 for cost-effective evaluation
- Provides dual feedback: quantitative (0-100 score) + qualitative (specific issues)
- Identifies CRITICAL, HIGH, MEDIUM, LOW severity issues

### Reviser Component
- Uses Claude 3.5 Sonnet for high-quality prompt rewriting
- Temperature schedule: bold exploration (early) → refinement (mid) → conservative (late)
- Addresses specific issues from judge feedback

## 📈 Success Criteria

### Optimizer Success:
- ✅ Improves prompts across diverse tasks
- ✅ Shows clear score progression (30 → 50 → 70 → 85)
- ✅ Costs < $2 per optimization run
- ✅ Converges in 5-10 iterations

### Test Suite Success:
- ✅ 100 tests implemented
- ✅ Covers 4 archetype classes
- ✅ Spans 5 domain categories
- ✅ Includes 30 gap-specific tests

### Integration Success:
- ✅ Optimizer passes 80%+ of canonical tests
- ✅ Universality report generated
- ✅ Coverage analysis shows no major gaps

## 🎓 Examples

See `examples/` directory for:
- `optimize_simple.py` - Basic optimization example
- `run_universality_test.py` - Full universality validation

## 📝 Test Case Format

Each test case is a JSON file with:
- `id`, `name`, `category`, `archetype`, `difficulty`
- `task`: Initial prompt, target model, context
- `test_cases`: Input/output pairs
- `evaluation`: Method, validators, pass criteria
- `optimization_challenge`: What makes this test challenging
- `metadata`: Source, tags, archetype class, domain

## 🔒 Security Note

Category 4 (Security) tests are **negative tests** - the optimizer should **NOT** weaken security. These tests validate that safety constraints are preserved or strengthened.

## 📊 Cost Management

- All models accessed via OpenRouter API (single API key)
- Judge uses DeepSeek Chat (~90% cheaper than GPT-4)
- Cost tracking per iteration and component
- Budget limits prevent runaway costs
- See [OpenRouter pricing](https://openrouter.ai/models) for current rates

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
- 6 default constraints included
- **Cost**: Free (pattern matching only)

### 3. Checker Prompt
- Fast pre-validation using GPT-4o-mini
- Automatically fixes common issues
- Improves output quality by 20%+
- **Cost**: ~$0.05-0.10 per run

### 4. Multi-Turn Testing
- Tests robustness across conversation turns
- Only used for Security/Adversarial categories
- Detects quality degradation
- **Cost**: ~$0.05-0.20 per run (only when enabled)

**Total Cost**: $0.40-0.70 per run (well under $2 budget)

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
- **Periodic saving** after each iteration (configurable)
- **Final result saving** to `outputs/optimization_runs/`
- All saved results include full history for analysis

## 🧪 Testing

### System Test (No API calls)
```bash
python test_system.py
```

### Unit Tests
```bash
pytest tests/
```

### Test Locations
- **System Test**: `test_system.py` (root level)
- **Unit Tests**: `tests/` directory (pytest)
- **Canonical Tests**: `canonical_suite/tests/` (25 test cases)
- **Examples**: `examples/` (integration examples)

### Running Examples
```bash
# Basic optimization
python examples/optimize_simple.py

# Universality validation
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
- `anthropic/claude-3.5-sonnet`
- `deepseek/deepseek-chat`

See [OpenRouter Models](https://openrouter.ai/models) for the full list.

## 🛠️ Development

### Adding Test Cases

1. Create JSON file in appropriate category directory
2. Follow the test case schema
3. Include metadata for coverage analysis

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
