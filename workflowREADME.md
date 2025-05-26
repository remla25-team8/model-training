# Complete CI/CD Pipeline

The implementation includes automated testing, linting, metrics calculation, and README badge updates.

## Quick Setup

```bash
# Clone and checkout PR branch
git clone https://github.com/remla25-team8/model-training.git
cd model-training
git fetch origin
git checkout feature/ci-cd-pipeline-setup
git pull origin feature/ci-cd-pipeline-setup

# Verify environment
python --version  # Requires >= 3.8
```

## Component Validation

### 1. File Structure Check
```bash
# Verify all required files exist
ls -la requirements-dev.txt .pylintrc .flake8 pyproject.toml
ls -la .github/workflows/ml-ci-cd.yml
ls -la scripts/create_test_data.py scripts/calculate_ml_test_score.py scripts/update_readme_badges.py
```

### 2. YAML Syntax Validation
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ml-ci-cd.yml')); print('YAML syntax OK')"
```

### 3. Script Functionality Test
```bash
python scripts/create_test_data.py
python scripts/calculate_ml_test_score.py
python scripts/update_readme_badges.py --help
```

## Assignment 4 Requirements Verification

### SUFFICIENT Level Requirements
```bash
# Check automated testing and linting
grep -q "pytest" .github/workflows/ml-ci-cd.yml && echo "PASS: pytest automation"
grep -q "pylint" .github/workflows/ml-ci-cd.yml && echo "PASS: pylint automation"
grep -q "exit 1" .github/workflows/ml-ci-cd.yml && echo "PASS: build breaking logic"
```

### GOOD Level Requirements
```bash
# Check metrics calculation
grep -q "calculate_ml_test_score" .github/workflows/ml-ci-cd.yml && echo "PASS: test adequacy calculation"
grep -q "coverage" .github/workflows/ml-ci-cd.yml && echo "PASS: coverage measurement"
```

### EXCELLENT Level Requirements
```bash
# Check README automation
grep -q "update_readme_badges" .github/workflows/ml-ci-cd.yml && echo "PASS: README automation"
[ -f scripts/update_readme_badges.py ] && echo "PASS: badge updater exists"
```

## End-to-End Integration Test

```bash
# Test complete pipeline
echo "Testing end-to-end pipeline..."

# 1. Create test data
python scripts/create_test_data.py
ls -la data/raw/

# 2. Calculate ML Test Score
python scripts/calculate_ml_test_score.py --output-json test_results.json --verbose
SCORE=$(python -c "import json; print(json.load(open('test_results.json'))['percentage'])" 2>/dev/null || echo "0")
echo "ML Test Score: $SCORE%"

# 3. Test README badge updates
echo "# Test README" > TEST_README.md
echo "<!-- BADGES:START -->" >> TEST_README.md
echo "<!-- BADGES:END -->" >> TEST_README.md
python scripts/update_readme_badges.py \
    --readme-path TEST_README.md \
    --pylint-score 8.5 --pylint-color green \
    --coverage-percent 85 --coverage-color yellow \
    --ml-test-score $SCORE --ml-test-color green

# Verify badge creation
grep -q "PyLint.*8.5" TEST_README.md && echo "PASS: Badge update works"

# Cleanup
rm -f test_results.json TEST_README.md
rm -rf data/
```

## Team Integration Compatibility

### For DVC Pipeline
The implementation supports DVC project structure and will work with pipeline stages and metrics files.

### For ML Test Score Tests
The ML Test Score calculator automatically discovers tests in the `tests/` directory following the official methodology.

### For Metamorphic Testing
The workflow includes  testing execution that will run metamorphic tests when implemented.

### For Code Quality
All linting configurations are integrated: PyLint, Flake8, Bandit, Black, and isort with ML-friendly settings.

## Workflow Structure

The GitHub workflow includes 4 jobs:
1. **Code Quality & Linting**: PyLint (≥8.0), Flake8, Bandit
2. **ML Testing & Coverage**: pytest, coverage (≥80%), ML Test Score
3. **README Updates**: Automatic badge updates (main branch only)
4. **Report Generation**: Comprehensive pipeline reports

## Post-Approval Testing

After approving, verify on GitHub:
1. GitHub Actions workflow triggers on the PR
2. All 4 jobs execute successfully
3. No syntax or execution errors in logs

## Expected Workflow Behavior

- **On PR**: Runs code quality, testing, and reporting (README updates skipped)
- **On main merge**: Runs all jobs including automatic README badge updates
- **On test failure**: Build breaks and prevents merge
- **On success**: Metrics are calculated and stored for README updates
