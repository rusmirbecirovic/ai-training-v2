---
name: unit-testing
description: Guide for writing pytest unit tests for the airline-discount-ml project. Use when asked to create, run, or debug Python unit tests for models, data access, or agents.
---

# Unit Testing with pytest

This skill helps you create and run unit tests for the airline-discount-ml project using pytest.

## When to use this skill

Use this skill when you need to:
- Create new pytest tests for models, data access layers, or agents
- Debug failing tests
- Add test coverage for edge cases and error paths
- Set up test fixtures and mocks

## Test Structure

Tests mirror the source code structure:
```
tests/
├── models/
│   ├── test_discount_predictor.py
│   └── test_passenger_profiler.py
├── data/
│   └── test_database.py
└── agents/
    └── test_discount_agent.py
```

## Creating tests

1. **Review the [test template](./test_template.py)** for the standard test structure
2. **Target code precisely** - Highlight the function/class to test
3. **Create test file** in the corresponding `tests/` subdirectory
4. **Follow naming conventions**:
   - Test files: `test_<module_name>.py`
   - Test functions: `test_<function_name>_<scenario>`
   - Test classes: `Test<ClassName>`
5. **Use Arrange-Act-Assert pattern** - Organize each test into three clear sections

## Running tests

Navigate to the project directory:
```bash
cd airline-discount-ml
```

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/models/test_discount_predictor.py -v
```

Run specific test:
```bash
pytest tests/models/test_discount_predictor.py::test_predict_preserves_index -v
```

Run with coverage report:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Best practices

### For Models (src/models/)
- **No I/O**: Models must not import from `src.data.database`
- **Deterministic**: Set random seeds (`random.seed(42)`, `np.random.seed(42)`)
- **Index preservation**: Verify pandas Series/DataFrames preserve input index
- **Type hints**: Test with correct types and verify type errors
- **Validation first**: Test that invalid inputs raise clear errors

Example:
```python
def test_predict_without_fit_raises_error():
    """Test that predict() before fit() raises RuntimeError."""
    predictor = DiscountPredictor()
    X = pd.DataFrame({"distance_km": [100]})
    
    with pytest.raises(RuntimeError, match="not fitted"):
        predictor.predict(X)
```

### For Data Access (src/data/)
- **Use fixtures**: Create test database fixtures, don't touch production DB
- **Isolation**: Each test should be independent
- **Transactions**: Roll back after each test
- **Mock external calls**: No real API/network calls

Example:
```python
@pytest.fixture
def test_db():
    """Create a temporary test database."""
    db = Database(":memory:")
    db.init_database()
    yield db
    db.close()

def test_insert_passenger(test_db):
    """Test inserting a passenger record."""
    test_db.insert_passenger({"name": "Test", "miles": 1000})
    result = test_db.get_passenger(1)
    assert result["name"] == "Test"
```

### For Agents (src/agents/)
- **Mock dependencies**: Mock database and model calls
- **Test business logic**: Focus on agent's orchestration logic
- **Verify interactions**: Use `unittest.mock` to verify method calls

Example:
```python
from unittest.mock import Mock

def test_discount_agent_calculates_discount():
    """Test agent calculates discount using model predictions."""
    mock_db = Mock()
    mock_model = Mock()
    mock_model.predict.return_value = pd.Series([15.5])
    
    agent = DiscountAgent(mock_db, mock_model)
    discount = agent.calculate_discount(passenger_id=1)
    
    assert discount == 15.5
    mock_db.get_passenger.assert_called_once()
```

## Common test scenarios

### Success cases (happy path)
- Valid inputs produce expected outputs
- Feature engineering transforms data correctly
- Models make predictions in expected range

### Edge cases
- Empty DataFrames
- Single-row inputs
- Missing optional columns
- Zero/negative values where applicable

### Error cases
- Invalid input types raise TypeError
- Missing required columns raise ValueError
- Unfitted models raise RuntimeError
- Database constraints violations raise appropriate errors

## Troubleshooting

### ImportError: No module named 'src'
Ensure you're running tests from the `airline-discount-ml` directory:
```bash
cd airline-discount-ml
pytest tests/
```

### Tests pass locally but fail in CI
Check for:
- Unseeded random number generators
- Hard-coded file paths (use fixtures)
- Time-dependent assertions
- Database state leaking between tests

### Slow tests
- Remove unnecessary I/O operations
- Use in-memory databases (`:memory:`)
- Mock expensive operations
- Avoid real API calls

### Test discovery issues
Check `pytest.ini` configuration:
```ini
[pytest]
testpaths = tests
pythonpath = .
```

## VS Code integration

Use VS Code Test Explorer:
1. Open Testing view (beaker icon in sidebar)
2. Click "Run All Tests" or run individual tests
3. Debug tests by clicking "Debug Test"
4. View test output in the Test Results panel

Configure in `.vscode/settings.json`:
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.cwd": "${workspaceFolder}/airline-discount-ml"
}
```
