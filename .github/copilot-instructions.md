## GitHub Copilot Instructions — Training Repository

### Repository Purpose
This is a **training repository** teaching GitHub Copilot best practices through hands-on ML and MCP server exercises. Contains:
- **ML pipeline**: airline-discount-ml/ (discount prediction with synthetic data)
- **Training exercises**: exercises/{01-setup, 02-mcp-server-for-github, 03-agents-and-skills, 04-custom-mcp-server, 05-unit-tests, 06-custom-instructions}/
- **MCP servers**: Custom Synth server (src/mcp_synth/) and GitHub MCP integration

### Project Structure & Navigation
```
├── .github/
│   ├── copilot-instructions.md          # This file (repo-wide rules)
│   └── instructions/                     # Path-scoped rules
│       ├── models.instructions.md        # ML models layer
│       └── tests.instructions.md         # Testing standards
├── airline-discount-ml/                  # Main ML project
│   ├── src/                             
│   │   ├── data/         # SQLite access (Database class)
│   │   ├── models/       # Pure ML (no I/O, no DB imports)
│   │   ├── agents/       # Business logic coordination
│   │   ├── training/     # Train/evaluate scripts
│   │   ├── mcp/          # MCP tools for DB/models
│   │   └── mcp_synth/    # Custom Synth MCP server
│   ├── data/             # SQLite DB + schemas + synthetic data
│   ├── tests/            # pytest suites
│   └── notebooks/        # Jupyter exploration
└── exercises/            # Step-by-step training materials
```

### Architecture Essentials

**Critical Layer Separation:**
- `src/models/` = Pure ML code. **NEVER import src.data.database**. Take DataFrames, return predictions.
- `src/data/` = All SQLite access. Use `Database` class or `get_connection()`. No direct sqlite3 elsewhere.
- `src/agents/` = Coordinate DB + models. Simple stubs for training exercises.

**Data Flow:**
```
Database (SQLite) → pandas DataFrame → models/passenger_profiler.py (build_features) 
→ models/discount_predictor.py (fit/predict) → predictions (index preserved)
```

**Database Schema:**
- Location: `airline-discount-ml/data/airline_discount.db` (SQLite)
- Tables: `passengers` (name, travel_history JSON), `routes` (origin, destination, distance), `discounts`
- Initialize: `python -c "from src.data.database import init_database; init_database()"`

**ML Models (src/models/):**
- `discount_predictor.py`: sklearn Pipeline → LinearRegression. API: `fit(X,y)`, `predict(X)`, `save(path)`, `load(path)`.
  - Required features: `['distance_km', 'history_trips', 'avg_spend', 'route_id', 'origin', 'destination']`
  - Preprocessing: SimpleImputer(median) → StandardScaler (numeric); SimpleImputer(most_frequent) → OneHotEncoder (categorical)
  - Determinism: `random_state=42` everywhere; module seeds: `random.seed(42); np.random.seed(42)`
  - Validation: ValueError for bad inputs; RuntimeError if predict before fit
  - predict() must preserve input DataFrame index
- `passenger_profiler.py`: `build_features(df) → DataFrame`. Miles→km (×1.60934), derive avg_spend, no PII (no passenger_id).

### Developer Workflows

**Setup (from airline-discount-ml/):**
```bash
./setup.sh          # macOS/Linux: venv + deps + Jupyter kernel + DB init
setup.bat           # Windows equivalent
make setup          # Alternative
```

**Manual setup:**
```bash
pip install -e ".[dev]"
python -m ipykernel install --user --name=airline-discount-ml
python -c "from src.data.database import init_database; init_database()"
```

**Testing:**
```bash
pytest tests/ -v                    # All tests
pytest tests/models/ -v             # Models only
make test-cov                       # Coverage report
```

**Key test files:** tests/models/test_discount_predictor.py, tests/models/test_passenger_profiler.py, tests/agents/test_discount_agent.py, tests/data/test_database.py

**Code quality:**
```bash
make format    # black src tests
make lint      # flake8 src tests
```

**Notebooks:**
```bash
jupyter lab    # or: make run-notebook
# Kernel name: "Python (airline-discount-ml)"
# See notebooks/exploratory_analysis.ipynb for DB access pattern
```

**Training:**
```bash
make train     # python src/training/train.py
make evaluate  # python src/training/evaluate.py
```

### Conventions & Best Practices

**Import structure:**
```python
# ✓ Correct (editable install required)
from src.data.database import Database
from src.models.discount_predictor import DiscountPredictor

# ✗ Wrong in models/
from src.data.database import ...  # NEVER import DB in models layer
```

**Change scope:**
- Edits ≤2 files and ≤50 lines → proceed
- Larger changes → propose plan first
- API changes → ask before modifying public interfaces

**Validation patterns:**
- Models: ValueError for bad inputs (missing columns, empty data)
- Models: RuntimeError if predict() called before fit()
- All stochastic operations: random_state=42

**Feature engineering:**
- Required columns strictly enforced by tests
- No PII in features (passenger_id excluded)
- Index preservation required for all model outputs

### MCP Integration

**Built-in tools (src/mcp/tools.py):**
- `query_db`, `train_model`, `predict`, `evaluate` — safe to call for read operations
- Ask before using write operations

**Custom MCP server (src/mcp_synth/server.py):**
- Wraps Synth CLI for synthetic data generation
- Connect via .vscode/mcp.json (see exercises/04-custom-mcp-server/)
- Tools: synth_generate, synth_inspect_model, preview_table_head, export_archive

**GitHub MCP server:**
- Official server for repo management (see exercises/02-mcp-server-for-github/)
- URL: https://api.githubcopilot.com/mcp/

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'src'` in notebooks | Add `sys.path.insert(0, '/path/to/repo')` (see notebooks/exploratory_analysis.ipynb) |
| SQLite table missing | `make db-init` or call `init_database()` |
| Jupyter kernel not found | `python -m ipykernel install --user --name=airline-discount-ml` |
| Import errors after changes | Ensure editable install: `pip install -e ".[dev]"` |

### Exercise-Specific Guidance

**Exercise 01 (Copilot Instructions):**
- Copy files from exercises/01-setup/ to .github/
- Verify path-scoped instructions with applyTo frontmatter
- Test "don't guess" behavior with non-existent columns

**Exercise 02 (GitHub MCP Server):**
- Add MCP server via Command Palette → "MCP: Add Server"
- Use `#github-mcp` prefix in Copilot Chat
- Read-only operations safe; ask before write operations

**Exercise 04 (Custom MCP):**
- Build FastAPI server wrapping Synth CLI
- Connect via .vscode/mcp.json
- Test tools: generate data, inspect models, preview outputs

**Exercise 05 (Unit Tests):**
- Follow pytest patterns in tests/models/
- Use fixtures from conftest.py
- Aim for 80%+ coverage

### Reference Files

| Layer | Key Files |
|-------|-----------|
| Data | src/data/database.py, src/data/preprocessor.py |
| Models | src/models/discount_predictor.py, src/models/passenger_profiler.py |
| Agents | src/agents/discount_agent.py, src/agents/route_analyzer.py |
| Training | src/training/train.py, src/training/evaluate.py |
| MCP | src/mcp/tools.py, src/mcp_synth/server.py |
| Tests | tests/models/, tests/data/, tests/agents/ |

### Core Principle
**Fast, reproducible iterations with clean boundaries.** If context is missing or requirements unclear, ask before guessing. Use attached context (files, selections) as source of truth.

### Additional Notes
VERY IMPORTANT:
- Always first outline the plan before coding.
- DO NOT make significant architectural changes without approval.
- ALWAYS make sure that the python environment is set and is activated before running any scripts.
- NEVER add new columns to the database or modify the scheme without explicit instructions.