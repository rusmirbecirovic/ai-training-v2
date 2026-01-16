---
applyTo: "airline-discount-ml/notebooks/**/*.ipynb,notebooks/**/*.ipynb"
---

# Copilot Instructions for notebooks/

## Purpose
Ensure Copilot produces well-structured, reproducible Jupyter notebooks for data exploration and analysis in the airline discount ML project.

## Notebook Setup Requirements

### First Code Cell: Path Configuration
Every notebook MUST start with this setup pattern to enable imports from `src/`:

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))
```

### Second Code Cell: Standard Imports
Group imports logically:

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from src.data.database import get_connection
from src.models import DiscountPredictor, build_features

# Set visualization style
sns.set(style='whitegrid')
```

## Database Access Pattern

**ALWAYS use the project's database module**, not raw sqlite3:

```python
from src.data.database import get_connection

db = get_connection()
connection = db.connection

# Query data
data = pd.read_sql_query('SELECT * FROM passengers', con=connection)
```

**Available tables:** `passengers`, `routes`, `discounts`

## Notebook Structure

Follow this cell organization:

1. **Title cell** (markdown): `# Notebook Title` with brief description
2. **Setup cell** (code): Path configuration
3. **Imports cell** (code): All imports
4. **Data loading cells** (code): Database queries
5. **Analysis cells** (code + markdown): Alternating code and explanations
6. **Conclusion cell** (markdown): Summary of findings

## Coding Standards

### Markdown Cells
- Use markdown cells to explain **what** and **why** before each analysis step
- Include section headers (`##`, `###`) for navigation
- Document assumptions and interpretations

### Code Cells
- One logical operation per cell (load data, transform, visualize)
- Print confirmation messages for long operations: `print("✓ Loaded N records")`
- Use descriptive variable names: `discount_data` not `df`
- Add inline comments for complex queries or transformations

### Visualizations
- Always set figure size: `plt.figure(figsize=(10, 6))`
- Include title, axis labels, and grid
- Use consistent color palette (seaborn's defaults or project colors)
- Call `plt.show()` explicitly

```python
plt.figure(figsize=(10, 6))
sns.histplot(data['column'], bins=20, kde=True, color='steelblue')
plt.title('Descriptive Title')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.grid(True, alpha=0.3)
plt.show()
```

## Hard Constraints

| Constraint | Reason |
|------------|--------|
| No hardcoded file paths | Use `Path` for portability |
| No raw sqlite3 connections | Use `src.data.database` |
| No `!pip install` cells | Dependencies in `requirements.txt` |
| No sensitive data display | Filter PII from outputs |
| Clear outputs before commit | Keep repo clean |
| Reproducible random operations | Use `np.random.seed(42)` |

## Kernel Selection

Use the project's virtual environment kernel:
- **Kernel name:** `Python (airline-discount-ml)`
- Install with: `python -m ipykernel install --user --name=airline-discount-ml`

## SQL Query Patterns

### Simple table query
```python
data = pd.read_sql_query('SELECT * FROM passengers', con=connection)
```

### Join query with aliases
```python
query = """
SELECT 
    p.id as passenger_id,
    p.name as passenger_name,
    r.origin,
    r.destination,
    r.distance,
    d.discount_value
FROM discounts d
JOIN passengers p ON d.passenger_id = p.id
JOIN routes r ON d.route_id = r.id
"""
discount_data = pd.read_sql_query(query, con=connection)
```

## Reference

See [exploratory_analysis.ipynb](../notebooks/exploratory_analysis.ipynb) for a working example of these patterns.
