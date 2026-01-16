"""
Unit tests for training module (train.py).

Validates:
- train_model function fits the model correctly
- Error handling for empty and invalid data
- Model is trained and callable after training
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import DiscountPredictor
from src.training.train import train_model


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create minimal synthetic dataset for training tests.
    
    Returns:
        tuple: (X: pd.DataFrame, y: pd.Series) with 100 rows
    """
    np.random.seed(42)
    n = 100
    
    X = pd.DataFrame({
        "distance_km": np.random.uniform(1000, 6000, n),
        "history_trips": np.random.randint(1, 50, n),
        "avg_spend": np.random.uniform(100, 2000, n),
        "route_id": np.random.choice(["R1", "R2", "R3"], n),
        "origin": np.random.choice(["NYC", "LAX", "SFO"], n),
        "destination": np.random.choice(["LON", "TYO", "PAR"], n),
    })
    
    # Simple linear target: discount increases with distance and history
    y = (
        0.002 * X["distance_km"] 
        + 0.3 * X["history_trips"] 
        + 0.005 * X["avg_spend"]
        + np.random.normal(0, 2, n)
    )
    y = pd.Series(y, name="discount_value")
    
    return X, y


@pytest.fixture
def training_data(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> dict:
    """Create training data dictionary with features and labels.
    
    Args:
        synthetic_data: Tuple of (X, y) from synthetic_data fixture
        
    Returns:
        dict: Dictionary with 'features' and 'labels' keys
    """
    X, y = synthetic_data
    return {'features': X, 'labels': y}


def test_train_model_returns_fitted_model(training_data: dict) -> None:
    """Test train_model returns a fitted model that can predict."""
    model = DiscountPredictor()
    
    trained_model = train_model(model, training_data)
    
    # Model should be returned
    assert trained_model is model
    
    # Model should be able to predict after training
    predictions = trained_model.predict(training_data['features'])
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(training_data['features'])


def test_train_model_produces_valid_predictions(training_data: dict) -> None:
    """Test train_model produces reasonable predictions after training."""
    model = DiscountPredictor()
    
    trained_model = train_model(model, training_data)
    predictions = trained_model.predict(training_data['features'])
    
    # Predictions should be numeric and not contain NaN
    assert not predictions.isna().any(), "Predictions should not contain NaN"
    assert predictions.dtype in [np.float64, np.float32], "Predictions should be float"


def test_train_model_preserves_index(training_data: dict) -> None:
    """Test train_model preserves DataFrame index in predictions."""
    # Create data with custom index
    X = training_data['features'].copy()
    X.index = pd.Index([f"sample_{i}" for i in range(len(X))])
    y = training_data['labels'].copy()
    y.index = X.index
    
    data = {'features': X, 'labels': y}
    model = DiscountPredictor()
    
    trained_model = train_model(model, data)
    predictions = trained_model.predict(X)
    
    assert predictions.index.equals(X.index), "Index should be preserved"


def test_train_model_with_empty_features_raises() -> None:
    """Test train_model raises error when features are empty."""
    model = DiscountPredictor()
    data = {
        'features': pd.DataFrame(),
        'labels': pd.Series(dtype=float)
    }
    
    with pytest.raises(ValueError, match="empty"):
        train_model(model, data)


def test_train_model_with_missing_columns_raises() -> None:
    """Test train_model raises error when required columns are missing."""
    model = DiscountPredictor()
    
    # Create data with missing required columns
    incomplete_data = {
        'features': pd.DataFrame({"distance_km": [3000, 4000]}),
        'labels': pd.Series([10.0, 15.0])
    }
    
    with pytest.raises(ValueError, match="column is not a column of the dataframe"):
        train_model(model, incomplete_data)


def test_train_model_deterministic(training_data: dict) -> None:
    """Test train_model produces deterministic results with same data."""
    model1 = DiscountPredictor()
    model2 = DiscountPredictor()
    
    trained1 = train_model(model1, training_data)
    trained2 = train_model(model2, training_data)
    
    preds1 = trained1.predict(training_data['features'])
    preds2 = trained2.predict(training_data['features'])
    
    pd.testing.assert_series_equal(preds1, preds2)


def test_train_model_fits_with_minimal_data() -> None:
    """Test train_model works with minimal valid data (2 rows)."""
    np.random.seed(42)
    
    X = pd.DataFrame({
        "distance_km": [3000.0, 4000.0],
        "history_trips": [5, 10],
        "avg_spend": [500.0, 800.0],
        "route_id": ["R1", "R2"],
        "origin": ["NYC", "LAX"],
        "destination": ["LON", "TYO"],
    })
    y = pd.Series([10.0, 15.0], name="discount_value")
    
    data = {'features': X, 'labels': y}
    model = DiscountPredictor()
    
    trained_model = train_model(model, data)
    predictions = trained_model.predict(X)
    
    assert len(predictions) == 2
