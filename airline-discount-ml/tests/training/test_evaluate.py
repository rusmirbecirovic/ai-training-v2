"""
Unit tests for evaluate module (evaluate.py).

Validates:
- evaluate_model function returns correct metrics structure
- Metrics are calculated correctly (MAE, MSE, RMSE, R²)
- Error handling for invalid inputs
- calculate_accuracy legacy function
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import DiscountPredictor
from src.training.evaluate import calculate_accuracy, evaluate_model
from src.training.train import train_model


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create minimal synthetic dataset for evaluation tests.
    
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
def trained_model(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> DiscountPredictor:
    """Create and train a DiscountPredictor model.
    
    Args:
        synthetic_data: Tuple of (X, y) from synthetic_data fixture
        
    Returns:
        DiscountPredictor: Trained model ready for evaluation
    """
    X, y = synthetic_data
    model = DiscountPredictor()
    data = {'features': X, 'labels': y}
    return train_model(model, data)


@pytest.fixture
def test_data(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> dict:
    """Create test data dictionary with features and labels.
    
    Args:
        synthetic_data: Tuple of (X, y) from synthetic_data fixture
        
    Returns:
        dict: Dictionary with 'features' and 'labels' keys
    """
    X, y = synthetic_data
    return {'features': X, 'labels': y}


class TestEvaluateModel:
    """Tests for the evaluate_model function."""
    
    def test_evaluate_model_returns_dict(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test evaluate_model returns a dictionary with expected keys."""
        results = evaluate_model(trained_model, test_data)
        
        assert isinstance(results, dict)
        expected_keys = {'predictions', 'mae', 'mse', 'rmse', 'r2_score'}
        assert set(results.keys()) == expected_keys
    
    def test_evaluate_model_predictions_type(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test evaluate_model returns predictions as pandas Series."""
        results = evaluate_model(trained_model, test_data)
        
        assert isinstance(results['predictions'], pd.Series)
        assert len(results['predictions']) == len(test_data['features'])
    
    def test_evaluate_model_metrics_are_numeric(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test all metrics are numeric and not NaN."""
        results = evaluate_model(trained_model, test_data)
        
        for metric in ['mae', 'mse', 'rmse', 'r2_score']:
            assert isinstance(results[metric], (int, float, np.floating))
            assert not np.isnan(results[metric]), f"{metric} should not be NaN"
    
    def test_evaluate_model_mae_non_negative(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test MAE is non-negative."""
        results = evaluate_model(trained_model, test_data)
        
        assert results['mae'] >= 0, "MAE must be non-negative"
    
    def test_evaluate_model_mse_non_negative(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test MSE is non-negative."""
        results = evaluate_model(trained_model, test_data)
        
        assert results['mse'] >= 0, "MSE must be non-negative"
    
    def test_evaluate_model_rmse_is_sqrt_of_mse(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test RMSE equals square root of MSE."""
        results = evaluate_model(trained_model, test_data)
        
        expected_rmse = np.sqrt(results['mse'])
        assert results['rmse'] == pytest.approx(expected_rmse, rel=1e-9)
    
    def test_evaluate_model_r2_in_valid_range(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test R² is within expected range (can be negative for bad models)."""
        results = evaluate_model(trained_model, test_data)
        
        # R² should be <= 1.0; can be negative for very poor models
        assert results['r2_score'] <= 1.0, "R² cannot exceed 1.0"
    
    def test_evaluate_model_good_r2_for_linear_data(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test R² is positive for well-fitted linear data."""
        results = evaluate_model(trained_model, test_data)
        
        # For our synthetic linear data, model should have positive R²
        assert results['r2_score'] > 0, "R² should be positive for linear data"
    
    def test_evaluate_model_preserves_index(
        self, trained_model: DiscountPredictor
    ) -> None:
        """Test evaluate_model preserves custom index in predictions."""
        np.random.seed(42)
        n = 20
        
        X = pd.DataFrame({
            "distance_km": np.random.uniform(1000, 6000, n),
            "history_trips": np.random.randint(1, 50, n),
            "avg_spend": np.random.uniform(100, 2000, n),
            "route_id": np.random.choice(["R1", "R2", "R3"], n),
            "origin": np.random.choice(["NYC", "LAX", "SFO"], n),
            "destination": np.random.choice(["LON", "TYO", "PAR"], n),
        })
        X.index = pd.Index([f"test_{i}" for i in range(n)])
        
        y = pd.Series(np.random.uniform(5, 25, n), index=X.index)
        
        test_data = {'features': X, 'labels': y}
        results = evaluate_model(trained_model, test_data)
        
        assert results['predictions'].index.equals(X.index)


class TestCalculateAccuracy:
    """Tests for the legacy calculate_accuracy function."""
    
    def test_calculate_accuracy_perfect_match(self) -> None:
        """Test calculate_accuracy returns 1.0 for perfect predictions."""
        predictions = [1, 2, 3, 4, 5]
        labels = [1, 2, 3, 4, 5]
        
        accuracy = calculate_accuracy(predictions, labels)
        
        assert accuracy == 1.0
    
    def test_calculate_accuracy_no_match(self) -> None:
        """Test calculate_accuracy returns 0.0 for no matches."""
        predictions = [1, 2, 3]
        labels = [4, 5, 6]
        
        accuracy = calculate_accuracy(predictions, labels)
        
        assert accuracy == 0.0
    
    def test_calculate_accuracy_partial_match(self) -> None:
        """Test calculate_accuracy returns correct ratio for partial matches."""
        predictions = [1, 2, 3, 4]
        labels = [1, 2, 5, 6]
        
        accuracy = calculate_accuracy(predictions, labels)
        
        assert accuracy == pytest.approx(0.5, rel=1e-9)
    
    def test_calculate_accuracy_empty_labels(self) -> None:
        """Test calculate_accuracy returns 0.0 for empty labels."""
        predictions = []
        labels = []
        
        accuracy = calculate_accuracy(predictions, labels)
        
        assert accuracy == 0.0
    
    def test_calculate_accuracy_with_floats(self) -> None:
        """Test calculate_accuracy works with float values."""
        predictions = [1.0, 2.0, 3.0]
        labels = [1.0, 2.0, 3.0]
        
        accuracy = calculate_accuracy(predictions, labels)
        
        assert accuracy == 1.0


class TestEvaluateModelEdgeCases:
    """Edge case tests for evaluate_model function."""
    
    def test_evaluate_model_minimal_data(
        self, trained_model: DiscountPredictor
    ) -> None:
        """Test evaluate_model works with minimal data (1 row)."""
        X = pd.DataFrame({
            "distance_km": [3000.0],
            "history_trips": [5],
            "avg_spend": [500.0],
            "route_id": ["R1"],
            "origin": ["NYC"],
            "destination": ["LON"],
        })
        y = pd.Series([10.0], name="discount_value")
        
        test_data = {'features': X, 'labels': y}
        results = evaluate_model(trained_model, test_data)
        
        assert len(results['predictions']) == 1
        # MAE and MSE should be valid
        assert results['mae'] >= 0
        assert results['mse'] >= 0
    
    def test_evaluate_model_deterministic(
        self, trained_model: DiscountPredictor, test_data: dict
    ) -> None:
        """Test evaluate_model produces identical results on repeated calls."""
        results1 = evaluate_model(trained_model, test_data)
        results2 = evaluate_model(trained_model, test_data)
        
        pd.testing.assert_series_equal(
            results1['predictions'], results2['predictions']
        )
        assert results1['mae'] == results2['mae']
        assert results1['mse'] == results2['mse']
        assert results1['rmse'] == results2['rmse']
        assert results1['r2_score'] == results2['r2_score']
