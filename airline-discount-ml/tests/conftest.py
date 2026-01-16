"""
Shared pytest fixtures for airline-discount-ml tests.

Fixtures defined here are automatically available to all tests
in this directory and subdirectories without explicit imports.

Usage in tests:
    def test_my_function(synthetic_data):
        X, y = synthetic_data
        # ... test code
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Model Testing Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data():
    """
    Create minimal synthetic dataset for model testing.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features X and target y with 100 rows.

    Features:
        - distance_km: Flight distance (1000-6000 km)
        - history_trips: Past trips count (1-50)
        - avg_spend: Average spending ($100-2000)
        - route_id: Categorical route identifier
        - origin: Origin airport code
        - destination: Destination airport code

    Target:
        - discount_value: Calculated discount amount
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
    y = pd.Series(
        0.002 * X["distance_km"] + 0.3 * X["history_trips"] + np.random.normal(0, 2, n),
        name="discount_value"
    )
    return X, y


@pytest.fixture
def small_synthetic_data():
    """
    Minimal dataset for quick validation tests.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Small features X and target y (2 rows).
    """
    np.random.seed(42)
    X = pd.DataFrame({
        "distance_km": [3000.0, 5000.0],
        "history_trips": [10, 25],
        "avg_spend": [500.0, 1200.0],
        "route_id": ["R1", "R2"],
        "origin": ["NYC", "LAX"],
        "destination": ["LON", "TYO"],
    })
    y = pd.Series([15.0, 25.0], name="discount_value")
    return X, y


@pytest.fixture
def synthetic_data_with_custom_index():
    """
    Synthetic data with non-default index for testing index preservation.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target with index 1000-1049.
    """
    np.random.seed(42)
    n = 50
    custom_index = pd.Index(range(1000, 1000 + n))

    X = pd.DataFrame({
        "distance_km": np.random.uniform(1000, 6000, n),
        "history_trips": np.random.randint(1, 50, n),
        "avg_spend": np.random.uniform(100, 2000, n),
        "route_id": np.random.choice(["R1", "R2", "R3"], n),
        "origin": np.random.choice(["NYC", "LAX", "SFO"], n),
        "destination": np.random.choice(["LON", "TYO", "PAR"], n),
    }, index=custom_index)

    y = pd.Series(
        0.002 * X["distance_km"] + 0.3 * X["history_trips"],
        name="discount_value",
        index=custom_index
    )
    return X, y


# =============================================================================
# Database Testing Fixtures
# =============================================================================


@pytest.fixture
def in_memory_db():
    """
    Create in-memory SQLite database with airline schema.

    Yields:
        sqlite3.Connection: In-memory database connection.

    Note:
        Connection is automatically closed after test.
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create schema matching project structure
    cursor.execute("""
        CREATE TABLE passengers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            travel_history TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            distance REAL NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE discounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            passenger_id INTEGER NOT NULL,
            route_id INTEGER NOT NULL,
            discount_value REAL NOT NULL,
            FOREIGN KEY (passenger_id) REFERENCES passengers(id),
            FOREIGN KEY (route_id) REFERENCES routes(id)
        )
    """)
    conn.commit()

    yield conn

    conn.close()


@pytest.fixture
def populated_db(in_memory_db):
    """
    In-memory database pre-populated with sample data.

    Yields:
        sqlite3.Connection: Database with sample rows.
    """
    cursor = in_memory_db.cursor()

    # Insert sample passengers
    cursor.executemany(
        "INSERT INTO passengers (name, travel_history) VALUES (?, ?)",
        [
            ("Alice Smith", '{"flights": 10, "last_flight": "2024-01-15"}'),
            ("Bob Jones", '{"flights": 25, "last_flight": "2024-06-20"}'),
            ("Carol White", '{"flights": 5, "last_flight": "2023-12-01"}'),
        ]
    )

    # Insert sample routes
    cursor.executemany(
        "INSERT INTO routes (origin, destination, distance) VALUES (?, ?, ?)",
        [
            ("NYC", "LON", 5571.0),
            ("LAX", "TYO", 8815.0),
            ("SFO", "PAR", 8973.0),
        ]
    )

    # Insert sample discounts
    cursor.executemany(
        "INSERT INTO discounts (passenger_id, route_id, discount_value) VALUES (?, ?, ?)",
        [
            (1, 1, 15.5),
            (1, 2, 22.0),
            (2, 1, 28.0),
            (3, 3, 10.0),
        ]
    )

    in_memory_db.commit()
    yield in_memory_db


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_model_path():
    """
    Provide temporary path for model save/load tests.

    Yields:
        Path: Temporary file path for model serialization.

    Note:
        Directory is automatically cleaned up after test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "model.joblib"


@pytest.fixture
def temp_directory():
    """
    Provide temporary directory for file I/O tests.

    Yields:
        Path: Temporary directory path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_predictor():
    """
    Create mock DiscountPredictor for agent testing.

    Returns:
        Mock: Mock predictor with standard interface.
    """
    mock = Mock()
    mock.predict.return_value = pd.Series([15.0, 20.0, 25.0], name="discount_value")
    mock.is_fitted = True
    return mock


@pytest.fixture
def mock_database():
    """
    Create mock Database for isolated testing.

    Returns:
        Mock: Mock database with standard query interface.
    """
    mock = Mock()
    mock.query.return_value = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "travel_history": ['{"flights": 10}', '{"flights": 25}', '{"flights": 5}']
    })
    return mock
