from __future__ import annotations

import random
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Determinism
random.seed(42)
np.random.seed(42)


REQUIRED_NUMERIC: List[str] = ["distance_km", "history_trips", "avg_spend"]
REQUIRED_CATEGORICAL: List[str] = ["route_id", "origin", "destination"]
REQUIRED_FEATURES: List[str] = REQUIRED_NUMERIC + REQUIRED_CATEGORICAL


class DiscountPredictor:
    """
    Discount prediction model using an sklearn Pipeline.

    Public API:
      - fit(X, y) -> DiscountPredictor
      - predict(X) -> pd.Series  (preserves index)
      - save(path) / load(path)
    """

    def __init__(self) -> None:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        self._prep = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, REQUIRED_NUMERIC),
                ("cat", categorical_pipe, REQUIRED_CATEGORICAL),
            ],
            remainder="drop",
        )
        self._model = LinearRegression()
        self._pipeline: Pipeline | None = Pipeline(
            steps=[("prep", self._prep), ("model", self._model)]
        )
        self._fitted: bool = False

    @staticmethod
    def _validate_X(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("X is empty.")
        missing = set(REQUIRED_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


    @staticmethod
    def _validate_y(y: pd.Series) -> None:
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series.")
        if y.empty:
            raise ValueError("y is empty.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DiscountPredictor:
        """
        Fit the discount prediction model.

        Args:
            X: Training features DataFrame with required columns:
               ['distance_km', 'history_trips', 'avg_spend', 'route_id', 'origin', 'destination']
            y: Target values (discount_value) as pandas Series

        Returns:
            Self for method chaining

        Raises:
            ValueError: If X is not a DataFrame, is empty, or missing required columns
            ValueError: If y is not a Series or is empty
            RuntimeError: If pipeline is not initialized
        """
        self._validate_X(X)
        self._validate_y(y)
        if self._pipeline is None:
            raise RuntimeError("Pipeline is not initialized.")
        # Align y to X index
        y = y.loc[X.index]
        self._pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict discount values for input features.

        Args:
            X: Features DataFrame with required columns:
               ['distance_km', 'history_trips', 'avg_spend', 'route_id', 'origin', 'destination']

        Returns:
            Predicted discount values as pandas Series with preserved index

        Raises:
            RuntimeError: If model has not been fitted yet
            ValueError: If X is not a DataFrame, is empty, or missing required columns
        """
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._validate_X(X)
        preds = self._pipeline.predict(X)
        return pd.Series(preds, index=X.index, name="discount_value")

    def save(self, path: str | Path) -> None:
        """
        Save the fitted model to disk.

        Args:
            path: File path where model should be saved (str or Path)

        Raises:
            RuntimeError: If model has not been fitted yet
        """
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "version": 1}, path)

    @classmethod
    def load(cls, path: str | Path) -> DiscountPredictor:
        """
        Load a previously saved model from disk.

        Args:
            path: File path to the saved model (str or Path)

        Returns:
            Loaded DiscountPredictor instance with fitted pipeline

        Raises:
            FileNotFoundError: If the model file does not exist
        """
        blob = joblib.load(Path(path))
        obj = cls()
        obj._pipeline = blob["pipeline"]
        obj._fitted = True
        return obj