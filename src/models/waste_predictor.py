"""
WastePredictor: Forecast waste generation at supply chain nodes
using gradient-boosted trees with temporal features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from loguru import logger
from dataclasses import dataclass


@dataclass
class WastePredictionReport:
    mae_kg: float
    rmse_kg: float
    feature_importances: pd.Series
    predictions: pd.Series

    def summary(self) -> str:
        top_features = self.feature_importances.head(5).to_dict()
        return (
            f"=== Waste Prediction Report ===\n"
            f"MAE  : {self.mae_kg:.2f} kg\n"
            f"RMSE : {self.rmse_kg:.2f} kg\n"
            f"Top Features: {top_features}\n"
        )


class WastePredictor:
    """
    Predict waste generation (kg) at a supply chain node for a future period.

    The model is trained on historical inventory and operational data and
    uses XGBoost with temporal and categorical features.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    test_size : float
        Fraction of data to use for validation during training.
    """

    FEATURE_COLS = [
        "quantity_units",
        "shelf_life_days",
        "storage_temp_c",
        "days_in_storage",
        "order_frequency",
        "demand_variability",
        "material_type_enc",
        "node_type_enc",
        "month",
        "quarter",
        "is_peak_season",
    ]

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state

        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._scaler = StandardScaler()
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> WastePredictionReport:
        """
        Train the waste prediction model.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns listed in FEATURE_COLS (pre-encoding) plus
            'waste_kg' as the target.

        Returns
        -------
        WastePredictionReport with validation metrics.
        """
        self._validate_train_input(df)
        df = self._engineer_features(df.copy())

        X = df[self.FEATURE_COLS]
        y = df["waste_kg"]

        X_scaled = self._scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Training WastePredictor on {len(X_train)} samples...")
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._is_fitted = True

        preds = self._model.predict(X_val)
        preds = np.maximum(preds, 0)  # Waste cannot be negative

        mae = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds) ** 0.5

        importances = pd.Series(
            self._model.feature_importances_,
            index=self.FEATURE_COLS,
        ).sort_values(ascending=False)

        report = WastePredictionReport(
            mae_kg=mae,
            rmse_kg=rmse,
            feature_importances=importances,
            predictions=pd.Series(preds),
        )
        logger.success(report.summary())
        return report

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict waste_kg for new observations.

        Parameters
        ----------
        df : pd.DataFrame
            Same schema as training data (excluding 'waste_kg').

        Returns
        -------
        pd.Series of predicted waste_kg values.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not trained. Call .train() first.")

        df = self._engineer_features(df.copy())
        X = df[self.FEATURE_COLS]
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return pd.Series(np.maximum(preds, 0), name="predicted_waste_kg")

    def waste_reduction_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return rows ranked by predicted waste with actionable reduction flags.
        """
        preds = self.predict(df)
        result = df.copy()
        result["predicted_waste_kg"] = preds
        result = result.sort_values("predicted_waste_kg", ascending=False)

        # Flag high-risk records
        threshold = result["predicted_waste_kg"].quantile(0.75)
        result["high_waste_risk"] = result["predicted_waste_kg"] >= threshold

        result["recommendation"] = result.apply(self._recommend, axis=1)
        return result

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categoricals and extract temporal features."""
        for col in ["material_type", "node_type"]:
            enc_col = f"{col}_enc"
            if col in df.columns:
                if col not in self._label_encoders:
                    self._label_encoders[col] = LabelEncoder()
                    df[enc_col] = self._label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    known = set(self._label_encoders[col].classes_)
                    df[col] = df[col].astype(str).where(df[col].astype(str).isin(known), other="unknown")
                    if "unknown" not in known:
                        self._label_encoders[col].classes_ = np.append(
                            self._label_encoders[col].classes_, "unknown"
                        )
                    df[enc_col] = self._label_encoders[col].transform(df[col])
            else:
                df[enc_col] = 0

        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            df["month"] = dates.dt.month
            df["quarter"] = dates.dt.quarter
            df["is_peak_season"] = dates.dt.month.isin([11, 12, 1]).astype(int)
        else:
            df.setdefault("month", 6)
            df.setdefault("quarter", 2)
            df.setdefault("is_peak_season", 0)

        for col in self.FEATURE_COLS:
            df.setdefault(col, 0)

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend(row: pd.Series) -> str:
        if not row.get("high_waste_risk", False):
            return "Monitor — within normal range"
        if row.get("shelf_life_days", 999) < 30:
            return "Reduce reorder quantity; increase turnover frequency"
        if row.get("days_in_storage", 0) > row.get("shelf_life_days", 999) * 0.8:
            return "Urgent: redistribute or discount to prevent expiry"
        if row.get("demand_variability", 0) > 0.5:
            return "Apply demand smoothing or safety stock recalibration"
        return "Review storage conditions and supplier lead times"

    @staticmethod
    def _validate_train_input(df: pd.DataFrame) -> None:
        if "waste_kg" not in df.columns:
            raise ValueError("Training data must contain 'waste_kg' target column.")
        if df["waste_kg"].lt(0).any():
            raise ValueError("waste_kg must be non-negative.")
