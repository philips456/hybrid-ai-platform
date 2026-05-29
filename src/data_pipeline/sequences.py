"""
src/data_pipeline/sequences.py
DataNormalizer + SequenceBuilder - CRISP-DM Phase 3.
Normalise les donnees et construit les sequences pour CNN+LSTM.

Usage:
    normalizer = DataNormalizer()
    X_norm, y_norm = normalizer.fit_transform(X, y)

    builder = SequenceBuilder(window_size=24)
    X_train, X_val, X_test, y_train, y_val, y_test = builder.build(df, target_col)
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalise les donnees pour le CNN+LSTM.

    ANTI DATA LEAKAGE :
    Les scalers sont ajustes UNIQUEMENT sur le train set,
    puis appliques sur val et test sans recalibration.
    """

    def __init__(self, method: str = "minmax"):
        """
        Args:
            method: "minmax" (0-1) ou "standard" (moyenne=0, std=1)
        """
        self.method = method
        self.feature_scaler = None
        self.target_scaler = None
        self._is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Ajuste les scalers sur le train set uniquement.

        Args:
            X_train: Features d entrainement shape (n, window, features)
            y_train: Cible d entrainement shape (n,)
        """
        ScalerClass = MinMaxScaler if self.method == "minmax" else StandardScaler

        # Reshaper pour sklearn (2D)
        n_samples, window_size, n_features = X_train.shape
        X_reshaped = X_train.reshape(-1, n_features)

        self.feature_scaler = ScalerClass()
        self.feature_scaler.fit(X_reshaped)

        self.target_scaler = ScalerClass()
        self.target_scaler.fit(y_train.reshape(-1, 1))

        self._is_fitted = True
        logger.info(f"Scalers ajustes ({self.method}) sur {n_samples} echantillons")

    def transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applique la normalisation sur n importe quel split.

        Args:
            X: Features shape (n, window, features)
            y: Cible shape (n,)

        Returns:
            (X_norm, y_norm)
        """
        if not self._is_fitted:
            raise RuntimeError("Appeler fit() avant transform()")

        n_samples, window_size, n_features = X.shape

        # Normaliser les features
        X_reshaped = X.reshape(-1, n_features)
        X_norm = self.feature_scaler.transform(X_reshaped)
        X_norm = X_norm.reshape(n_samples, window_size, n_features)

        # Normaliser la cible
        y_norm = self.target_scaler.transform(y.reshape(-1, 1)).flatten()

        return X_norm, y_norm

    def fit_transform(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine fit() et transform() pour le train set."""
        self.fit(X_train, y_train)
        return self.transform(X_train, y_train)

    def inverse_transform_target(self, y_norm: np.ndarray) -> np.ndarray:
        """
        Inverse la normalisation sur les predictions.
        Utilise pour obtenir les valeurs reelles a partir des predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Appeler fit() avant inverse_transform_target()")
        return self.target_scaler.inverse_transform(
            y_norm.reshape(-1, 1)
        ).flatten()

    def save(self, path: str) -> None:
        """Sauvegarde les scalers pour la production."""
        if not self._is_fitted:
            raise RuntimeError("Rien a sauvegarder : fit() pas encore appele")
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.feature_scaler, save_path / "feature_scaler.pkl")
        joblib.dump(self.target_scaler, save_path / "target_scaler.pkl")
        logger.info(f"Scalers sauvegardes : {path}")

    def load(self, path: str) -> None:
        """Charge les scalers sauvegardes."""
        save_path = Path(path)
        self.feature_scaler = joblib.load(save_path / "feature_scaler.pkl")
        self.target_scaler = joblib.load(save_path / "target_scaler.pkl")
        self._is_fitted = True
        logger.info(f"Scalers charges depuis : {path}")

    def get_params(self) -> dict:
        """Retourne les parametres des scalers pour MLflow."""
        if not self._is_fitted:
            return {}
        params = {"normalizer_method": self.method}
        if hasattr(self.feature_scaler, "data_min_"):
            params["feature_min"] = float(self.feature_scaler.data_min_.mean())
            params["feature_max"] = float(self.feature_scaler.data_max_.mean())
        return params


class SequenceBuilder:
    """
    Construit les sequences temporelles pour le CNN+LSTM.

    Fenetre glissante : prend W observations consecutives
    comme entree et predit la suivante.

    Ex avec window_size=3:
        Input:  [t-3, t-2, t-1]  -> Output: t
        Input:  [t-2, t-1, t]    -> Output: t+1
        ...
    """

    def __init__(
        self,
        window_size: int = 24,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        """
        Args:
            window_size: Taille de la fenetre temporelle
            train_ratio: Proportion du train set (0-1)
            val_ratio: Proportion du val set (0-1)
            test_ratio est calcule automatiquement : 1 - train - val
        """
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio

        if self.test_ratio <= 0:
            raise ValueError(
                "train_ratio + val_ratio doit etre < 1.0 "
                f"(actuel: {train_ratio + val_ratio})"
            )

    def build(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list[str]] = None,
        timestamp_col: str = "timestamp",
    ) -> tuple:
        """
        Construit les sequences et effectue le split temporel.

        Args:
            df: DataFrame avec features et cible
            target_col: Colonne cible
            feature_cols: Colonnes de features (defaut: toutes sauf target et timestamp)
            timestamp_col: Colonne timestamp

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
            X shape: (n_samples, window_size, n_features)
            y shape: (n_samples,)
        """
        if feature_cols is None:
            exclude = {timestamp_col, target_col, "is_anomaly"}
            feature_cols = [c for c in df.columns if c not in exclude]

        # Extraire les arrays
        features = df[feature_cols].values.astype(np.float32)
        target = df[target_col].values.astype(np.float32)

        # Construire les sequences
        X, y = self._create_sequences(features, target)

        # Split temporel strict (pas de shuffle - preserve l ordre)
        n = len(X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]

        logger.info(
            f"Sequences construites : "
            f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, "
            f"shape=({self.window_size}, {len(feature_cols)})"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_sequences(
        self, features: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cree les sequences par fenetre glissante.

        Args:
            features: Array (n_timesteps, n_features)
            target: Array (n_timesteps,)

        Returns:
            X: Array (n_sequences, window_size, n_features)
            y: Array (n_sequences,)
        """
        n = len(features) - self.window_size
        n_features = features.shape[1]

        X = np.zeros((n, self.window_size, n_features), dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)

        for i in range(n):
            X[i] = features[i:i + self.window_size]
            y[i] = target[i + self.window_size]

        return X, y

    def get_shapes(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> dict:
        """Retourne les dimensions des splits pour le reporting."""
        return {
            "window_size": self.window_size,
            "n_features": X_train.shape[2],
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "input_shape": (self.window_size, X_train.shape[2]),
        }
