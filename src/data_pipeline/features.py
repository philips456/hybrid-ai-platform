"""
src/data_pipeline/features.py
FeatureEngineer - CRISP-DM Phase 3 : ingenierie des variables.
Cree les features temporelles et specifiques au domaine.

Usage:
    engineer = FeatureEngineer(domain="telecom")
    df_features = engineer.transform(df, target_col="debit_download_mbps")
"""
import logging

import numpy as np
import pandas as pd

from src.data_pipeline.domains.adapters import get_adapter

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Construit les variables d entree pour le modele CNN+LSTM.

    Categories de features :
    1. Features temporelles generiques (toujours creees)
    2. Lag features (valeurs passees)
    3. Rolling statistics (moyenne, ecart-type mobiles)
    4. Features specifiques au domaine (selon adapter)
    """

    def __init__(
        self,
        domain: str = "synthetic",
        lag_periods: list[int] = None,
        rolling_windows: list[int] = None,
    ):
        """
        Args:
            domain: Domaine applicatif
            lag_periods: Periodes de decalage ex [1, 2, 6, 24]
            rolling_windows: Fenetres glissantes ex [7, 14, 30]
        """
        self.domain = domain
        self.adapter = get_adapter(domain)
        self.lag_periods = lag_periods or [1, 2, 3, 6, 12, 24]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.feature_names = []

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Applique toutes les transformations de feature engineering.

        Args:
            df: DataFrame nettoye
            target_col: Colonne cible
            timestamp_col: Colonne timestamp

        Returns:
            DataFrame enrichi avec toutes les features
        """
        logger.info(f"Feature engineering : {len(df)} lignes, domaine={self.domain}")
        df = df.copy()

        # 1. Features temporelles cycliques
        df = self._add_temporal_features(df, timestamp_col)

        # 2. Lag features sur la cible
        df = self._add_lag_features(df, target_col)

        # 3. Rolling statistics sur la cible
        df = self._add_rolling_features(df, target_col)

        # 4. Differentiation (stationnarisation)
        df = self._add_diff_features(df, target_col)

        # 5. Supprimer les NaN crees par les lags/rolling
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = initial_len - len(df)

        if dropped > 0:
            logger.info(
                f"Lignes supprimees apres feature engineering (NaN) : {dropped}"
            )

        # Mettre a jour la liste des features
        exclude_cols = {timestamp_col, target_col, "is_anomaly"}
        self.feature_names = [
            col for col in df.columns if col not in exclude_cols
        ]

        logger.info(
            f"Feature engineering termine : "
            f"{len(self.feature_names)} features creees"
        )
        return df

    def _add_temporal_features(
        self, df: pd.DataFrame, timestamp_col: str
    ) -> pd.DataFrame:
        """
        Encode les composantes temporelles de facon cyclique.
        sin/cos preservent la continuite du cycle (ex: 23h et 0h sont proches).
        """
        if timestamp_col not in df.columns:
            return df

        ts = df[timestamp_col]

        # Heure du jour (cycle 24h)
        hour = ts.dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Jour de la semaine (cycle 7j)
        dow = ts.dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Mois (cycle 12 mois)
        month = ts.dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        # Indicateur weekend (0 ou 1)
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        return df

    def _add_lag_features(
        self, df: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        """
        Cree des variables de decalage (lag features).
        Permet au modele de voir les valeurs passees comme contexte.
        """
        for lag in self.lag_periods:
            col_name = f"{target_col}_lag_{lag}"
            df[col_name] = df[target_col].shift(lag)

        return df

    def _add_rolling_features(
        self, df: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        """
        Cree des statistiques glissantes (rolling statistics).
        Capture les tendances et la variabilite locale.
        """
        for window in self.rolling_windows:
            # Moyenne mobile
            df[f"{target_col}_rolling_mean_{window}"] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
            # Ecart-type mobile
            df[f"{target_col}_rolling_std_{window}"] = (
                df[target_col].rolling(window=window, min_periods=2).std()
            )

        return df

    def _add_diff_features(
        self, df: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        """
        Cree la premiere difference pour stationnariser la serie.
        Utile pour les series avec tendance.
        """
        df[f"{target_col}_diff_1"] = df[target_col].diff(1)
        df[f"{target_col}_diff_7"] = df[target_col].diff(7)

        return df

    def get_feature_names(self) -> list[str]:
        """Retourne la liste des features creees."""
        return self.feature_names

    def get_feature_count(self) -> int:
        """Retourne le nombre de features."""
        return len(self.feature_names)
