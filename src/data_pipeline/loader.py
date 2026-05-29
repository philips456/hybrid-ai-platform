"""
src/data_pipeline/loader.py
DataLoader - charge les donnees depuis differentes sources.
S adapte automatiquement au domaine configure dans .env.

Usage:
    loader = DataLoader(domain="telecom")
    df = loader.load_csv("data/raw/mes_donnees.csv")
    info = loader.get_info(df)
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_pipeline.base import DatasetInfo
from src.data_pipeline.domains.adapters import get_adapter
from src.data_pipeline.synthetic import SyntheticDataGenerator

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Charge les donnees depuis differentes sources.

    Supporte :
    - Fichiers CSV
    - Fichiers JSON
    - Donnees synthetiques (pour les tests)
    - TODO: API REST (a implementer avec les donnees reelles)
    - TODO: Base de donnees existante (a implementer)
    """

    def __init__(self, domain: str = "synthetic"):
        self.domain = domain
        self.adapter = get_adapter(domain)
        logger.info(f"DataLoader initialise pour le domaine : {domain}")

    def load_csv(
        self,
        path: str,
        timestamp_format: Optional[str] = None,
        encoding: str = "utf-8",
        separator: str = ",",
    ) -> pd.DataFrame:
        """
        Charge un fichier CSV et valide sa structure.

        Args:
            path: Chemin vers le fichier CSV
            timestamp_format: Format de la date ex "%Y-%m-%d %H:%M:%S"
            encoding: Encodage du fichier
            separator: Separateur de colonnes

        Returns:
            DataFrame valide et nettoye

        Raises:
            FileNotFoundError: Si le fichier n existe pas
            ValueError: Si le format est incompatible avec le domaine
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        logger.info(f"Chargement CSV : {path}")

        df = pd.read_csv(path, encoding=encoding, sep=separator)

        # Parser le timestamp
        ts_col = self.adapter.get_timestamp_column()
        if ts_col in df.columns:
            if timestamp_format:
                df[ts_col] = pd.to_datetime(df[ts_col], format=timestamp_format)
            else:
                df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True)

            # Assurer le timezone UTC
            if df[ts_col].dt.tz is None:
                df[ts_col] = df[ts_col].dt.tz_localize("UTC")
            else:
                df[ts_col] = df[ts_col].dt.tz_convert("UTC")

        # Trier par timestamp
        df = df.sort_values(ts_col).reset_index(drop=True)

        # Valider la structure
        self.adapter.validate_dataframe(df)

        logger.info(f"CSV charge : {len(df)} lignes, {len(df.columns)} colonnes")
        return df

    def load_json(self, path: str) -> pd.DataFrame:
        """
        Charge un fichier JSON.

        Args:
            path: Chemin vers le fichier JSON

        Returns:
            DataFrame valide
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        logger.info(f"Chargement JSON : {path}")
        df = pd.read_json(path)

        ts_col = self.adapter.get_timestamp_column()
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
            df = df.sort_values(ts_col).reset_index(drop=True)

        self.adapter.validate_dataframe(df)

        logger.info(f"JSON charge : {len(df)} lignes")
        return df

    def load_synthetic(
        self,
        n_samples: int = 10000,
        anomaly_rate: float = 0.02,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """
        Genere et charge des donnees synthetiques.
        Utilise pendant le developpement avant les donnees reelles.

        Args:
            n_samples: Nombre d observations
            anomaly_rate: Proportion d anomalies (0-1)
            freq: Frequence d echantillonnage

        Returns:
            DataFrame synthetique avec anomalies injectees
        """
        logger.info(
            f"Generation donnees synthetiques : {n_samples} observations, "
            f"domaine={self.domain}, anomaly_rate={anomaly_rate}"
        )
        gen = SyntheticDataGenerator(domain=self.domain)
        df = gen.generate(
            n_samples=n_samples,
            anomaly_rate=anomaly_rate,
            freq=freq,
        )
        logger.info(f"Donnees synthetiques generees : {len(df)} lignes")
        return df

    def get_info(self, df: pd.DataFrame) -> DatasetInfo:
        """
        Retourne les informations sur le dataset charge.
        Correspond au rapport de qualite CRISP-DM Phase 2.

        Args:
            df: DataFrame a analyser

        Returns:
            DatasetInfo avec toutes les caracteristiques du dataset
        """
        ts_col = self.adapter.get_timestamp_column()
        target_col = self.adapter.get_target_column()

        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))

        date_range = (None, None)
        sampling_interval = None

        if ts_col in df.columns and len(df) > 1:
            date_range = (
                df[ts_col].min().isoformat(),
                df[ts_col].max().isoformat()
            )
            intervals = df[ts_col].diff().dropna()
            if len(intervals) > 0:
                median_interval = intervals.median()
                sampling_interval = str(median_interval)

        return DatasetInfo(
            name=f"{self.domain}_dataset",
            domain=self.domain,
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=list(df.columns),
            target_column=target_col,
            timestamp_column=ts_col,
            missing_rate=round(missing_rate, 4),
            date_range=date_range,
            sampling_interval=sampling_interval,
        )

    def save_to_parquet(self, df: pd.DataFrame, path: str) -> None:
        """
        Sauvegarde le DataFrame au format Parquet pour un acces rapide.
        Plus efficace que CSV pour les grands volumes.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")
        logger.info(f"DataFrame sauvegarde en Parquet : {path}")
