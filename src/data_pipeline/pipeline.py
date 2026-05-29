"""
src/data_pipeline/pipeline.py
DataPipeline - classe principale qui orchestre tout le pipeline.
CRISP-DM Phases 2 et 3 en une seule commande.

Usage:
    # Avec donnees synthetiques (developpement)
    pipeline = DataPipeline(domain="telecom")
    result = pipeline.run_synthetic(n_samples=10000)

    # Avec donnees reelles (production)
    pipeline = DataPipeline(domain="telecom")
    result = pipeline.run_from_csv("data/raw/mes_donnees.csv")
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.data_pipeline.cleaner import DataCleaner
from src.data_pipeline.domains.adapters import get_adapter
from src.data_pipeline.features import FeatureEngineer
from src.data_pipeline.loader import DataLoader
from src.data_pipeline.sequences import DataNormalizer, SequenceBuilder

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Resultat complet du pipeline de donnees.
    Contient tout ce dont le module Deep Learning a besoin.
    """
    # Donnees pretes pour CNN+LSTM
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    # Informations sur les features
    feature_names: list[str]
    n_features: int
    window_size: int

    # Normalizer (pour inverser les predictions)
    normalizer: DataNormalizer

    # Rapports CRISP-DM
    loading_report: dict
    cleaning_report: dict
    shapes: dict

    # Donnees intermediaires (optionnel, pour debug)
    df_raw: Optional[pd.DataFrame] = None
    df_clean: Optional[pd.DataFrame] = None
    df_features: Optional[pd.DataFrame] = None


class DataPipeline:
    """
    Orchestre le pipeline complet de donnees selon CRISP-DM.

    Etapes :
    1. Chargement (DataLoader)
    2. Nettoyage (DataCleaner)
    3. Feature Engineering (FeatureEngineer)
    4. Normalisation (DataNormalizer)
    5. Construction des sequences (SequenceBuilder)

    Quand tu auras tes donnees reelles :
    - Change DOMAIN dans configs/.env
    - Utilise run_from_csv() au lieu de run_synthetic()
    - Tout le reste est automatique
    """

    def __init__(
        self,
        domain: str = "synthetic",
        window_size: int = 24,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        normalizer_method: str = "minmax",
        missing_threshold: float = 0.40,
        outlier_method: str = "iqr",
    ):
        self.domain = domain
        self.adapter = get_adapter(domain)
        self.window_size = window_size

        # Initialiser les composants
        self.loader = DataLoader(domain=domain)
        self.cleaner = DataCleaner(
            missing_threshold=missing_threshold,
            outlier_method=outlier_method,
        )
        self.engineer = FeatureEngineer(domain=domain)
        self.normalizer = DataNormalizer(method=normalizer_method)
        self.builder = SequenceBuilder(
            window_size=window_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        logger.info(f"DataPipeline initialise : domaine={domain}, window={window_size}")

    def run_synthetic(
        self,
        n_samples: int = 10000,
        anomaly_rate: float = 0.02,
        keep_intermediates: bool = False,
    ) -> PipelineResult:
        """
        Lance le pipeline complet avec des donnees synthetiques.
        Utilise pendant le developpement.

        Args:
            n_samples: Nombre d observations a generer
            anomaly_rate: Proportion d anomalies injectees
            keep_intermediates: Conserver les DataFrames intermediaires

        Returns:
            PipelineResult pret pour le CNN+LSTM
        """
        logger.info(f"Pipeline synthetique : {n_samples} samples, domaine={self.domain}")

        # Etape 1 : Chargement
        df_raw = self.loader.load_synthetic(
            n_samples=n_samples,
            anomaly_rate=anomaly_rate,
        )
        info = self.loader.get_info(df_raw)

        return self._process(
            df_raw=df_raw,
            loading_report={"info": info.__dict__},
            keep_intermediates=keep_intermediates,
        )

    def run_from_csv(
        self,
        path: str,
        timestamp_format: Optional[str] = None,
        keep_intermediates: bool = False,
    ) -> PipelineResult:
        """
        Lance le pipeline complet depuis un fichier CSV.
        Utilise avec les donnees reelles.

        Args:
            path: Chemin vers le fichier CSV
            timestamp_format: Format de la date
            keep_intermediates: Conserver les DataFrames intermediaires

        Returns:
            PipelineResult pret pour le CNN+LSTM
        """
        logger.info(f"Pipeline CSV : {path}, domaine={self.domain}")

        # Etape 1 : Chargement
        df_raw = self.loader.load_csv(path, timestamp_format=timestamp_format)
        info = self.loader.get_info(df_raw)

        return self._process(
            df_raw=df_raw,
            loading_report={"info": info.__dict__},
            keep_intermediates=keep_intermediates,
        )

    def _process(
        self,
        df_raw: pd.DataFrame,
        loading_report: dict,
        keep_intermediates: bool = False,
    ) -> PipelineResult:
        """
        Traitement interne commun a toutes les sources de donnees.
        """
        target_col = self.adapter.get_target_column()
        timestamp_col = self.adapter.get_timestamp_column()

        # Etape 2 : Nettoyage
        df_clean, cleaning_report = self.cleaner.clean(
            df_raw, target_col=target_col, timestamp_col=timestamp_col
        )

        # Etape 3 : Feature Engineering
        df_features = self.engineer.transform(
            df_clean, target_col=target_col, timestamp_col=timestamp_col
        )
        feature_names = self.engineer.get_feature_names()

        # Etape 4 : Construction des sequences
        X_train, X_val, X_test, y_train, y_val, y_test = self.builder.build(
            df_features,
            target_col=target_col,
            feature_cols=feature_names,
            timestamp_col=timestamp_col,
        )

        # Etape 5 : Normalisation (fit sur train uniquement)
        X_train_norm, y_train_norm = self.normalizer.fit_transform(X_train, y_train)
        X_val_norm, y_val_norm = self.normalizer.transform(X_val, y_val)
        X_test_norm, y_test_norm = self.normalizer.transform(X_test, y_test)

        shapes = self.builder.get_shapes(X_train_norm, X_val_norm, X_test_norm)

        logger.info(
            f"Pipeline termine : "
            f"train={shapes['train_samples']}, "
            f"val={shapes['val_samples']}, "
            f"test={shapes['test_samples']}, "
            f"features={shapes['n_features']}"
        )

        return PipelineResult(
            X_train=X_train_norm,
            X_val=X_val_norm,
            X_test=X_test_norm,
            y_train=y_train_norm,
            y_val=y_val_norm,
            y_test=y_test_norm,
            feature_names=feature_names,
            n_features=shapes["n_features"],
            window_size=self.window_size,
            normalizer=self.normalizer,
            loading_report=loading_report,
            cleaning_report=cleaning_report,
            shapes=shapes,
            df_raw=df_raw if keep_intermediates else None,
            df_clean=df_clean if keep_intermediates else None,
            df_features=df_features if keep_intermediates else None,
        )
