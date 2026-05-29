"""
src/data_pipeline/base.py
Classe abstraite commune a tous les adaptateurs de domaine.
Definit le contrat que chaque domaine (telecom, finance, industry) doit respecter.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DatasetInfo:
    """
    Informations sur un dataset charge.
    Retourne par DataLoader.load() pour donner un apercu des donnees.
    """
    name: str
    domain: str
    n_rows: int
    n_columns: int
    columns: list[str]
    target_column: str
    timestamp_column: str
    missing_rate: float        # Taux global de valeurs manquantes (0-1)
    date_range: tuple          # (date_debut, date_fin)
    sampling_interval: Optional[str]  # ex: "1min", "1h", "1d"


@dataclass
class ProcessedDataset:
    """
    Dataset complet apres traitement CRISP-DM Phase 3.
    Pret a etre utilise par le module CNN+LSTM.
    """
    X_train: "np.ndarray"
    X_val: "np.ndarray"
    X_test: "np.ndarray"
    y_train: "np.ndarray"
    y_val: "np.ndarray"
    y_test: "np.ndarray"
    feature_names: list[str]
    scaler_params: dict        # Parametres du scaler pour reproductibilite
    metadata: dict             # Infos supplementaires (window_size, etc.)


class BaseDomainAdapter(ABC):
    """
    Classe abstraite definissant le contrat de chaque adaptateur de domaine.

    Chaque domaine (telecom, finance, industry) doit implementer
    ces methodes pour s adapter au pipeline generique.

    Usage:
        class TelecomAdapter(BaseDomainAdapter):
            def get_target_column(self) -> str:
                return "debit_download_mbps"
    """

    @abstractmethod
    def get_target_column(self) -> str:
        """Retourne le nom de la colonne cible a predire."""
        pass

    @abstractmethod
    def get_timestamp_column(self) -> str:
        """Retourne le nom de la colonne timestamp."""
        pass

    @abstractmethod
    def get_feature_columns(self) -> list[str]:
        """Retourne la liste des colonnes de features specifiques au domaine."""
        pass

    @abstractmethod
    def get_domain_name(self) -> str:
        """Retourne le nom du domaine (telecom, finance, industry, synthetic)."""
        pass

    @abstractmethod
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Valide que le DataFrame contient les colonnes attendues pour ce domaine.
        Leve une ValueError si le format est incorrect.
        """
        pass

    def get_anomaly_context_columns(self) -> list[str]:
        """
        Colonnes supplementaires a inclure dans le contexte d une anomalie.
        Optionnel - peut etre surcharge par les sous-classes.
        """
        return []
