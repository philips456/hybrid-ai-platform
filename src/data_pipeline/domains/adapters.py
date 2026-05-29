"""
src/data_pipeline/domains/adapters.py
Adaptateurs specifiques a chaque domaine.
Implementent le contrat defini par BaseDomainAdapter.

Quand tu auras tes donnees reelles, tu modifies uniquement
le fichier de ton domaine — le reste du pipeline ne change pas.
"""
import pandas as pd

from src.data_pipeline.base import BaseDomainAdapter


class TelecomAdapter(BaseDomainAdapter):
    """
    Adaptateur pour les donnees reseau 5G / telecommunications.
    Compatible avec les donnees du projet Pulse5G.
    """

    def get_target_column(self) -> str:
        return "debit_download_mbps"

    def get_timestamp_column(self) -> str:
        return "timestamp"

    def get_feature_columns(self) -> list[str]:
        return [
            "debit_upload_mbps",
            "snr_db",
            "rssi_dbm",
            "latency_ms",
            "packet_error_rate",
        ]

    def get_domain_name(self) -> str:
        return "telecom"

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        required = [self.get_timestamp_column(), self.get_target_column()]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes pour le domaine telecom : {missing}\n"
                f"Colonnes disponibles : {list(df.columns)}"
            )
        return True

    def get_anomaly_context_columns(self) -> list[str]:
        return ["snr_db", "rssi_dbm", "latency_ms"]


class FinanceAdapter(BaseDomainAdapter):
    """
    Adaptateur pour les donnees financieres.
    Compatible avec les donnees bancaires (CEPI-SA) et
    les donnees de marche (cours, volumes, spreads).
    """

    def get_target_column(self) -> str:
        return "price"

    def get_timestamp_column(self) -> str:
        return "timestamp"

    def get_feature_columns(self) -> list[str]:
        return [
            "volume",
            "volatility_realized",
            "spread",
            "momentum",
            "return",
        ]

    def get_domain_name(self) -> str:
        return "finance"

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        required = [self.get_timestamp_column(), self.get_target_column()]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes pour le domaine finance : {missing}\n"
                f"Colonnes disponibles : {list(df.columns)}"
            )
        return True

    def get_anomaly_context_columns(self) -> list[str]:
        return ["volume", "volatility_realized", "spread"]


class IndustryAdapter(BaseDomainAdapter):
    """
    Adaptateur pour les donnees industrielles / manufacturing.
    Compatible avec les donnees de capteurs de production
    (temperature, pression, vibration, qualite).
    """

    def get_target_column(self) -> str:
        return "quality_score"

    def get_timestamp_column(self) -> str:
        return "timestamp"

    def get_feature_columns(self) -> list[str]:
        return [
            "temperature_celsius",
            "pressure_bar",
            "speed_rpm",
            "vibration_mm_s",
            "cycle_time_seconds",
        ]

    def get_domain_name(self) -> str:
        return "industry"

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        required = [self.get_timestamp_column(), self.get_target_column()]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes pour le domaine industry : {missing}\n"
                f"Colonnes disponibles : {list(df.columns)}"
            )
        return True

    def get_anomaly_context_columns(self) -> list[str]:
        return ["temperature_celsius", "vibration_mm_s", "pressure_bar"]


class SyntheticAdapter(BaseDomainAdapter):
    """
    Adaptateur pour les donnees synthetiques de test.
    Utilise pendant le developpement avant d avoir les donnees reelles.
    """

    def get_target_column(self) -> str:
        return "value"

    def get_timestamp_column(self) -> str:
        return "timestamp"

    def get_feature_columns(self) -> list[str]:
        return ["feature_1", "feature_2"]

    def get_domain_name(self) -> str:
        return "synthetic"

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        required = [self.get_timestamp_column(), self.get_target_column()]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes pour le domaine synthetic : {missing}"
            )
        return True


def get_adapter(domain: str) -> BaseDomainAdapter:
    """
    Factory function — retourne l adaptateur correct selon le domaine.

    Usage:
        adapter = get_adapter("telecom")
        target = adapter.get_target_column()  # "debit_download_mbps"

    Quand tu auras tes donnees reelles, change juste DOMAIN dans .env
    et tout le pipeline s adapte automatiquement.
    """
    adapters = {
        "telecom": TelecomAdapter,
        "finance": FinanceAdapter,
        "industry": IndustryAdapter,
        "synthetic": SyntheticAdapter,
    }

    if domain not in adapters:
        raise ValueError(
            f"Domaine '{domain}' non supporte. "
            f"Choisir parmi : {list(adapters.keys())}"
        )

    return adapters[domain]()
