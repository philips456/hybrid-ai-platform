"""
src/data_pipeline/synthetic.py
Generateur de donnees synthetiques realistes pour les 3 domaines.
Permet de tester tout le pipeline sans donnees reelles.

Usage:
    from src.data_pipeline.synthetic import SyntheticDataGenerator

    gen = SyntheticDataGenerator(domain="telecom")
    df = gen.generate(n_samples=10000)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional


class SyntheticDataGenerator:
    """
    Genere des donnees synthetiques realistes pour les 3 domaines.

    Les donnees generees reproduisent les caracteristiques reelles :
    - Tendances temporelles
    - Saisonnalite (jour/nuit, semaine/weekend)
    - Bruit gaussien
    - Anomalies injectees artificiellement (pour tester la detection)
    """

    SUPPORTED_DOMAINS = ["telecom", "finance", "industry", "synthetic"]

    def __init__(
        self,
        domain: str = "synthetic",
        random_seed: int = 42,
    ):
        if domain not in self.SUPPORTED_DOMAINS:
            raise ValueError(
                f"Domaine '{domain}' non supporte. "
                f"Choisir parmi : {self.SUPPORTED_DOMAINS}"
            )
        self.domain = domain
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate(
        self,
        n_samples: int = 10000,
        start_date: Optional[datetime] = None,
        freq: str = "1min",
        anomaly_rate: float = 0.02,
    ) -> pd.DataFrame:
        """
        Genere un DataFrame synthetique selon le domaine choisi.

        Args:
            n_samples: Nombre d observations a generer
            start_date: Date de debut (defaut: 30 jours avant maintenant)
            freq: Frequence d echantillonnage ("1min", "1h", "1d")
            anomaly_rate: Proportion d anomalies injectees (0-1)

        Returns:
            DataFrame avec colonnes timestamp + features + is_anomaly
        """
        # Reinitialiser le seed a chaque appel pour la reproducibilite
        np.random.seed(self.random_seed)

        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(
                minutes=n_samples
            )

        timestamps = pd.date_range(
            start=start_date,
            periods=n_samples,
            freq=freq,
            tz="UTC"
        )

        generators = {
            "telecom": self._generate_telecom,
            "finance": self._generate_finance,
            "industry": self._generate_industry,
            "synthetic": self._generate_generic,
        }

        df = generators[self.domain](timestamps, n_samples)

        # Injecter des anomalies
        df = self._inject_anomalies(df, anomaly_rate)

        return df

    # ── GENERATEURS PAR DOMAINE ──────────────────────

    def _generate_telecom(
        self, timestamps: pd.DatetimeIndex, n: int
    ) -> pd.DataFrame:
        """
        Genere des donnees reseau 5G mmWave simulees.
        Reproduit les patterns de Pulse5G :
        - Variation jour/nuit du debit
        - Degradation weekend
        - Instabilite du signal (SNR variable)
        """
        t = np.arange(n)

        # Composante circadienne (cycle 24h)
        hours = np.array([ts.hour for ts in timestamps])
        daily_pattern = 1 + 0.4 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)

        # Debit download (Mbps) - objectif de prediction
        base_dl = 250
        debit_dl = (
            base_dl * daily_pattern
            + 30 * np.sin(2 * np.pi * t / (60 * 24 * 7))
            + np.random.normal(0, 15, n)
        ).clip(50, 600)

        # Debit upload (Mbps)
        debit_ul = (
            debit_dl * 0.2
            + np.random.normal(0, 5, n)
        ).clip(10, 120)

        # SNR (Signal to Noise Ratio, dB)
        snr = (
            20 + 5 * np.sin(2 * np.pi * t / 1440)
            + np.random.normal(0, 2, n)
        ).clip(5, 35)

        # RSSI (Received Signal Strength Indicator, dBm)
        rssi = (
            -70 + 10 * np.sin(2 * np.pi * hours / 24)
            + np.random.normal(0, 3, n)
        ).clip(-95, -45)

        # Latence (ms)
        latency = (
            10 + 5 * (1 / daily_pattern)
            + np.random.exponential(2, n)
        ).clip(1, 100)

        # Taux d erreur paquets (%)
        packet_error = (
            0.5 + 0.3 * np.random.beta(0.5, 5, n)
        ).clip(0, 10)

        return pd.DataFrame({
            "timestamp": timestamps,
            "debit_download_mbps": debit_dl.round(2),
            "debit_upload_mbps": debit_ul.round(2),
            "snr_db": snr.round(2),
            "rssi_dbm": rssi.round(2),
            "latency_ms": latency.round(2),
            "packet_error_rate": packet_error.round(4),
        })

    def _generate_finance(
        self, timestamps: pd.DatetimeIndex, n: int
    ) -> pd.DataFrame:
        """
        Genere des donnees financieres simulees.
        Suit un modele de marche aleatoire geometrique (GBM)
        similaire au modele Black-Scholes.
        """
        mu = 0.0001
        sigma = 0.02
        S0 = 100.0

        returns = np.random.normal(mu, sigma, n)
        prices = S0 * np.exp(np.cumsum(returns))

        volume = (
            1_000_000
            + 500_000 * np.abs(returns) / sigma
            + np.random.exponential(200_000, n)
        ).clip(100_000, 10_000_000)

        vol_realized = pd.Series(returns).rolling(20).std().fillna(sigma).values

        spread = (
            0.01 + 0.05 * vol_realized / sigma
            + np.random.exponential(0.005, n)
        ).clip(0.001, 0.5)

        momentum = pd.Series(returns).rolling(10).mean().fillna(0).values

        return pd.DataFrame({
            "timestamp": timestamps,
            "price": prices.round(4),
            "volume": volume.astype(int),
            "volatility_realized": vol_realized.round(6),
            "spread": spread.round(4),
            "momentum": momentum.round(6),
            "return": returns.round(6),
        })

    def _generate_industry(
        self, timestamps: pd.DatetimeIndex, n: int
    ) -> pd.DataFrame:
        """
        Genere des donnees de capteurs industriels simulees.
        Typique d une ligne de production avec cycles machine.
        """
        t = np.arange(n)

        machine_cycle = np.sin(2 * np.pi * t / 90)

        temperature = (
            180 + 20 * machine_cycle
            + np.random.normal(0, 2, n)
        ).clip(100, 250)

        pressure = (
            150 + 30 * machine_cycle
            + np.random.normal(0, 5, n)
        ).clip(80, 220)

        speed = (
            1500 + 200 * machine_cycle
            + np.random.normal(0, 20, n)
        ).clip(800, 2000)

        vibration = (
            2.0 + 0.5 * np.abs(machine_cycle)
            + np.random.exponential(0.3, n)
        ).clip(0.1, 15)

        quality = (
            95 - 10 * np.abs(temperature - 180) / 20
            - 5 * vibration / 15
            + np.random.normal(0, 2, n)
        ).clip(0, 100)

        cycle_time = (
            45 + 5 * np.abs(machine_cycle)
            + np.random.normal(0, 1, n)
        ).clip(30, 70)

        return pd.DataFrame({
            "timestamp": timestamps,
            "temperature_celsius": temperature.round(2),
            "pressure_bar": pressure.round(2),
            "speed_rpm": speed.round(1),
            "vibration_mm_s": vibration.round(3),
            "quality_score": quality.round(2),
            "cycle_time_seconds": cycle_time.round(2),
        })

    def _generate_generic(
        self, timestamps: pd.DatetimeIndex, n: int
    ) -> pd.DataFrame:
        """
        Genere des donnees generiques pour les tests de base.
        Signal sinusoidal avec bruit gaussien.
        """
        t = np.arange(n)

        value = (
            100
            + 20 * np.sin(2 * np.pi * t / 144)
            + 10 * np.sin(2 * np.pi * t / 1440)
            + np.random.normal(0, 5, n)
        )

        feature_1 = (
            50 + 10 * np.cos(2 * np.pi * t / 144)
            + np.random.normal(0, 2, n)
        )

        feature_2 = (
            np.abs(np.random.normal(0, 1, n))
            * (1 + 0.5 * np.sin(2 * np.pi * t / 288))
        )

        return pd.DataFrame({
            "timestamp": timestamps,
            "value": value.round(4),
            "feature_1": feature_1.round(4),
            "feature_2": feature_2.round(4),
        })

    # ── INJECTION D ANOMALIES ────────────────────────

    def _inject_anomalies(
        self, df: pd.DataFrame, anomaly_rate: float
    ) -> pd.DataFrame:
        """
        Injecte des anomalies artificielles dans le dataset.

        Trois types d anomalies :
        1. Point anomaly : valeur unique extremement elevee ou basse
        2. Contextual anomaly : valeur anormale pour le contexte temporel
        3. Collective anomaly : sequence de valeurs anormales consecutives
        """
        df = df.copy()
        n = len(df)
        n_anomalies = int(n * anomaly_rate)

        df["is_anomaly"] = False

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "is_anomaly" in numeric_cols:
            numeric_cols.remove("is_anomaly")

        target_col = numeric_cols[0]

        anomaly_indices = np.random.choice(n, n_anomalies, replace=False)

        for idx in anomaly_indices:
            anomaly_type = np.random.choice(
                ["point", "contextual", "collective"],
                p=[0.5, 0.3, 0.2]
            )

            if anomaly_type == "point":
                factor = np.random.choice([3.5, 4.0, 0.1, 0.2])
                df.at[df.index[idx], target_col] *= factor

            elif anomaly_type == "contextual":
                std = df[target_col].std()
                df.at[df.index[idx], target_col] += (
                    np.random.choice([-1, 1]) * 3.5 * std
                )

            elif anomaly_type == "collective":
                seq_len = np.random.randint(3, 6)
                end_idx = min(idx + seq_len, n)
                std = df[target_col].std()
                for j in range(idx, end_idx):
                    df.at[df.index[j], target_col] += (
                        np.random.choice([-1, 1]) * 2.5 * std
                    )

            df.at[df.index[idx], "is_anomaly"] = True

        return df

    # ── UTILITAIRES ──────────────────────────────────

    def get_column_info(self) -> dict:
        """
        Retourne les informations sur les colonnes selon le domaine.
        Utilise par FeatureEngineer pour construire les features.
        """
        info = {
            "telecom": {
                "target": "debit_download_mbps",
                "timestamp": "timestamp",
                "features": [
                    "debit_upload_mbps", "snr_db",
                    "rssi_dbm", "latency_ms", "packet_error_rate"
                ],
            },
            "finance": {
                "target": "price",
                "timestamp": "timestamp",
                "features": [
                    "volume", "volatility_realized",
                    "spread", "momentum", "return"
                ],
            },
            "industry": {
                "target": "quality_score",
                "timestamp": "timestamp",
                "features": [
                    "temperature_celsius", "pressure_bar",
                    "speed_rpm", "vibration_mm_s", "cycle_time_seconds"
                ],
            },
            "synthetic": {
                "target": "value",
                "timestamp": "timestamp",
                "features": ["feature_1", "feature_2"],
            },
        }
        return info[self.domain]

    def generate_quality_report(self, df: pd.DataFrame) -> dict:
        """
        Genere un rapport de qualite sur le dataset synthetique.
        Simule le rapport qualite de la CRISP-DM Phase 2.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        report = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "missing_rate": df.isnull().sum().sum() / (len(df) * len(df.columns)),
            "anomaly_rate": df["is_anomaly"].mean() if "is_anomaly" in df.columns else 0,
            "columns": {},
        }

        for col in numeric_df.columns:
            if col == "is_anomaly":
                continue
            report["columns"][col] = {
                "mean": round(numeric_df[col].mean(), 4),
                "std": round(numeric_df[col].std(), 4),
                "min": round(numeric_df[col].min(), 4),
                "max": round(numeric_df[col].max(), 4),
                "missing_count": int(numeric_df[col].isnull().sum()),
            }

        return report