"""
src/data_pipeline/cleaner.py
DataCleaner - CRISP-DM Phase 3 : nettoyage des donnees.
Gere les valeurs manquantes, outliers et doublons.

Usage:
    cleaner = DataCleaner()
    df_clean, report = cleaner.clean(df, target_col="debit_download_mbps")
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Nettoie les donnees brutes selon les bonnes pratiques CRISP-DM.

    Ordre des operations (important - ne pas changer) :
    1. Suppression des doublons
    2. Traitement des valeurs manquantes
    3. Traitement des outliers
    4. Validation finale
    """

    def __init__(
        self,
        missing_threshold: float = 0.40,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
    ):
        """
        Args:
            missing_threshold: Si taux manquant > seuil, supprimer la colonne
            outlier_method: Methode de detection ("iqr" ou "zscore")
            outlier_threshold: Seuil pour la detection des outliers
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.cleaning_report = {}

    def clean(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: str = "timestamp",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Pipeline de nettoyage complet.

        Args:
            df: DataFrame brut
            target_col: Colonne cible (ne pas supprimer meme si manquante)
            timestamp_col: Colonne timestamp (ne pas modifier)

        Returns:
            (DataFrame nettoye, rapport de nettoyage)
        """
        logger.info(f"Debut nettoyage : {len(df)} lignes, {len(df.columns)} colonnes")
        df = df.copy()
        report = {"initial_shape": df.shape, "steps": {}}

        # Etape 1 : Supprimer les doublons
        df, step_report = self._remove_duplicates(df, timestamp_col)
        report["steps"]["duplicates"] = step_report

        # Etape 2 : Traiter les valeurs manquantes
        df, step_report = self._handle_missing_values(
            df, target_col, timestamp_col
        )
        report["steps"]["missing_values"] = step_report

        # Etape 3 : Traiter les outliers
        df, step_report = self._handle_outliers(df, target_col, timestamp_col)
        report["steps"]["outliers"] = step_report

        # Etape 4 : Validation finale
        df, step_report = self._validate(df, target_col)
        report["steps"]["validation"] = step_report

        report["final_shape"] = df.shape
        report["rows_removed"] = report["initial_shape"][0] - report["final_shape"][0]
        report["data_retention_rate"] = round(
            report["final_shape"][0] / report["initial_shape"][0], 4
        )

        logger.info(
            f"Nettoyage termine : {report['final_shape'][0]} lignes conservees "
            f"({report['data_retention_rate']*100:.1f}%)"
        )

        self.cleaning_report = report
        return df, report

    def _remove_duplicates(
        self, df: pd.DataFrame, timestamp_col: str
    ) -> tuple[pd.DataFrame, dict]:
        """Supprime les lignes dupliquees."""
        initial_count = len(df)

        if timestamp_col in df.columns:
            # Doublons sur timestamp uniquement
            df = df.drop_duplicates(subset=[timestamp_col], keep="last")
        else:
            df = df.drop_duplicates()

        removed = initial_count - len(df)
        report = {
            "duplicates_removed": removed,
            "rows_after": len(df),
        }

        if removed > 0:
            logger.info(f"Doublons supprimes : {removed}")

        return df, report

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: str,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Traite les valeurs manquantes colonne par colonne.

        Strategie :
        - Taux > missing_threshold : supprimer la colonne (sauf target)
        - Colonne temporelle : interpolation lineaire
        - Autres colonnes numeriques : mediane
        - Colonnes categorielles : mode
        """
        report = {"columns_dropped": [], "imputation": {}}
        cols_to_drop = []

        for col in df.columns:
            if col == timestamp_col:
                continue

            missing_rate = df[col].isnull().mean()
            if missing_rate == 0:
                continue

            if missing_rate > self.missing_threshold and col != target_col:
                cols_to_drop.append(col)
                report["columns_dropped"].append(
                    {"column": col, "missing_rate": round(missing_rate, 4)}
                )
                logger.warning(
                    f"Colonne '{col}' supprimee : {missing_rate*100:.1f}% manquant"
                )
                continue

            # Imputation
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isnull().sum() < len(df) * 0.1:
                    # Peu de manquants : interpolation lineaire
                    df[col] = df[col].interpolate(method="linear", limit_direction="both")
                else:
                    # Beaucoup de manquants : mediane
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    report["imputation"][col] = {
                        "method": "median",
                        "value": round(median_val, 4),
                        "n_imputed": int(df[col].isnull().sum()),
                    }
            else:
                # Categorielle : mode
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(mode_val)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df, report

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: str,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Traite les outliers par winsorisation.

        Winsorisation : limiter les valeurs extremes au percentile 1-99
        sans les supprimer (preserve la continuite de la serie temporelle).
        """
        report = {"columns_treated": []}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in [timestamp_col, "is_anomaly"]:
                continue

            if self.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.outlier_threshold * IQR
                upper = Q3 + self.outlier_threshold * IQR
            else:  # zscore
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - self.outlier_threshold * std
                upper = mean + self.outlier_threshold * std

            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()

            if n_outliers > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                report["columns_treated"].append({
                    "column": col,
                    "n_outliers": int(n_outliers),
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                })

        return df, report

    def _validate(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[pd.DataFrame, dict]:
        """
        Validation finale : supprime les lignes avec target manquante.
        """
        report = {}
        initial = len(df)

        if target_col in df.columns:
            df = df.dropna(subset=[target_col])

        report["rows_dropped_missing_target"] = initial - len(df)
        report["final_missing_rate"] = round(
            df.isnull().sum().sum() / (len(df) * len(df.columns)), 6
        )

        return df, report

    def get_quality_score(self) -> float:
        """
        Calcule un score de qualite global (0-1) apres nettoyage.
        Utilise dans le rapport CRISP-DM Phase 2.
        """
        if not self.cleaning_report:
            return 0.0

        retention = self.cleaning_report.get("data_retention_rate", 1.0)
        missing_after = self.cleaning_report.get("steps", {}).get(
            "validation", {}
        ).get("final_missing_rate", 0.0)

        score = retention * (1 - missing_after * 10)
        return round(max(0.0, min(1.0, score)), 4)
