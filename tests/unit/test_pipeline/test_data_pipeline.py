"""
tests/unit/test_pipeline/test_data_pipeline.py
Tests unitaires pour le pipeline de donnees.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from src.data_pipeline.cleaner import DataCleaner
from src.data_pipeline.domains.adapters import get_adapter
from src.data_pipeline.features import FeatureEngineer
from src.data_pipeline.pipeline import DataPipeline
from src.data_pipeline.sequences import DataNormalizer, SequenceBuilder
from src.data_pipeline.synthetic import SyntheticDataGenerator, timezone


class TestSyntheticDataGenerator:

    def test_generate_telecom(self):
        gen = SyntheticDataGenerator(domain="telecom")
        df = gen.generate(n_samples=1000)
        assert len(df) == 1000
        assert "debit_download_mbps" in df.columns
        assert "timestamp" in df.columns
        assert "is_anomaly" in df.columns

    def test_generate_finance(self):
        gen = SyntheticDataGenerator(domain="finance")
        df = gen.generate(n_samples=500)
        assert len(df) == 500
        assert "price" in df.columns

    def test_generate_industry(self):
        gen = SyntheticDataGenerator(domain="industry")
        df = gen.generate(n_samples=500)
        assert "quality_score" in df.columns
        assert "temperature_celsius" in df.columns

    def test_anomaly_injection(self):
        gen = SyntheticDataGenerator(domain="synthetic", random_seed=42)
        df = gen.generate(n_samples=1000, anomaly_rate=0.05)
        anomaly_rate = df["is_anomaly"].mean()
        # Taux d anomalies entre 1% et 10% (variable aleatoire)
        assert 0.01 <= anomaly_rate <= 0.10

    def test_unsupported_domain(self):
        with pytest.raises(ValueError):
            SyntheticDataGenerator(domain="unknown")

    def test_reproducibility(self):
        fixed_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        gen1 = SyntheticDataGenerator(domain="telecom", random_seed=42)
        gen2 = SyntheticDataGenerator(domain="telecom", random_seed=42)
        df1 = gen1.generate(n_samples=100, start_date=fixed_date)
        df2 = gen2.generate(n_samples=100, start_date=fixed_date)
        pd.testing.assert_frame_equal(df1, df2)

class TestDomainAdapters:

    def test_telecom_adapter(self):
        adapter = get_adapter("telecom")
        assert adapter.get_target_column() == "debit_download_mbps"
        assert adapter.get_timestamp_column() == "timestamp"
        assert len(adapter.get_feature_columns()) > 0
        assert adapter.get_domain_name() == "telecom"

    def test_finance_adapter(self):
        adapter = get_adapter("finance")
        assert adapter.get_target_column() == "price"
        assert adapter.get_domain_name() == "finance"

    def test_industry_adapter(self):
        adapter = get_adapter("industry")
        assert adapter.get_target_column() == "quality_score"
        assert adapter.get_domain_name() == "industry"

    def test_validation_success(self):
        adapter = get_adapter("telecom")
        gen = SyntheticDataGenerator(domain="telecom")
        df = gen.generate(n_samples=100)
        assert adapter.validate_dataframe(df) is True

    def test_validation_failure(self):
        adapter = get_adapter("telecom")
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError):
            adapter.validate_dataframe(df)

    def test_unsupported_domain(self):
        with pytest.raises(ValueError):
            get_adapter("unknown_domain")


class TestDataCleaner:

    def _make_dirty_df(self):
        gen = SyntheticDataGenerator(domain="synthetic", random_seed=42)
        df = gen.generate(n_samples=500)
        # Injecter des valeurs manquantes
        df.loc[10:20, "value"] = np.nan
        df.loc[50:52, "feature_1"] = np.nan
        # Injecter des doublons
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        return df

    def test_remove_duplicates(self):
        df = self._make_dirty_df()
        cleaner = DataCleaner()
        df_clean, report = cleaner.clean(df, target_col="value")
        assert len(df_clean) < len(df)

    def test_handle_missing_values(self):
        df = self._make_dirty_df()
        cleaner = DataCleaner()
        df_clean, report = cleaner.clean(df, target_col="value")
        assert df_clean["value"].isnull().sum() == 0

    def test_data_retention(self):
        gen = SyntheticDataGenerator(domain="synthetic")
        df = gen.generate(n_samples=1000)
        cleaner = DataCleaner()
        df_clean, report = cleaner.clean(df, target_col="value")
        # Au moins 80% des donnees doivent etre conservees
        assert report["data_retention_rate"] >= 0.80

    def test_quality_score(self):
        gen = SyntheticDataGenerator(domain="synthetic")
        df = gen.generate(n_samples=500)
        cleaner = DataCleaner()
        cleaner.clean(df, target_col="value")
        score = cleaner.get_quality_score()
        assert 0.0 <= score <= 1.0


class TestSequenceBuilder:

    def _make_feature_df(self, n=500):
        gen = SyntheticDataGenerator(domain="synthetic", random_seed=42)
        df = gen.generate(n_samples=n)
        engineer = FeatureEngineer(domain="synthetic")
        df = engineer.transform(df, target_col="value")
        return df, engineer.get_feature_names()

    def test_sequence_shapes(self):
        df, feature_names = self._make_feature_df()
        builder = SequenceBuilder(window_size=24)
        X_train, X_val, X_test, y_train, y_val, y_test = builder.build(
            df, target_col="value", feature_cols=feature_names
        )
        assert X_train.ndim == 3
        assert X_train.shape[1] == 24
        assert X_train.shape[2] == len(feature_names)
        assert len(X_train) == len(y_train)

    def test_temporal_split(self):
        df, feature_names = self._make_feature_df(n=1000)
        builder = SequenceBuilder(window_size=24, train_ratio=0.7, val_ratio=0.15)
        X_train, X_val, X_test, y_train, y_val, y_test = builder.build(
            df, target_col="value", feature_cols=feature_names
        )
        total = len(X_train) + len(X_val) + len(X_test)
        train_pct = len(X_train) / total
        # Train doit etre entre 65% et 75%
        assert 0.65 <= train_pct <= 0.75

    def test_invalid_ratios(self):
        with pytest.raises(ValueError):
            SequenceBuilder(window_size=24, train_ratio=0.8, val_ratio=0.3)


class TestDataNormalizer:

    def test_fit_transform(self):
        X = np.random.randn(100, 24, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        norm = DataNormalizer(method="minmax")
        X_norm, y_norm = norm.fit_transform(X, y)
        assert X_norm.shape == X.shape
        assert y_norm.shape == y.shape

    def test_minmax_range(self):
        X = np.random.randn(200, 24, 3).astype(np.float32)
        y = np.random.randn(200).astype(np.float32)
        norm = DataNormalizer(method="minmax")
        X_norm, y_norm = norm.fit_transform(X, y)
        assert X_norm.min() >= -0.01
        assert X_norm.max() <= 1.01

    def test_inverse_transform(self):
        X = np.random.randn(100, 24, 3).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        norm = DataNormalizer(method="minmax")
        _, y_norm = norm.fit_transform(X, y)
        y_reconstructed = norm.inverse_transform_target(y_norm)
        np.testing.assert_array_almost_equal(y, y_reconstructed, decimal=5)

    def test_transform_without_fit(self):
        norm = DataNormalizer()
        X = np.random.randn(10, 5, 3).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        with pytest.raises(RuntimeError):
            norm.transform(X, y)


class TestDataPipeline:

    def test_run_synthetic_telecom(self):
        pipeline = DataPipeline(domain="telecom", window_size=24)
        result = pipeline.run_synthetic(n_samples=2000)
        assert result.X_train.ndim == 3
        assert result.X_train.shape[1] == 24
        assert result.n_features > 0
        assert len(result.feature_names) == result.n_features

    def test_run_synthetic_shapes_consistent(self):
        pipeline = DataPipeline(domain="synthetic", window_size=12)
        result = pipeline.run_synthetic(n_samples=1000)
        assert result.X_train.shape[2] == result.X_val.shape[2]
        assert result.X_train.shape[2] == result.X_test.shape[2]
        assert result.X_train.shape[1] == 12

    def test_normalizer_available(self):
        pipeline = DataPipeline(domain="synthetic")
        result = pipeline.run_synthetic(n_samples=500)
        assert result.normalizer is not None
        assert result.normalizer._is_fitted

    def test_all_domains(self):
        for domain in ["telecom", "finance", "industry", "synthetic"]:
            pipeline = DataPipeline(domain=domain, window_size=12)
            result = pipeline.run_synthetic(n_samples=500)
            assert result.X_train is not None
