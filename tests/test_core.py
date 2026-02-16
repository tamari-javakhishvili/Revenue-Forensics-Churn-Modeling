import sys
import os
import pandas as pd
import pytest
import json
import numpy as np

# Ensure 'src' is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import ChurnPredictor


class TestChurnPredictor:
    """
    Comprehensive test suite for ChurnPredictor ML pipeline.
    
    Test Coverage:
    1. Target leakage prevention (Recency exclusion)
    2. Feature engineering correctness
    3. Data split integrity
    4. Edge case handling (division by zero, encoding)
    """
    
    @pytest.fixture
    def predictor(self):
        """Fixture to initialize ChurnPredictor instance."""
        return ChurnPredictor()
    
    @pytest.fixture
    def mock_data(self):
        """
        Fixture to create realistic mock datasets.
        
        Returns:
            tuple: (mock_rfm, mock_raw) DataFrames with sufficient rows
                   for stratified train/test split.
        """
        np.random.seed(42)
        n_customers = 100  # Enough for 80/20 split
        
        mock_rfm = pd.DataFrame({
            'CustomerID': range(1, n_customers + 1),
            'Recency': np.random.randint(5, 200, n_customers),
            'Frequency': np.random.randint(1, 20, n_customers),
            'Monetary': np.random.uniform(20, 1500, n_customers)
        })
        
        # Create realistic country distribution
        countries = ['UK', 'Germany', 'France', 'Spain']
        mock_raw = pd.DataFrame({
            'CustomerID': range(1, n_customers + 1),
            'Country': np.random.choice(countries, n_customers)
        })
        
        return mock_rfm, mock_raw
    
    def test_no_target_leakage(self, predictor, mock_data):
        """
        CRITICAL TEST: Ensures Recency is NOT used as a feature.
        
        Target leakage occurs when the feature set includes information
        that directly determines the target variable. Since churn is defined
        as Recency > threshold, including Recency creates a deterministic
        relationship that the model will memorize.
        """
        mock_rfm, mock_raw = mock_data
        
        _, _, _, _, features, _, _ = predictor.prepare_ml_data(mock_rfm, mock_raw)
        
        # CRITICAL ASSERTION
        assert 'Recency' not in features, (
            "❌ CRITICAL: Target leakage detected! Recency should not be in the feature set. "
            "This would allow the model to memorize the churn definition rather than learn "
            "behavioral patterns."
        )
        
        print("✓ No target leakage: Recency correctly excluded from features")
    
    def test_feature_count(self, predictor, mock_data):
        """
        Validates that all 8 engineered features are present.
        
        Expected features:
        1. Frequency
        2. Monetary
        3. AOV
        4. Country_encoded
        5. Low_Frequency_Flag
        6. Low_Monetary_Flag
        7. Below_Median_AOV
        8. FM_Ratio
        """
        mock_rfm, mock_raw = mock_data
        
        _, _, _, _, features, _, _ = predictor.prepare_ml_data(mock_rfm, mock_raw)
        
        expected_count = 8
        assert len(features) == expected_count, (
            f"❌ Feature count mismatch. Expected {expected_count}, got {len(features)}. "
            f"Features: {features}"
        )
        
        # Check for specific critical features
        critical_features = [
            'Frequency', 'Monetary', 'AOV', 'Country_encoded',
            'Low_Frequency_Flag', 'Low_Monetary_Flag'
        ]
        
        for feature in critical_features:
            assert feature in features, f"❌ Missing critical feature: {feature}"
        
        print(f"✓ All {expected_count} features present: {features}")
    
    def test_churn_labeling(self, predictor, mock_data):
        """
        Validates churn target creation based on configured threshold.
        """
        mock_rfm, mock_raw = mock_data
        threshold = predictor.config.get('churn_threshold', 90)
        
        _, _, _, _, _, ml_data, _ = predictor.prepare_ml_data(mock_rfm, mock_raw)
        
        # Check churn logic for specific customers
        for cid in [1, 2, 3]:
            recency = ml_data.loc[cid, 'Recency']
            expected_churn = 1 if recency > threshold else 0
            actual_churn = ml_data.loc[cid, 'is_churn']
            
            assert actual_churn == expected_churn, (
                f"❌ Churn labeling error for Customer {cid}: "
                f"Recency={recency}, Expected={expected_churn}, Got={actual_churn}"
            )
        
        print(f"✓ Churn labeling correct (threshold={threshold})")
    
    def test_aov_calculation(self, predictor, mock_data):
        """
        Validates AOV (Average Order Value) calculation with edge case handling.
        
        Tests:
        1. Standard calculation: Monetary / Frequency
        2. Edge case: Frequency = 0 should not cause division by zero
        """
        mock_rfm, mock_raw = mock_data
        
        # Add a customer with Frequency=0 (edge case)
        mock_rfm = pd.concat([
            mock_rfm,
            pd.DataFrame({'CustomerID': [9999], 'Recency': [50], 'Frequency': [0], 'Monetary': [100]})
        ], ignore_index=True)
        
        mock_raw = pd.concat([
            mock_raw,
            pd.DataFrame({'CustomerID': [9999], 'Country': ['UK']})
        ], ignore_index=True)
        
        _, _, _, _, _, ml_data, _ = predictor.prepare_ml_data(mock_rfm, mock_raw)
        
        # Check standard AOV calculation
        cid_normal = 1
        expected_aov = ml_data.loc[cid_normal, 'Monetary'] / ml_data.loc[cid_normal, 'Frequency']
        actual_aov = ml_data.loc[cid_normal, 'AOV']
        
        assert np.isclose(actual_aov, expected_aov, rtol=0.01), (
            f"❌ AOV calculation error: Expected {expected_aov:.2f}, Got {actual_aov:.2f}"
        )
        
        # Check edge case: No division by zero
        aov_edge = ml_data.loc[9999, 'AOV']
        assert not np.isnan(aov_edge), "❌ AOV is NaN for Frequency=0 case"
        assert not np.isinf(aov_edge), "❌ AOV is Inf for Frequency=0 case"
        
        print("✓ AOV calculation correct (including Frequency=0 edge case)")
    
    def test_country_encoding(self, predictor, mock_data):
        """
        Validates LabelEncoder for Country feature.
        
        Tests:
        1. Encoding creates numeric values
        2. Range is valid (0 to n_countries-1)
        3. Consistent encoding (same country → same number)
        """
        mock_rfm, mock_raw = mock_data
        
        _, _, _, _, _, ml_data, le = predictor.prepare_ml_data(mock_rfm, mock_raw)
        
        # Check encoded values are numeric
        assert pd.api.types.is_numeric_dtype(ml_data['Country_encoded']), (
            "❌ Country_encoded should be numeric"
        )
        
        # Check encoding range
        n_countries = ml_data['Country'].nunique()
        encoded_values = ml_data['Country_encoded'].unique()
        
        assert encoded_values.min() >= 0, "❌ Encoded values should start at 0"
        assert encoded_values.max() < n_countries, (
            f"❌ Encoded values should be < {n_countries}, got max={encoded_values.max()}"
        )
        
        # Check consistency: Same country → Same encoding
        for country in ml_data['Country'].unique():
            encodings = ml_data[ml_data['Country'] == country]['Country_encoded'].unique()
            assert len(encodings) == 1, (
                f"❌ Inconsistent encoding for {country}: {encodings}"
            )
        
        print(f"✓ Country encoding valid: {n_countries} countries → [0, {n_countries-1}]")
    
    def test_train_test_split(self, predictor, mock_data):
        """
        Validates stratified train/test split.
        
        Tests:
        1. Split sizes match configuration
        2. Churn ratio is maintained in both sets (stratification)
        3. No data leakage between train and test
        """
        mock_rfm, mock_raw = mock_data
        
        X_train, X_test, y_train, y_test, _, _, _ = predictor.prepare_ml_data(mock_rfm, mock_raw)
        
        # Check split ratio
        test_size = predictor.config.get('test_size', 0.2)
        total = len(X_train) + len(X_test)
        actual_test_ratio = len(X_test) / total
        
        assert np.isclose(actual_test_ratio, test_size, atol=0.05), (
            f"❌ Test size mismatch: Expected ~{test_size:.1%}, Got {actual_test_ratio:.1%}"
        )
        
        # Check stratification (churn rate should be similar)
        train_churn_rate = y_train.mean()
        test_churn_rate = y_test.mean()
        
        assert np.isclose(train_churn_rate, test_churn_rate, atol=0.1), (
            f"❌ Stratification failed: Train churn={train_churn_rate:.1%}, "
            f"Test churn={test_churn_rate:.1%}"
        )
        
        print(f"✓ Split integrity: Train={len(X_train)}, Test={len(X_test)}, "
              f"Churn rate similar ({train_churn_rate:.1%} vs {test_churn_rate:.1%})")
    
    def test_engineered_flags(self, predictor, mock_data):
        """
        Validates binary flag features are correctly calculated.
        
        Tests:
        1. Low_Frequency_Flag: Bottom 25% of Frequency
        2. Low_Monetary_Flag: Bottom 25% of Monetary
        3. Below_Median_AOV: Below median AOV
        """
        mock_rfm, mock_raw = mock_data
        
        _, _, _, _, _, ml_data, _ = predictor.prepare_ml_data(mock_rfm, mock_raw)
        
        # Check Low_Frequency_Flag
        freq_q25 = ml_data['Frequency'].quantile(0.25)
        expected_low_freq = (ml_data['Frequency'] < freq_q25).astype(int)
        
        assert (ml_data['Low_Frequency_Flag'] == expected_low_freq).all(), (
            "❌ Low_Frequency_Flag calculation error"
        )
        
        # Check Low_Monetary_Flag
        mon_q25 = ml_data['Monetary'].quantile(0.25)
        expected_low_mon = (ml_data['Monetary'] < mon_q25).astype(int)
        
        assert (ml_data['Low_Monetary_Flag'] == expected_low_mon).all(), (
            "❌ Low_Monetary_Flag calculation error"
        )
        
        # Check Below_Median_AOV
        aov_median = ml_data['AOV'].median()
        expected_below_aov = (ml_data['AOV'] < aov_median).astype(int)
        
        assert (ml_data['Below_Median_AOV'] == expected_below_aov).all(), (
            "❌ Below_Median_AOV calculation error"
        )
        
        print("✓ All engineered flags calculated correctly")
    
    def test_no_missing_values(self, predictor, mock_data):
        """
        Ensures feature engineering doesn't introduce missing values.
        """
        mock_rfm, mock_raw = mock_data
        
        X_train, X_test, _, _, features, ml_data, _ = predictor.prepare_ml_data(mock_rfm, mock_raw)
        
        # Check training data
        assert not X_train.isnull().any().any(), (
            f"❌ Missing values in X_train: {X_train.isnull().sum()[X_train.isnull().any()]}"
        )
        
        # Check test data
        assert not X_test.isnull().any().any(), (
            f"❌ Missing values in X_test: {X_test.isnull().sum()[X_test.isnull().any()]}"
        )
        
        print("✓ No missing values in training or test data")


def test_churn_pipeline_integration():
    """
    High-level integration test for the entire ML pipeline.
    
    This test validates the end-to-end workflow:
    1. Data preparation
    2. Feature engineering
    3. Model training
    4. Prediction generation
    """
    predictor = ChurnPredictor()
    
    # Create realistic mock data (100 customers)
    np.random.seed(42)
    n = 100
    
    mock_rfm = pd.DataFrame({
        'CustomerID': range(1, n + 1),
        'Recency': np.random.randint(5, 200, n),
        'Frequency': np.random.randint(1, 20, n),
        'Monetary': np.random.uniform(20, 1500, n)
    })
    
    mock_raw = pd.DataFrame({
        'CustomerID': range(1, n + 1),
        'Country': np.random.choice(['UK', 'Germany', 'France', 'Spain'], n)
    })
    
    # Step 1: Data preparation
    X_train, X_test, y_train, y_test, features, ml_data, le = predictor.prepare_ml_data(
        mock_rfm, mock_raw
    )
    
    # Step 2: Model training
    model = predictor.train_model(X_train, y_train)
    
    # Step 3: Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Assertions
    assert len(y_pred) == len(y_test), "❌ Prediction length mismatch"
    assert all(prob >= 0 and prob <= 1 for prob in y_probs), "❌ Invalid probabilities"
    assert set(y_pred) == {0, 1}, "❌ Predictions should be binary"
    
    print("✓ Integration test passed: Full pipeline executed successfully")


if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])