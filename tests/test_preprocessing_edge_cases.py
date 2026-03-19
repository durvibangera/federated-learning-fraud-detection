"""
Unit Tests for IEEE-CIS Preprocessing Edge Cases

Tests specific edge cases and error conditions in data preprocessing:
- Empty datasets
- All missing values
- Invalid ProductCD values
- Temporal split edge cases
- Encoding unseen categories
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import Data_Preprocessor


class TestPreprocessingEdgeCases:
    """Unit tests for data preprocessing edge cases."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        preprocessor = Data_Preprocessor()

        # Create empty transaction and identity DataFrames
        transaction_df = pd.DataFrame(columns=["TransactionID", "TransactionDT", "isFraud"])
        identity_df = pd.DataFrame(columns=["TransactionID"])

        # Should handle empty merge gracefully
        merged = preprocessor.merge_datasets(transaction_df, identity_df)

        assert len(merged) == 0
        assert "TransactionID" in merged.columns

    def test_all_missing_values_column(self):
        """Test dropping columns with 100% missing values."""
        preprocessor = Data_Preprocessor(missing_threshold=0.5)

        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "TransactionDT": [100.0, 200.0, 300.0],
                "all_missing": [None, None, None],
                "isFraud": [0, 1, 0],
            }
        )

        cleaned = preprocessor.handle_missing_values(df)

        # Column with all missing values should be dropped
        assert "all_missing" not in cleaned.columns
        assert "TransactionID" in cleaned.columns
        assert "isFraud" in cleaned.columns

    def test_missing_threshold_boundary(self):
        """Test missing value threshold at exactly 50%."""
        preprocessor = Data_Preprocessor(missing_threshold=0.5)

        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3, 4],
                "exactly_50_missing": [1.0, None, 2.0, None],  # Exactly 50% missing
                "over_50_missing": [1.0, None, None, None],  # 75% missing
                "isFraud": [0, 1, 0, 1],
            }
        )

        cleaned = preprocessor.handle_missing_values(df)

        # Column with >50% missing should be dropped
        assert "over_50_missing" not in cleaned.columns
        # Column with exactly 50% should be kept
        assert "exactly_50_missing" in cleaned.columns

    def test_invalid_product_cd_values(self):
        """Test handling of invalid ProductCD values."""
        preprocessor = Data_Preprocessor()

        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3, 4, 5],
                "TransactionDT": [100.0, 200.0, 300.0, 400.0, 500.0],
                "ProductCD": ["W", "H", None, "INVALID", "S"],
                "isFraud": [0, 1, 0, 1, 0],
            }
        )

        # Should handle None and invalid values gracefully
        partitions = preprocessor.partition_by_product_cd(df, num_banks=3)

        # Should create 3 partitions
        assert len(partitions) == 3

        # Total records should be preserved
        total_records = sum(len(partition) for partition in partitions.values())
        assert total_records == len(df)

    def test_temporal_split_with_duplicate_timestamps(self):
        """Test temporal split when multiple records have same timestamp."""
        preprocessor = Data_Preprocessor()

        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3, 4, 5, 6],
                "TransactionDT": [100.0, 100.0, 200.0, 200.0, 300.0, 300.0],  # Duplicates
                "isFraud": [0, 1, 0, 1, 0, 1],
            }
        )

        train, val, test = preprocessor.temporal_split(df, 0.5, 0.25, 0.25)

        # Should split correctly despite duplicates
        assert len(train) + len(val) + len(test) == len(df)
        assert len(train) >= 2
        assert len(val) >= 1
        assert len(test) >= 1

    def test_temporal_split_maintains_order(self):
        """Test that temporal split maintains chronological order."""
        preprocessor = Data_Preprocessor()

        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3, 4, 5],
                "TransactionDT": [500.0, 100.0, 300.0, 200.0, 400.0],  # Unsorted
                "isFraud": [0, 1, 0, 1, 0],
            }
        )

        train, val, test = preprocessor.temporal_split(df, 0.6, 0.2, 0.2)

        # Train should have earliest timestamps
        # Test should have latest timestamps
        if len(train) > 0 and len(test) > 0:
            assert train["TransactionDT"].max() <= test["TransactionDT"].min()

    def test_encoding_unseen_categories(self):
        """Test encoding when validation/test sets have unseen categories."""
        preprocessor = Data_Preprocessor()

        train_df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "ProductCD": ["W", "H", "W"],
                "card4": ["visa", "mastercard", "visa"],
                "isFraud": [0, 1, 0],
            }
        )

        test_df = pd.DataFrame(
            {
                "TransactionID": [4, 5],
                "ProductCD": ["R", "S"],  # Unseen categories
                "card4": ["amex", "discover"],  # Unseen categories
                "isFraud": [1, 0],
            }
        )

        # Should handle unseen categories gracefully
        encoded_train, encoded_test = preprocessor.encode_categorical_features(train_df, test_df=test_df)

        # Should not raise error
        assert len(encoded_train) == len(train_df)
        assert len(encoded_test) == len(test_df)

        # Encoded values should be integers
        assert encoded_train["ProductCD"].dtype in [np.int32, np.int64]
        assert encoded_test["ProductCD"].dtype in [np.int32, np.int64]

    def test_single_record_partition(self):
        """Test partitioning with very small dataset."""
        preprocessor = Data_Preprocessor()

        df = pd.DataFrame({"TransactionID": [1], "TransactionDT": [100.0], "ProductCD": ["W"], "isFraud": [0]})

        partitions = preprocessor.partition_by_product_cd(df, num_banks=3)

        # Should create partitions even with single record
        assert len(partitions) == 3

        # Single record should go to one bank
        non_empty_partitions = [p for p in partitions.values() if len(p) > 0]
        assert len(non_empty_partitions) == 1
        assert len(non_empty_partitions[0]) == 1

    def test_missing_essential_columns(self):
        """Test error handling when essential columns are missing."""
        preprocessor = Data_Preprocessor()

        # Missing TransactionID
        transaction_df = pd.DataFrame({"TransactionDT": [100.0, 200.0], "isFraud": [0, 1]})
        identity_df = pd.DataFrame({"TransactionID": [1, 2]})

        with pytest.raises(ValueError, match="TransactionID"):
            preprocessor.merge_datasets(transaction_df, identity_df)

    def test_temporal_split_invalid_ratios(self):
        """Test error handling for invalid split ratios."""
        preprocessor = Data_Preprocessor()

        df = pd.DataFrame({"TransactionID": [1, 2, 3], "TransactionDT": [100.0, 200.0, 300.0], "isFraud": [0, 1, 0]})

        # Ratios don't sum to 1.0
        with pytest.raises(ValueError, match="sum to 1.0"):
            preprocessor.temporal_split(df, 0.5, 0.3, 0.3)

    def test_extreme_class_imbalance(self):
        """Test handling of extreme class imbalance (all fraud or all legitimate)."""
        preprocessor = Data_Preprocessor()

        # All fraud
        df_all_fraud = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3, 4, 5],
                "TransactionDT": [100.0, 200.0, 300.0, 400.0, 500.0],
                "ProductCD": ["W", "H", "R", "S", "C"],
                "isFraud": [1, 1, 1, 1, 1],
            }
        )

        partitions = preprocessor.partition_by_product_cd(df_all_fraud, num_banks=3)

        # Should handle extreme imbalance
        for bank_id, partition in partitions.items():
            if len(partition) > 0:
                assert partition["isFraud"].mean() == 1.0

    def test_mixed_data_types_in_categorical(self):
        """Test handling of mixed data types in categorical columns."""
        preprocessor = Data_Preprocessor()

        train_df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "ProductCD": ["W", "H", "W"],
                "mixed_col": [1, "two", 3.0],  # Mixed types
                "isFraud": [0, 1, 0],
            }
        )

        # Should convert to string and encode
        result = preprocessor.encode_categorical_features(train_df)
        encoded_train = result[0] if isinstance(result, tuple) else result

        assert len(encoded_train) == len(train_df)

    def test_very_high_cardinality_categorical(self):
        """Test encoding of very high cardinality categorical features."""
        preprocessor = Data_Preprocessor()

        # Create feature with unique value per row (extreme cardinality)
        train_df = pd.DataFrame(
            {
                "TransactionID": list(range(1, 101)),
                "high_cardinality": [f"value_{i}" for i in range(100)],
                "isFraud": [0] * 100,
            }
        )

        # Should handle high cardinality
        result = preprocessor.encode_categorical_features(train_df)
        encoded_train = result[0] if isinstance(result, tuple) else result

        assert len(encoded_train) == len(train_df)
        assert len(preprocessor.label_encoders["high_cardinality"].classes_) == 100

    def test_preprocessing_stats_tracking(self):
        """Test that preprocessing statistics are tracked correctly."""
        preprocessor = Data_Preprocessor()

        transaction_df = pd.DataFrame(
            {"TransactionID": [1, 2, 3], "TransactionDT": [100.0, 200.0, 300.0], "isFraud": [0, 1, 0]}
        )
        identity_df = pd.DataFrame({"TransactionID": [1, 2, 3]})

        merged = preprocessor.merge_datasets(transaction_df, identity_df)

        # Check that stats were recorded
        stats = preprocessor.get_preprocessing_stats()
        assert "merge" in stats
        assert stats["merge"]["merged_records"] == 3
        assert stats["merge"]["initial_transactions"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
