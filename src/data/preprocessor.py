"""
Data Preprocessor for IEEE-CIS Fraud Detection Dataset

This module provides comprehensive data preprocessing capabilities including
dataset merging, bank partitioning, missing value handling, temporal splitting,
and categorical encoding for federated learning scenarios.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class Data_Preprocessor:
    """
    Comprehensive data preprocessor for IEEE-CIS fraud detection dataset.

    Handles dataset merging, bank partitioning, missing value processing,
    temporal splitting, and categorical encoding for federated learning.
    """

    def __init__(self, missing_threshold: float = 0.5, random_state: int = 42):
        """
        Initialize data preprocessor.

        Args:
            missing_threshold: Threshold for dropping columns with missing values (0.5 = 50%)
            random_state: Random seed for reproducible results
        """
        self.missing_threshold = missing_threshold
        self.random_state = random_state
        self.label_encoders = {}
        self.preprocessing_stats = {}

    def merge_datasets(self, transaction_df: pd.DataFrame, identity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge transaction and identity datasets on TransactionID.

        Args:
            transaction_df: Transaction dataset
            identity_df: Identity dataset

        Returns:
            Merged dataset with combined features
        """
        logger.info("Merging transaction and identity datasets")

        # Validate input datasets
        if "TransactionID" not in transaction_df.columns:
            raise ValueError("Transaction dataset missing TransactionID column")
        if "TransactionID" not in identity_df.columns:
            raise ValueError("Identity dataset missing TransactionID column")

        initial_transaction_count = len(transaction_df)
        initial_identity_count = len(identity_df)

        # Perform left join to keep all transactions
        merged_df = transaction_df.merge(identity_df, on="TransactionID", how="left")

        # Log merge statistics
        merge_stats = {
            "initial_transactions": initial_transaction_count,
            "initial_identities": initial_identity_count,
            "merged_records": len(merged_df),
            "identity_match_rate": (
                (
                    len(merged_df.dropna(subset=identity_df.columns.drop("TransactionID")))
                    / initial_transaction_count
                    * 100
                )
                if initial_transaction_count > 0
                else 0.0
            ),
        }

        self.preprocessing_stats["merge"] = merge_stats

        logger.info(
            f"Merged datasets: {merge_stats['merged_records']} records, "
            f"{merge_stats['identity_match_rate']:.2f}% identity match rate"
        )

        return merged_df

    def partition_by_product_cd(self, df: pd.DataFrame, num_banks: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Partition dataset by ProductCD to simulate different banks.

        Args:
            df: Merged dataset
            num_banks: Number of banks to simulate (default: 3)

        Returns:
            Dictionary mapping bank names to their data partitions
        """
        logger.info(f"Partitioning dataset by ProductCD for {num_banks} banks")

        if "ProductCD" not in df.columns:
            raise ValueError("Dataset missing ProductCD column for partitioning")

        # Get unique ProductCD values
        product_codes = df["ProductCD"].dropna().unique()
        logger.info(f"Found ProductCD values: {product_codes}")

        # Assign ProductCDs to banks in round-robin fashion
        bank_assignments = {}
        for i, product_cd in enumerate(product_codes):
            bank_id = f"bank_{(i % num_banks) + 1}"
            bank_assignments[product_cd] = bank_id

        # Create bank partitions
        bank_partitions = {}
        partition_stats = {}

        for bank_id in [f"bank_{i+1}" for i in range(num_banks)]:
            # Get ProductCDs assigned to this bank
            assigned_products = [pc for pc, bid in bank_assignments.items() if bid == bank_id]

            # Filter data for this bank
            bank_data = df[df["ProductCD"].isin(assigned_products)].copy()

            # Handle records with missing ProductCD - distribute evenly
            missing_product_data = df[df["ProductCD"].isna()]
            if len(missing_product_data) > 0:
                # Split missing ProductCD data evenly among banks
                bank_idx = int(bank_id.split("_")[1]) - 1
                missing_split = np.array_split(missing_product_data, num_banks)
                if bank_idx < len(missing_split):
                    bank_data = pd.concat([bank_data, missing_split[bank_idx]], ignore_index=True)

            bank_partitions[bank_id] = bank_data

            # Calculate statistics
            fraud_count = bank_data["isFraud"].sum() if "isFraud" in bank_data.columns else 0
            partition_stats[bank_id] = {
                "total_records": len(bank_data),
                "fraud_records": fraud_count,
                "fraud_rate": fraud_count / len(bank_data) * 100 if len(bank_data) > 0 else 0,
                "assigned_products": assigned_products,
            }

            logger.info(
                f"{bank_id}: {len(bank_data)} records, "
                f"fraud rate: {partition_stats[bank_id]['fraud_rate']:.2f}%, "
                f"products: {assigned_products}"
            )

        self.preprocessing_stats["partitioning"] = partition_stats
        return bank_partitions

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with 50% threshold and imputation strategies.

        Args:
            df: Input dataset

        Returns:
            Dataset with missing values handled
        """
        logger.info("Handling missing values")

        initial_shape = df.shape

        # Calculate missing value percentages
        missing_percentages = df.isnull().sum() / len(df)

        # Drop columns with more than threshold missing values
        columns_to_drop = missing_percentages[missing_percentages > self.missing_threshold].index.tolist()

        # Never drop essential columns
        essential_columns = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
        columns_to_drop = [col for col in columns_to_drop if col not in essential_columns]

        if columns_to_drop:
            logger.info(f"Dropping {len(columns_to_drop)} columns with >{self.missing_threshold*100}% missing values")
            df = df.drop(columns=columns_to_drop)

        # Impute remaining missing values
        imputation_stats = {}

        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ["int64", "int8", "float64"]:
                    # Numerical columns: impute with median
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                    imputation_stats[column] = {"method": "median", "value": median_value}

                elif df[column].dtype == "category" or df[column].dtype == "object":
                    # Categorical columns: impute with mode or 'Unknown'
                    if df[column].mode().empty:
                        fill_value = "Unknown"
                    else:
                        fill_value = df[column].mode().iloc[0]
                    df[column] = df[column].fillna(fill_value)
                    imputation_stats[column] = {"method": "mode", "value": fill_value}

        final_shape = df.shape

        missing_stats = {
            "initial_shape": initial_shape,
            "final_shape": final_shape,
            "columns_dropped": len(columns_to_drop),
            "columns_imputed": len(imputation_stats),
            "dropped_columns": columns_to_drop,
        }

        self.preprocessing_stats["missing_values"] = missing_stats

        logger.info(
            f"Missing value handling complete: {initial_shape} -> {final_shape}, "
            f"dropped {len(columns_to_drop)} columns, imputed {len(imputation_stats)} columns"
        )

        return df

    def temporal_split(
        self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset temporally based on TransactionDT with 80/10/10 ratios.

        Args:
            df: Input dataset
            train_ratio: Training set ratio (default: 0.8)
            val_ratio: Validation set ratio (default: 0.1)
            test_ratio: Test set ratio (default: 0.1)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Performing temporal split: {train_ratio}/{val_ratio}/{test_ratio}")

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        if "TransactionDT" not in df.columns:
            raise ValueError("Dataset missing TransactionDT column for temporal splitting")

        # Sort by transaction datetime
        df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)

        # Calculate split indices
        total_records = len(df_sorted)
        train_end = int(total_records * train_ratio)
        val_end = int(total_records * (train_ratio + val_ratio))

        # Perform splits
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        # Calculate split statistics
        split_stats = {
            "total_records": total_records,
            "train_records": len(train_df),
            "val_records": len(val_df),
            "test_records": len(test_df),
            "train_ratio_actual": len(train_df) / total_records,
            "val_ratio_actual": len(val_df) / total_records,
            "test_ratio_actual": len(test_df) / total_records,
        }

        # Add fraud statistics if available
        if "isFraud" in df.columns:
            split_stats.update(
                {
                    "train_fraud_rate": train_df["isFraud"].mean() * 100,
                    "val_fraud_rate": val_df["isFraud"].mean() * 100,
                    "test_fraud_rate": test_df["isFraud"].mean() * 100,
                }
            )

        self.preprocessing_stats["temporal_split"] = split_stats

        logger.info(f"Temporal split complete: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df

    def encode_categorical_features(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, ...]:
        """
        Encode categorical features using LabelEncoder.

        Args:
            train_df: Training dataset
            val_df: Validation dataset (optional)
            test_df: Test dataset (optional)

        Returns:
            Tuple of encoded datasets in same order as input
        """
        logger.info("Encoding categorical features")

        # Identify categorical columns
        categorical_columns = []
        for col in train_df.columns:
            if (
                train_df[col].dtype == "category"
                or train_df[col].dtype == "object"
                or col
                in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
                + [f"id_{i:02d}" for i in [11, 12, 15, 16, 20]]
            ):
                categorical_columns.append(col)

        logger.info(f"Found {len(categorical_columns)} categorical columns to encode")

        # Fit encoders on training data
        encoded_train = train_df.copy()
        encoding_stats = {}

        for col in categorical_columns:
            if col in encoded_train.columns:
                # Convert to string and handle missing values
                if encoded_train[col].dtype == "category":
                    # For categorical columns, add 'Unknown' to categories first
                    if "Unknown" not in encoded_train[col].cat.categories:
                        encoded_train[col] = encoded_train[col].cat.add_categories(["Unknown"])
                    encoded_train[col] = encoded_train[col].fillna("Unknown")
                    encoded_train[col] = encoded_train[col].astype(str)
                else:
                    encoded_train[col] = encoded_train[col].fillna("Unknown").astype(str)

                # Fit label encoder
                encoder = LabelEncoder()
                encoded_train[col] = encoder.fit_transform(encoded_train[col])

                self.label_encoders[col] = encoder
                encoding_stats[col] = {
                    "unique_values": len(encoder.classes_),
                    "classes": encoder.classes_[:10].tolist(),  # First 10 classes for logging
                }

        # Apply encoders to validation and test sets
        results = [encoded_train]

        for df_name, df in [("validation", val_df), ("test", test_df)]:
            if df is not None:
                encoded_df = df.copy()

                for col in categorical_columns:
                    if col in encoded_df.columns and col in self.label_encoders:
                        # Convert to string and handle missing values
                        if encoded_df[col].dtype == "category":
                            # For categorical columns, add 'Unknown' to categories first if needed
                            if "Unknown" not in encoded_df[col].cat.categories:
                                encoded_df[col] = encoded_df[col].cat.add_categories(["Unknown"])
                            encoded_df[col] = encoded_df[col].fillna("Unknown")
                            encoded_df[col] = encoded_df[col].astype(str)
                        else:
                            encoded_df[col] = encoded_df[col].fillna("Unknown").astype(str)

                        # Handle unseen categories
                        encoder = self.label_encoders[col]
                        col_values = encoded_df[col]

                        # Map unseen categories to 'Unknown' if it exists, otherwise to first class
                        unknown_mask = ~col_values.isin(encoder.classes_)
                        if unknown_mask.any():
                            if "Unknown" in encoder.classes_:
                                col_values.loc[unknown_mask] = "Unknown"
                            else:
                                logger.warning(f"Found unseen categories in {col} for {df_name} set")
                                col_values.loc[unknown_mask] = encoder.classes_[0]  # Use first class as fallback

                        encoded_df[col] = encoder.transform(col_values)

                results.append(encoded_df)
            else:
                results.append(None)

        self.preprocessing_stats["categorical_encoding"] = {
            "encoded_columns": len(categorical_columns),
            "column_stats": encoding_stats,
        }

        logger.info(f"Categorical encoding complete: {len(categorical_columns)} columns encoded")

        # Return results, filtering out None values
        return tuple(result for result in results if result is not None)

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Return comprehensive preprocessing statistics."""
        return self.preprocessing_stats.copy()

    def save_encoders(self, filepath: Path) -> None:
        """Save label encoders for future use."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.label_encoders, f)

        logger.info(f"Saved {len(self.label_encoders)} label encoders to {filepath}")

    def load_encoders(self, filepath: Path) -> None:
        """Load previously saved label encoders."""
        import pickle

        with open(filepath, "rb") as f:
            self.label_encoders = pickle.load(f)

        logger.info(f"Loaded {len(self.label_encoders)} label encoders from {filepath}")

    def preprocess_full_pipeline(
        self, transaction_df: pd.DataFrame, identity_df: pd.DataFrame, num_banks: int = 3
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Execute complete preprocessing pipeline for federated learning.

        Args:
            transaction_df: Transaction dataset
            identity_df: Identity dataset
            num_banks: Number of banks to simulate

        Returns:
            Dictionary with bank data splits: {bank_id: {train, val, test}}
        """
        logger.info("Starting full preprocessing pipeline")

        # Step 1: Merge datasets
        merged_df = self.merge_datasets(transaction_df, identity_df)

        # Step 2: Handle missing values
        cleaned_df = self.handle_missing_values(merged_df)

        # Step 3: Partition by banks
        bank_partitions = self.partition_by_product_cd(cleaned_df, num_banks)

        # Step 4: Temporal split and encoding for each bank
        bank_splits = {}

        for bank_id, bank_data in bank_partitions.items():
            logger.info(f"Processing {bank_id}")

            # Temporal split
            train_df, val_df, test_df = self.temporal_split(bank_data)

            # Categorical encoding (fit on train, transform val/test)
            encoded_train, encoded_val, encoded_test = self.encode_categorical_features(train_df, val_df, test_df)

            bank_splits[bank_id] = {"train": encoded_train, "val": encoded_val, "test": encoded_test}

            logger.info(
                f"{bank_id} preprocessing complete: "
                f"train={len(encoded_train)}, val={len(encoded_val)}, test={len(encoded_test)}"
            )

        logger.info("Full preprocessing pipeline complete")
        return bank_splits
