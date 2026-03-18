"""
CSV Parser for IEEE-CIS Fraud Detection Dataset

This module provides robust CSV parsing capabilities for the IEEE-CIS dataset
with error handling, data type validation, and automatic correction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class CSV_Parser:
    """
    Robust CSV parser for IEEE-CIS fraud detection dataset.
    
    Handles malformed records, data type validation, and automatic correction
    for both transaction and identity datasets.
    """
    
    def __init__(self):
        """Initialize CSV parser with IEEE-CIS specific configurations."""
        self.transaction_dtypes = self._get_transaction_dtypes()
        self.identity_dtypes = self._get_identity_dtypes()
        self.parse_errors = []
        
    def _get_transaction_dtypes(self) -> Dict[str, str]:
        """Define expected data types for transaction dataset columns."""
        return {
            'TransactionID': 'int64',
            'isFraud': 'int8',
            'TransactionDT': 'int64',
            'TransactionAmt': 'float64',
            'ProductCD': 'category',
            'card1': 'int64',
            'card2': 'float64',
            'card3': 'float64',
            'card4': 'category',
            'card5': 'float64',
            'card6': 'category',
            'addr1': 'float64',
            'addr2': 'float64',
            'dist1': 'float64',
            'dist2': 'float64',
            'P_emaildomain': 'category',
            'R_emaildomain': 'category'
        }
    
    def _get_identity_dtypes(self) -> Dict[str, str]:
        """Define expected data types for identity dataset columns."""
        return {
            'TransactionID': 'int64',
            'id_01': 'float64',
            'id_02': 'float64',
            'id_03': 'float64',
            'id_04': 'float64',
            'id_05': 'float64',
            'id_06': 'float64',
            'id_07': 'float64',
            'id_08': 'float64',
            'id_09': 'float64',
            'id_10': 'float64',
            'id_11': 'category',
            'id_12': 'category',
            'id_13': 'float64',
            'id_14': 'float64',
            'id_15': 'category',
            'id_16': 'category',
            'id_17': 'float64',
            'id_18': 'float64',
            'id_19': 'float64',
            'id_20': 'category',
            'DeviceType': 'category',
            'DeviceInfo': 'category'
        }
    
    def parse_csv(self, file_path: Path, dataset_type: str = 'auto') -> pd.DataFrame:
        """
        Parse CSV file with robust error handling and data validation.
        
        Args:
            file_path: Path to CSV file
            dataset_type: 'transaction', 'identity', or 'auto' for auto-detection
            
        Returns:
            Parsed and validated DataFrame
            
        Raises:
            ValueError: If file cannot be parsed or validated
        """
        try:
            logger.info(f"Parsing CSV file: {file_path}")
            
            # Auto-detect dataset type if not specified
            if dataset_type == 'auto':
                dataset_type = self._detect_dataset_type(file_path)
            
            # Read CSV with error handling
            df = self._read_csv_robust(file_path)
            
            # Validate and correct data types
            df = self._validate_and_correct_dtypes(df, dataset_type)
            
            # Handle malformed records
            df = self._handle_malformed_records(df, dataset_type)
            
            logger.info(f"Successfully parsed {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            error_msg = f"Failed to parse CSV file {file_path}: {str(e)}"
            logger.error(error_msg)
            self.parse_errors.append(error_msg)
            raise ValueError(error_msg)
    
    def _detect_dataset_type(self, file_path: Path) -> str:
        """Auto-detect whether file is transaction or identity dataset."""
        filename = file_path.name.lower()
        if 'transaction' in filename:
            return 'transaction'
        elif 'identity' in filename:
            return 'identity'
        else:
            # Try to detect by reading first few rows
            try:
                sample_df = pd.read_csv(file_path, nrows=5)
                if 'isFraud' in sample_df.columns:
                    return 'transaction'
                elif 'DeviceType' in sample_df.columns:
                    return 'identity'
                else:
                    raise ValueError(f"Cannot auto-detect dataset type for {file_path}")
            except Exception as e:
                raise ValueError(f"Cannot auto-detect dataset type for {file_path}: {str(e)}")
    
    def _read_csv_robust(self, file_path: Path) -> pd.DataFrame:
        """Read CSV with multiple fallback strategies for robustness."""
        try:
            # First attempt: standard read
            return pd.read_csv(file_path, low_memory=False)
        except UnicodeDecodeError:
            # Second attempt: different encoding
            logger.warning(f"Unicode error, trying latin-1 encoding for {file_path}")
            return pd.read_csv(file_path, encoding='latin-1', low_memory=False)
        except pd.errors.ParserError as e:
            # Third attempt: skip bad lines
            logger.warning(f"Parser error, skipping bad lines for {file_path}: {str(e)}")
            return pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    
    def _validate_and_correct_dtypes(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validate and automatically correct data types."""
        expected_dtypes = (self.transaction_dtypes if dataset_type == 'transaction' 
                          else self.identity_dtypes)
        
        corrections_made = []
        
        for column, expected_dtype in expected_dtypes.items():
            if column in df.columns:
                try:
                    # Handle categorical columns
                    if expected_dtype == 'category':
                        df[column] = df[column].astype('category')
                    
                    # Handle numeric columns with potential string contamination
                    elif expected_dtype in ['int64', 'int8', 'float64']:
                        if df[column].dtype == 'object':
                            # Try to convert, coercing errors to NaN
                            df[column] = pd.to_numeric(df[column], errors='coerce')
                        
                        # Convert to target dtype
                        if expected_dtype in ['int64', 'int8']:
                            # Fill NaN with 0 for integer columns before conversion
                            df[column] = df[column].fillna(0).astype(expected_dtype)
                        else:
                            df[column] = df[column].astype(expected_dtype)
                    
                    corrections_made.append(f"{column}: {expected_dtype}")
                    
                except Exception as e:
                    logger.warning(f"Could not convert {column} to {expected_dtype}: {str(e)}")
                    self.parse_errors.append(f"Type conversion error for {column}: {str(e)}")
        
        if corrections_made:
            logger.info(f"Data type corrections made: {len(corrections_made)} columns")
        
        return df
    
    def _handle_malformed_records(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Handle and clean malformed records."""
        initial_rows = len(df)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # For transaction data, ensure TransactionID and isFraud are present
        if dataset_type == 'transaction':
            df = df.dropna(subset=['TransactionID'])
            if 'isFraud' in df.columns:
                # Ensure isFraud is 0 or 1
                df = df[df['isFraud'].isin([0, 1])]
        
        # For identity data, ensure TransactionID is present
        elif dataset_type == 'identity':
            df = df.dropna(subset=['TransactionID'])
        
        # Remove duplicate TransactionIDs
        duplicates_before = df.duplicated(subset=['TransactionID']).sum()
        if duplicates_before > 0:
            df = df.drop_duplicates(subset=['TransactionID'], keep='first')
            logger.warning(f"Removed {duplicates_before} duplicate TransactionIDs")
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"Cleaned {rows_removed} malformed records ({rows_removed/initial_rows*100:.2f}%)")
        
        return df
    
    def get_parse_errors(self) -> List[str]:
        """Return list of parsing errors encountered."""
        return self.parse_errors.copy()
    
    def clear_parse_errors(self) -> None:
        """Clear the list of parsing errors."""
        self.parse_errors.clear()
    
    def validate_dataset_integrity(self, df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
        """
        Validate dataset integrity and return summary statistics.
        
        Args:
            df: DataFrame to validate
            dataset_type: 'transaction' or 'identity'
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            'dataset_type': dataset_type,
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'validation_errors': []
        }
        
        # Check for required columns
        required_cols = ['TransactionID']
        if dataset_type == 'transaction':
            required_cols.append('isFraud')
        
        for col in required_cols:
            if col not in df.columns:
                validation_results['validation_errors'].append(f"Missing required column: {col}")
        
        # Check TransactionID uniqueness
        if 'TransactionID' in df.columns:
            duplicate_ids = df['TransactionID'].duplicated().sum()
            if duplicate_ids > 0:
                validation_results['validation_errors'].append(f"Found {duplicate_ids} duplicate TransactionIDs")
        
        # Check fraud label distribution for transaction data
        if dataset_type == 'transaction' and 'isFraud' in df.columns:
            fraud_dist = df['isFraud'].value_counts()
            validation_results['fraud_distribution'] = fraud_dist.to_dict()
            fraud_rate = fraud_dist.get(1, 0) / len(df) * 100
            validation_results['fraud_rate_percent'] = fraud_rate
        
        return validation_results