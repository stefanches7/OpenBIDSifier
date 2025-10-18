"""
Label encoding processor for neuroimaging metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
import logging

from ..utils.logger import setup_logger


class LabelEncoder:
    """
    Handles label encoding for neuroimaging metadata.
    
    Features:
    - Categorical variable encoding
    - Ordinal encoding
    - One-hot encoding
    - Target encoding
    - Custom encoding schemes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the label encoder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = setup_logger("LabelEncoder")
        self.encoding_type = self.config.get("encoding_type", "label")
        self.encodings = {}
    
    def fit_label_encoding(self, series: pd.Series) -> Dict[str, int]:
        """
        Fit label encoding for a categorical series.
        
        Args:
            series: Pandas series with categorical data
            
        Returns:
            Mapping dictionary
        """
        unique_values = series.dropna().unique()
        encoding = {val: idx for idx, val in enumerate(sorted(unique_values))}
        return encoding
    
    def apply_label_encoding(
        self,
        series: pd.Series,
        encoding: Optional[Dict[str, int]] = None
    ) -> pd.Series:
        """
        Apply label encoding to a series.
        
        Args:
            series: Input series
            encoding: Encoding mapping (if None, will fit new encoding)
            
        Returns:
            Encoded series
        """
        if encoding is None:
            encoding = self.fit_label_encoding(series)
        
        return series.map(encoding)
    
    def fit_onehot_encoding(self, series: pd.Series) -> List[str]:
        """
        Fit one-hot encoding for a categorical series.
        
        Args:
            series: Pandas series with categorical data
            
        Returns:
            List of unique categories
        """
        return sorted(series.dropna().unique().tolist())
    
    def apply_onehot_encoding(
        self,
        series: pd.Series,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply one-hot encoding to a series.
        
        Args:
            series: Input series
            categories: List of categories (if None, will fit new)
            
        Returns:
            One-hot encoded DataFrame
        """
        if categories is None:
            categories = self.fit_onehot_encoding(series)
        
        # Create one-hot encoded columns
        onehot_df = pd.DataFrame()
        for cat in categories:
            column_name = f"{series.name}_{cat}"
            onehot_df[column_name] = (series == cat).astype(int)
        
        return onehot_df
    
    def fit_ordinal_encoding(
        self,
        series: pd.Series,
        order: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Fit ordinal encoding with specified order.
        
        Args:
            series: Pandas series with ordinal data
            order: Ordered list of categories (if None, sorts alphabetically)
            
        Returns:
            Mapping dictionary
        """
        if order is None:
            order = sorted(series.dropna().unique())
        
        encoding = {val: idx for idx, val in enumerate(order)}
        return encoding
    
    def encode_metadata(
        self,
        df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        ordinal_columns: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Encode categorical and ordinal columns in metadata.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            ordinal_columns: Dict mapping ordinal columns to their order
            
        Returns:
            Encoded DataFrame
        """
        encoded_df = df.copy()
        
        # Auto-detect categorical columns if not specified
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle ordinal columns first
        if ordinal_columns:
            for col, order in ordinal_columns.items():
                if col in encoded_df.columns:
                    encoding = self.fit_ordinal_encoding(df[col], order)
                    encoded_df[col] = self.apply_label_encoding(df[col], encoding)
                    self.encodings[col] = encoding
                    self.logger.info(f"Ordinal encoded: {col}")
        
        # Handle remaining categorical columns
        for col in categorical_columns:
            if col in encoded_df.columns and col not in (ordinal_columns or {}):
                if self.encoding_type == "onehot":
                    # One-hot encoding
                    onehot_df = self.apply_onehot_encoding(df[col])
                    encoded_df = pd.concat([encoded_df, onehot_df], axis=1)
                    encoded_df = encoded_df.drop(columns=[col])
                    self.encodings[col] = self.fit_onehot_encoding(df[col])
                    self.logger.info(f"One-hot encoded: {col}")
                else:
                    # Label encoding
                    encoding = self.fit_label_encoding(df[col])
                    encoded_df[col] = self.apply_label_encoding(df[col], encoding)
                    self.encodings[col] = encoding
                    self.logger.info(f"Label encoded: {col}")
        
        return encoded_df
    
    def process(self, metadata_path: Optional[str], output_path: str) -> Dict[str, Any]:
        """
        Process metadata file with label encoding.
        
        Args:
            metadata_path: Path to metadata CSV/JSON file
            output_path: Path for output
            
        Returns:
            Processing results
        """
        self.logger.info(f"Processing metadata from {metadata_path}")
        
        results = {
            "encoding_type": self.encoding_type,
            "encoded_columns": [],
            "status": "success"
        }
        
        try:
            if metadata_path is None:
                self.logger.warning("No metadata path provided")
                results["status"] = "skipped"
                return results
            
            metadata_path_obj = Path(metadata_path)
            
            if not metadata_path_obj.exists():
                self.logger.warning(f"Metadata file not found: {metadata_path}")
                results["status"] = "not_found"
                return results
            
            # Load metadata
            if metadata_path_obj.suffix == ".csv":
                df = pd.read_csv(metadata_path)
            elif metadata_path_obj.suffix == ".json":
                df = pd.read_json(metadata_path)
            else:
                raise ValueError(f"Unsupported metadata format: {metadata_path_obj.suffix}")
            
            results["original_shape"] = df.shape
            
            # Get encoding configuration
            categorical_cols = self.config.get("categorical_columns")
            ordinal_cols = self.config.get("ordinal_columns")
            
            # Encode metadata
            encoded_df = self.encode_metadata(df, categorical_cols, ordinal_cols)
            
            results["encoded_shape"] = encoded_df.shape
            results["encoded_columns"] = list(self.encodings.keys())
            
            # Save encoded metadata
            output_path_obj = Path(output_path)
            output_path_obj.mkdir(parents=True, exist_ok=True)
            
            encoded_metadata_path = output_path_obj / "encoded_metadata.csv"
            encoded_df.to_csv(encoded_metadata_path, index=False)
            
            # Save encoding mappings
            encoding_map_path = output_path_obj / "encoding_mappings.json"
            with open(encoding_map_path, "w") as f:
                # Convert numpy types to native Python types for JSON serialization
                json_encodings = {}
                for key, val in self.encodings.items():
                    if isinstance(val, dict):
                        json_encodings[key] = {str(k): int(v) if isinstance(v, (np.integer, np.int64)) else v 
                                               for k, v in val.items()}
                    elif isinstance(val, list):
                        json_encodings[key] = [str(v) for v in val]
                    else:
                        json_encodings[key] = val
                
                json.dump(json_encodings, f, indent=2)
            
            results["output_files"] = [
                str(encoded_metadata_path),
                str(encoding_map_path)
            ]
            
            self.logger.info(f"Encoded {len(results['encoded_columns'])} columns")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Label encoding failed: {str(e)}")
        
        return results
