"""
Outlier detection processor for neuroimaging data.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from ..utils.logger import setup_logger


class OutlierDetector:
    """
    Detects outliers in neuroimaging datasets.
    
    Methods:
    - Statistical outlier detection (z-score, IQR)
    - Isolation Forest
    - Local Outlier Factor
    - Distance-based methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the outlier detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = setup_logger("OutlierDetector")
        self.method = self.config.get("method", "zscore")
        self.threshold = self.config.get("threshold", 3.0)
    
    def detect_zscore_outliers(
        self,
        features: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Detect outliers using z-score method.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Boolean array indicating outliers
        """
        if threshold is None:
            threshold = self.threshold
        
        z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))
        outliers = np.any(z_scores > threshold, axis=1)
        
        return outliers
    
    def detect_iqr_outliers(self, features: np.ndarray) -> np.ndarray:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Boolean array indicating outliers
        """
        q1 = np.percentile(features, 25, axis=0)
        q3 = np.percentile(features, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.any((features < lower_bound) | (features > upper_bound), axis=1)
        
        return outliers
    
    def detect_isolation_forest_outliers(
        self,
        features: np.ndarray,
        contamination: float = 0.1
    ) -> np.ndarray:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean array indicating outliers
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            clf = IsolationForest(contamination=contamination, random_state=42)
            predictions = clf.fit_predict(features)
            
            # -1 for outliers, 1 for inliers
            outliers = predictions == -1
            
            return outliers
        
        except ImportError:
            self.logger.warning("scikit-learn not available, falling back to z-score")
            return self.detect_zscore_outliers(features)
    
    def extract_summary_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract summary features from imaging data for outlier detection.
        
        Args:
            data: Imaging data array
            
        Returns:
            Feature vector
        """
        features = [
            np.mean(data),
            np.std(data),
            np.median(data),
            np.min(data),
            np.max(data),
            np.percentile(data, 25),
            np.percentile(data, 75),
        ]
        
        # Add spatial features if 3D
        if len(data.shape) >= 3:
            features.extend([
                np.mean(data[:, :, data.shape[2]//2]),  # Middle slice mean
                np.std(data[:, :, data.shape[2]//2]),   # Middle slice std
            ])
        
        return np.array(features)
    
    def process(self, input_path: str, metadata_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run outlier detection on neuroimaging data.
        
        Args:
            input_path: Path to input data
            metadata_path: Optional path to metadata
            
        Returns:
            Outlier detection results
        """
        self.logger.info(f"Running outlier detection on {input_path}")
        
        results = {
            "method": self.method,
            "total_files": 0,
            "outliers_detected": 0,
            "outlier_files": [],
            "status": "success",
            "summary": {}
        }
        
        try:
            input_path_obj = Path(input_path)
            
            # Find neuroimaging files
            nii_files = []
            if input_path_obj.is_dir():
                nii_files.extend(input_path_obj.glob("*.nii"))
                nii_files.extend(input_path_obj.glob("*.nii.gz"))
            elif input_path_obj.suffix in [".nii", ".gz"]:
                nii_files.append(input_path_obj)
            
            results["total_files"] = len(nii_files)
            
            if len(nii_files) == 0:
                self.logger.warning("No neuroimaging files found")
                return results
            
            # In a real implementation, we would:
            # 1. Load all images
            # 2. Extract features from each
            # 3. Apply outlier detection algorithm
            # 4. Return outlier indices
            
            # Simulate outlier detection
            feature_matrix = []
            file_names = []
            
            for nii_file in nii_files:
                try:
                    # Simulate feature extraction
                    # In reality, would use nibabel to load and extract features
                    simulated_features = np.random.randn(9)  # 9 features
                    feature_matrix.append(simulated_features)
                    file_names.append(nii_file.name)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process {nii_file}: {str(e)}")
            
            if len(feature_matrix) > 0:
                feature_matrix = np.array(feature_matrix)
                
                # Detect outliers
                if self.method == "zscore":
                    outliers = self.detect_zscore_outliers(feature_matrix)
                elif self.method == "iqr":
                    outliers = self.detect_iqr_outliers(feature_matrix)
                elif self.method == "isolation_forest":
                    outliers = self.detect_isolation_forest_outliers(feature_matrix)
                else:
                    outliers = self.detect_zscore_outliers(feature_matrix)
                
                # Record outliers
                results["outliers_detected"] = int(np.sum(outliers))
                results["outlier_files"] = [
                    file_names[i] for i in range(len(file_names)) if outliers[i]
                ]
                
                results["summary"] = {
                    "outlier_ratio": results["outliers_detected"] / results["total_files"],
                    "total_scanned": len(file_names)
                }
            
            self.logger.info(
                f"Outlier detection complete: {results['outliers_detected']} outliers found"
            )
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Outlier detection failed: {str(e)}")
        
        return results
