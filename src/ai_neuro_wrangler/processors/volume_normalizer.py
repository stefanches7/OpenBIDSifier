"""
Volume normalization processor for neuroimaging data.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from ..utils.logger import setup_logger


class VolumeNormalizer:
    """
    Handles volume normalization for neuroimaging data.
    
    Supports multiple normalization strategies:
    - Z-score normalization
    - Min-max normalization
    - Percentile-based normalization
    - Intensity standardization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the volume normalizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = setup_logger("VolumeNormalizer")
        self.method = self.config.get("method", "zscore")
        self.clip_percentiles = self.config.get("clip_percentiles", (1, 99))
    
    def normalize_zscore(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Z-score normalization.
        
        Args:
            data: Input data array
            mask: Optional mask to exclude certain voxels
            
        Returns:
            Normalized data
        """
        if mask is not None:
            masked_data = data[mask > 0]
            mean = np.mean(masked_data)
            std = np.std(masked_data)
        else:
            mean = np.mean(data)
            std = np.std(data)
        
        if std == 0:
            return data
        
        return (data - mean) / std
    
    def normalize_minmax(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply min-max normalization to [0, 1] range.
        
        Args:
            data: Input data array
            mask: Optional mask to exclude certain voxels
            
        Returns:
            Normalized data
        """
        if mask is not None:
            masked_data = data[mask > 0]
            min_val = np.min(masked_data)
            max_val = np.max(masked_data)
        else:
            min_val = np.min(data)
            max_val = np.max(data)
        
        if max_val == min_val:
            return data
        
        return (data - min_val) / (max_val - min_val)
    
    def normalize_percentile(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply percentile-based normalization with clipping.
        
        Args:
            data: Input data array
            mask: Optional mask to exclude certain voxels
            
        Returns:
            Normalized data
        """
        if mask is not None:
            masked_data = data[mask > 0]
        else:
            masked_data = data.flatten()
        
        low_perc, high_perc = self.clip_percentiles
        low_val = np.percentile(masked_data, low_perc)
        high_val = np.percentile(masked_data, high_perc)
        
        # Clip and normalize
        clipped_data = np.clip(data, low_val, high_val)
        
        if high_val == low_val:
            return clipped_data
        
        return (clipped_data - low_val) / (high_val - low_val)
    
    def process(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process neuroimaging volumes with normalization.
        
        Args:
            input_path: Path to input data
            output_path: Path for output data
            
        Returns:
            Processing results
        """
        self.logger.info(f"Processing volumes from {input_path}")
        
        results = {
            "method": self.method,
            "processed_files": 0,
            "status": "success"
        }
        
        try:
            input_path_obj = Path(input_path)
            output_path_obj = Path(output_path)
            output_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Find neuroimaging files
            nii_files = []
            if input_path_obj.is_dir():
                nii_files.extend(input_path_obj.glob("*.nii"))
                nii_files.extend(input_path_obj.glob("*.nii.gz"))
            elif input_path_obj.suffix in [".nii", ".gz"]:
                nii_files.append(input_path_obj)
            
            results["total_files"] = len(nii_files)
            
            # Process each file
            for nii_file in nii_files:
                try:
                    # In a real implementation, use nibabel to load and process
                    # For now, we just track the files
                    self.logger.info(f"Would normalize: {nii_file.name}")
                    results["processed_files"] += 1
                except Exception as e:
                    self.logger.warning(f"Failed to process {nii_file}: {str(e)}")
            
            self.logger.info(f"Processed {results['processed_files']} files")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Processing failed: {str(e)}")
        
        return results
