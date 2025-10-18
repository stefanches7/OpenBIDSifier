"""
Tests for data processors.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import pandas as pd

from ai_neuro_wrangler.processors.volume_normalizer import VolumeNormalizer
from ai_neuro_wrangler.processors.quality_controller import QualityController
from ai_neuro_wrangler.processors.outlier_detector import OutlierDetector
from ai_neuro_wrangler.processors.label_encoder import LabelEncoder


class TestVolumeNormalizer:
    """Tests for VolumeNormalizer."""
    
    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = VolumeNormalizer()
        assert normalizer.method == "zscore"
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        normalizer = VolumeNormalizer()
        data = np.random.randn(10, 10, 10)
        normalized = normalizer.normalize_zscore(data)
        
        assert normalized.shape == data.shape
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        normalizer = VolumeNormalizer()
        data = np.random.randn(10, 10, 10)
        normalized = normalizer.normalize_minmax(data)
        
        assert normalized.shape == data.shape
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1
    
    def test_process(self):
        """Test process method."""
        normalizer = VolumeNormalizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            
            results = normalizer.process(str(input_dir), str(output_dir))
            
            assert "method" in results
            assert "status" in results


class TestQualityController:
    """Tests for QualityController."""
    
    def test_initialization(self):
        """Test QC initialization."""
        qc = QualityController()
        assert qc.min_snr == 10.0
    
    def test_check_dimensions(self):
        """Test dimension checking."""
        qc = QualityController()
        
        assert qc.check_dimensions((64, 64, 32))
        assert not qc.check_dimensions((64, 64))
    
    def test_calculate_snr(self):
        """Test SNR calculation."""
        qc = QualityController()
        data = np.random.randn(10, 10, 10) + 100
        snr = qc.calculate_snr(data)
        
        assert isinstance(snr, float)
        assert snr > 0
    
    def test_detect_artifacts(self):
        """Test artifact detection."""
        qc = QualityController()
        data = np.random.randn(10, 10, 10)
        artifacts = qc.detect_artifacts(data)
        
        assert "ghosting" in artifacts
        assert "ringing" in artifacts
        assert "blank_slices" in artifacts


class TestOutlierDetector:
    """Tests for OutlierDetector."""
    
    def test_initialization(self):
        """Test outlier detector initialization."""
        detector = OutlierDetector()
        assert detector.method == "zscore"
        assert detector.threshold == 3.0
    
    def test_zscore_outlier_detection(self):
        """Test z-score based outlier detection."""
        detector = OutlierDetector()
        
        # Create data with clear outliers
        features = np.random.randn(100, 5)
        features[0] = 10  # Outlier
        features[1] = -10  # Outlier
        
        outliers = detector.detect_zscore_outliers(features)
        
        assert len(outliers) == 100
        assert outliers[0] or outliers[1]
    
    def test_iqr_outlier_detection(self):
        """Test IQR based outlier detection."""
        detector = OutlierDetector()
        
        features = np.random.randn(100, 5)
        features[0] = 10
        
        outliers = detector.detect_iqr_outliers(features)
        
        assert len(outliers) == 100
    
    def test_extract_summary_features(self):
        """Test feature extraction."""
        detector = OutlierDetector()
        data = np.random.randn(10, 10, 10)
        
        features = detector.extract_summary_features(data)
        
        assert len(features) > 0
        assert isinstance(features, np.ndarray)


class TestLabelEncoder:
    """Tests for LabelEncoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = LabelEncoder()
        assert encoder.encoding_type == "label"
    
    def test_fit_label_encoding(self):
        """Test fitting label encoding."""
        encoder = LabelEncoder()
        series = pd.Series(['cat', 'dog', 'cat', 'bird', 'dog'])
        
        encoding = encoder.fit_label_encoding(series)
        
        assert len(encoding) == 3
        assert 'cat' in encoding
        assert 'dog' in encoding
        assert 'bird' in encoding
    
    def test_apply_label_encoding(self):
        """Test applying label encoding."""
        encoder = LabelEncoder()
        series = pd.Series(['cat', 'dog', 'cat', 'bird'])
        
        encoded = encoder.apply_label_encoding(series)
        
        assert len(encoded) == len(series)
        assert encoded.dtype in [np.int64, np.float64]
    
    def test_onehot_encoding(self):
        """Test one-hot encoding."""
        encoder = LabelEncoder()
        series = pd.Series(['cat', 'dog', 'cat', 'bird'])
        
        onehot = encoder.apply_onehot_encoding(series)
        
        assert isinstance(onehot, pd.DataFrame)
        assert onehot.shape[0] == len(series)
    
    def test_encode_metadata(self):
        """Test metadata encoding."""
        encoder = LabelEncoder()
        
        df = pd.DataFrame({
            'subject': ['S1', 'S2', 'S3'],
            'category': ['A', 'B', 'A'],
            'age': [25, 30, 35]
        })
        
        encoded_df = encoder.encode_metadata(df, categorical_columns=['category'])
        
        assert 'age' in encoded_df.columns
        assert isinstance(encoded_df, pd.DataFrame)
    
    def test_process_with_metadata(self):
        """Test process with metadata file."""
        encoder = LabelEncoder()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample metadata
            metadata_path = Path(tmpdir) / "metadata.csv"
            df = pd.DataFrame({
                'subject': ['S1', 'S2', 'S3'],
                'category': ['A', 'B', 'A'],
            })
            df.to_csv(metadata_path, index=False)
            
            output_dir = Path(tmpdir) / "output"
            
            results = encoder.process(str(metadata_path), str(output_dir))
            
            assert "status" in results
            assert results["status"] in ["success", "failed"]
