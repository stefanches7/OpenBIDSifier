"""
AI-Assisted Neuroimaging Data Wrangler

A framework for automating neuroimaging data preprocessing tasks using AI agents.
"""

__version__ = "0.1.0"

from .agents.wrangling_agent import DataWranglingAgent
from .processors.volume_normalizer import VolumeNormalizer
from .processors.quality_controller import QualityController
from .processors.outlier_detector import OutlierDetector
from .processors.label_encoder import LabelEncoder

__all__ = [
    "DataWranglingAgent",
    "VolumeNormalizer",
    "QualityController",
    "OutlierDetector",
    "LabelEncoder",
]
