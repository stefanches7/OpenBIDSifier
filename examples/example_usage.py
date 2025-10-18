"""
Example usage of the AI Neuro Wrangler.
"""

from pathlib import Path
from ai_neuro_wrangler import DataWranglingAgent
from ai_neuro_wrangler.utils.config_loader import get_default_config


def example_basic_usage():
    """Basic usage example."""
    
    # Create agent with default config
    agent = DataWranglingAgent()
    
    # Analyze a dataset
    results = agent.analyze_dataset("/path/to/dataset")
    print(f"Found {results['file_count']} files")
    print(f"Recommended steps: {results['recommended_pipeline']}")
    
    # Run the full pipeline
    pipeline_results = agent.run_pipeline(
        input_path="/path/to/input",
        output_path="/path/to/output",
        metadata_path="/path/to/metadata.csv"
    )
    
    print(f"Pipeline status: {pipeline_results['status']}")


def example_custom_config():
    """Example with custom configuration."""
    
    config = {
        "normalization": {
            "method": "percentile",
            "clip_percentiles": [2, 98]
        },
        "outlier_detection": {
            "method": "isolation_forest",
            "contamination": 0.05
        },
        "label_encoding": {
            "encoding_type": "onehot"
        }
    }
    
    agent = DataWranglingAgent(config)
    
    # Run specific steps only
    results = agent.run_pipeline(
        input_path="/path/to/input",
        output_path="/path/to/output",
        steps=["volume_normalization", "outlier_detection"]
    )


def example_with_report():
    """Example with report generation."""
    
    agent = DataWranglingAgent()
    
    # Run pipeline
    results = agent.run_pipeline(
        input_path="/path/to/input",
        output_path="/path/to/output",
        metadata_path="/path/to/metadata.csv"
    )
    
    # Generate detailed report
    agent.generate_report(results, "/path/to/report.md")


def example_step_by_step():
    """Example using individual processors."""
    
    from ai_neuro_wrangler import (
        VolumeNormalizer,
        QualityController,
        OutlierDetector,
        LabelEncoder
    )
    
    # Volume normalization
    normalizer = VolumeNormalizer({"method": "zscore"})
    norm_results = normalizer.process(
        input_path="/path/to/input",
        output_path="/path/to/normalized"
    )
    
    # Quality control
    qc = QualityController({"min_snr": 15.0})
    qc_results = qc.process(
        input_path="/path/to/normalized"
    )
    
    # Outlier detection
    outlier_detector = OutlierDetector({"method": "iqr"})
    outlier_results = outlier_detector.process(
        input_path="/path/to/normalized"
    )
    
    # Label encoding
    encoder = LabelEncoder({"encoding_type": "label"})
    encoding_results = encoder.process(
        metadata_path="/path/to/metadata.csv",
        output_path="/path/to/output"
    )


if __name__ == "__main__":
    print("AI Neuro Wrangler Examples")
    print("=" * 50)
    
    print("\n1. Basic Usage:")
    print("   agent = DataWranglingAgent()")
    print("   results = agent.analyze_dataset('/path/to/data')")
    
    print("\n2. Run Pipeline:")
    print("   results = agent.run_pipeline(")
    print("       input_path='/path/to/input',")
    print("       output_path='/path/to/output'")
    print("   )")
    
    print("\n3. CLI Usage:")
    print("   ai-neuro-wrangler analyze /path/to/dataset")
    print("   ai-neuro-wrangler wrangle /path/to/input /path/to/output")
    
    print("\nSee documentation for more examples!")
