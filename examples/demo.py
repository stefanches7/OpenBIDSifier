#!/usr/bin/env python3
"""
Demo script showing AI Neuro Wrangler capabilities.

This script demonstrates the main features of the framework
without requiring actual neuroimaging data.
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_neuro_wrangler import DataWranglingAgent
from ai_neuro_wrangler.utils.config_loader import get_default_config


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def demo_basic_usage():
    """Demonstrate basic usage."""
    print_section("Basic Usage Demo")
    
    print("\n1. Creating a DataWranglingAgent...")
    agent = DataWranglingAgent()
    print("   ✓ Agent created successfully")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock dataset
        data_dir = Path(tmpdir) / "dataset"
        data_dir.mkdir()
        
        # Create some mock files
        (data_dir / "scan_001.nii").touch()
        (data_dir / "scan_002.nii").touch()
        (data_dir / "scan_003.nii.gz").touch()
        
        print(f"\n2. Analyzing mock dataset...")
        results = agent.analyze_dataset(str(data_dir))
        
        print(f"   - Path: {results['path']}")
        print(f"   - Exists: {results['exists']}")
        print(f"   - File count: {results['file_count']}")
        print(f"   - File types: {results['file_types']}")
        print(f"   - Recommended pipeline: {results['recommended_pipeline']}")


def demo_custom_config():
    """Demonstrate custom configuration."""
    print_section("Custom Configuration Demo")
    
    print("\n1. Getting default configuration...")
    config = get_default_config()
    print(f"   - Default normalization method: {config['normalization']['method']}")
    print(f"   - Default outlier threshold: {config['outlier_detection']['threshold']}")
    
    print("\n2. Creating custom configuration...")
    custom_config = {
        "normalization": {
            "method": "percentile",
            "clip_percentiles": [2, 98]
        },
        "outlier_detection": {
            "method": "iqr",
            "threshold": 2.5
        },
        "label_encoding": {
            "encoding_type": "onehot"
        }
    }
    
    print("   ✓ Custom configuration created")
    print(f"   - Normalization method: {custom_config['normalization']['method']}")
    print(f"   - Outlier method: {custom_config['outlier_detection']['method']}")
    
    print("\n3. Creating agent with custom config...")
    agent = DataWranglingAgent(custom_config)
    print("   ✓ Agent created with custom configuration")


def demo_metadata_encoding():
    """Demonstrate metadata encoding."""
    print_section("Metadata Encoding Demo")
    
    print("\n1. Creating sample metadata...")
    metadata = pd.DataFrame({
        'subject_id': ['sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005'],
        'age': [45, 52, 38, 61, 43],
        'sex': ['M', 'F', 'M', 'F', 'M'],
        'diagnosis': ['control', 'patient', 'control', 'patient', 'control'],
        'site': ['site1', 'site1', 'site2', 'site2', 'site1']
    })
    
    print("\n   Original metadata:")
    print(metadata.to_string(index=False))
    
    print("\n2. Encoding categorical variables...")
    from ai_neuro_wrangler.processors.label_encoder import LabelEncoder
    
    encoder = LabelEncoder({"encoding_type": "label"})
    encoded_df = encoder.encode_metadata(
        metadata,
        categorical_columns=['sex', 'diagnosis', 'site']
    )
    
    print("\n   Encoded metadata:")
    print(encoded_df.to_string(index=False))
    
    print("\n   Encoding mappings:")
    for col, mapping in encoder.encodings.items():
        print(f"   - {col}: {mapping}")


def demo_pipeline():
    """Demonstrate full pipeline."""
    print_section("Full Pipeline Demo")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup directories
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        
        # Create mock data
        (input_dir / "scan_001.nii").touch()
        (input_dir / "scan_002.nii").touch()
        
        # Create mock metadata
        metadata_path = Path(tmpdir) / "metadata.csv"
        metadata = pd.DataFrame({
            'subject_id': ['sub-001', 'sub-002'],
            'diagnosis': ['control', 'patient'],
        })
        metadata.to_csv(metadata_path, index=False)
        
        print("\n1. Initializing pipeline...")
        agent = DataWranglingAgent()
        
        print("\n2. Running data wrangling pipeline...")
        print(f"   - Input: {input_dir}")
        print(f"   - Output: {output_dir}")
        print(f"   - Metadata: {metadata_path}")
        
        results = agent.run_pipeline(
            input_path=str(input_dir),
            output_path=str(output_dir),
            metadata_path=str(metadata_path)
        )
        
        print(f"\n3. Pipeline Results:")
        print(f"   - Status: {results['status']}")
        print(f"   - Steps executed: {len(results['steps_executed'])}")
        for step in results['steps_executed']:
            print(f"     ✓ {step}")
        
        print("\n4. Generating report...")
        report_path = Path(tmpdir) / "report.md"
        agent.generate_report(results, str(report_path))
        print(f"   ✓ Report saved to: {report_path}")
        
        # Show a snippet of the report
        with open(report_path) as f:
            lines = f.readlines()[:10]
            print("\n   Report preview:")
            for line in lines:
                print(f"   {line.rstrip()}")


def demo_specific_processors():
    """Demonstrate individual processors."""
    print_section("Individual Processors Demo")
    
    from ai_neuro_wrangler.processors.volume_normalizer import VolumeNormalizer
    from ai_neuro_wrangler.processors.quality_controller import QualityController
    from ai_neuro_wrangler.processors.outlier_detector import OutlierDetector
    
    import numpy as np
    
    print("\n1. Volume Normalizer")
    normalizer = VolumeNormalizer({"method": "zscore"})
    
    # Create synthetic data
    data = np.random.randn(10, 10, 10) * 10 + 50
    print(f"   - Original: mean={np.mean(data):.2f}, std={np.std(data):.2f}")
    
    normalized = normalizer.normalize_zscore(data)
    print(f"   - Normalized: mean={np.mean(normalized):.2f}, std={np.std(normalized):.2f}")
    
    print("\n2. Quality Controller")
    qc = QualityController()
    
    snr = qc.calculate_snr(data)
    print(f"   - SNR: {snr:.2f}")
    
    artifacts = qc.detect_artifacts(data)
    print(f"   - Artifacts detected: {artifacts}")
    
    print("\n3. Outlier Detector")
    detector = OutlierDetector({"method": "zscore", "threshold": 3.0})
    
    # Create data with outliers
    features = np.random.randn(100, 5)
    features[0] = 10  # Clear outlier
    
    outliers = detector.detect_zscore_outliers(features)
    print(f"   - Number of outliers: {np.sum(outliers)} / {len(outliers)}")
    print(f"   - Outlier ratio: {np.sum(outliers) / len(outliers):.2%}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  AI Neuro Wrangler - Interactive Demo")
    print("=" * 60)
    print("\nThis demo showcases the capabilities of the AI-assisted")
    print("neuroimaging data wrangling framework.")
    
    try:
        demo_basic_usage()
        demo_custom_config()
        demo_metadata_encoding()
        demo_specific_processors()
        demo_pipeline()
        
        print_section("Demo Complete")
        print("\n✓ All demos completed successfully!")
        print("\nNext steps:")
        print("  - Try the CLI: ai-neuro-wrangler --help")
        print("  - Read the documentation in README.md")
        print("  - Check examples/example_usage.py for more examples")
        print("")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
