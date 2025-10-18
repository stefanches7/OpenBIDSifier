# AI-Assisted Neuroimaging Data Harmonization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automate neuroimaging data preprocessing and harmonization using AI-powered agents. This framework dramatically reduces the time and manual labor required for standard data wrangling procedures, making it easier to create AI-ready dataset collections.

## 🎯 Problem Statement

Standard data wrangling procedures in neuroimaging research (volume normalization, quality control, outlier detection, label encoding) traditionally consume a significant amount of time and manual effort. This project explores how AI agents, particularly LLM-based systems, can meaningfully assist in these tasks to:

- Reduce manual labor in data preprocessing
- Standardize preprocessing pipelines across datasets
- Enable rapid creation of AI-ready dataset collections
- Improve reproducibility and documentation

## 🚀 Features

### Core Capabilities

- **Volume Normalization**: Multiple normalization strategies (z-score, min-max, percentile-based)
- **Quality Control**: Automated QC checks including SNR estimation, artifact detection, motion assessment
- **Outlier Detection**: Statistical and ML-based outlier detection (z-score, IQR, Isolation Forest)
- **Label Encoding**: Flexible encoding schemes for categorical and ordinal metadata

### AI-Powered Orchestration

- Intelligent pipeline recommendation based on dataset analysis
- Automated parameter selection
- Comprehensive reporting and documentation
- Easy-to-use CLI and Python API

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/stefanches7/AI-assisted-Neuroimaging-harmonization.git
cd AI-assisted-Neuroimaging-harmonization

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🔧 Quick Start

### Command Line Interface

```bash
# Analyze a dataset and get recommendations
ai-neuro-wrangler analyze /path/to/dataset

# Run the full preprocessing pipeline
ai-neuro-wrangler wrangle /path/to/input /path/to/output \
    --metadata /path/to/metadata.csv \
    --report /path/to/report.md

# Generate a configuration file
ai-neuro-wrangler init-config config.yaml

# Run specific processing steps
ai-neuro-wrangler wrangle /path/to/input /path/to/output \
    -s volume_normalization \
    -s outlier_detection
```

### Python API

```python
from ai_neuro_wrangler import DataWranglingAgent

# Create an agent with default configuration
agent = DataWranglingAgent()

# Analyze your dataset
analysis = agent.analyze_dataset("/path/to/dataset")
print(f"Found {analysis['file_count']} files")
print(f"Recommended steps: {analysis['recommended_pipeline']}")

# Run the complete pipeline
results = agent.run_pipeline(
    input_path="/path/to/input",
    output_path="/path/to/output",
    metadata_path="/path/to/metadata.csv"
)

# Generate a comprehensive report
agent.generate_report(results, "/path/to/report.md")
```

## 📚 Documentation

### Project Structure

```
AI-assisted-Neuroimaging-harmonization/
├── src/ai_neuro_wrangler/
│   ├── agents/              # AI orchestration agents
│   │   └── wrangling_agent.py
│   ├── processors/          # Data processing modules
│   │   ├── volume_normalizer.py
│   │   ├── quality_controller.py
│   │   ├── outlier_detector.py
│   │   └── label_encoder.py
│   ├── utils/              # Utility functions
│   │   ├── config_loader.py
│   │   ├── data_loader.py
│   │   └── logger.py
│   └── cli.py              # Command-line interface
├── config/                 # Configuration files
├── examples/               # Usage examples
├── tests/                  # Test suite
└── data/                   # Sample data (not tracked)
```

### Configuration

The framework uses YAML configuration files for customization:

```yaml
# Volume normalization
normalization:
  method: zscore  # zscore, minmax, or percentile
  clip_percentiles: [1, 99]

# Quality control
quality_control:
  min_snr: 10.0
  max_outlier_ratio: 0.05

# Outlier detection
outlier_detection:
  method: zscore  # zscore, iqr, or isolation_forest
  threshold: 3.0

# Label encoding
label_encoding:
  encoding_type: label  # label or onehot
  categorical_columns: null  # Auto-detect if null
```

### Processing Steps

#### 1. Volume Normalization

Normalizes imaging volumes using various strategies:

- **Z-score**: Standardizes data to zero mean and unit variance
- **Min-Max**: Scales data to [0, 1] range
- **Percentile**: Clips and normalizes based on percentile thresholds

#### 2. Quality Control

Performs comprehensive quality checks:

- Image dimension validation
- Signal-to-noise ratio (SNR) estimation
- Artifact detection (ghosting, ringing, intensity spikes)
- Motion assessment for time-series data

#### 3. Outlier Detection

Identifies problematic scans using:

- Statistical methods (z-score, IQR)
- Machine learning (Isolation Forest, LOF)
- Feature-based analysis

#### 4. Label Encoding

Handles categorical and ordinal metadata:

- Label encoding for categorical variables
- One-hot encoding for nominal categories
- Ordinal encoding with custom ordering

## 🎓 Use Cases

### 1. Creating AI-Ready Datasets

Rapidly prepare neuroimaging datasets for machine learning:

```python
from ai_neuro_wrangler import DataWranglingAgent

agent = DataWranglingAgent()
results = agent.run_pipeline(
    input_path="raw_data/",
    output_path="processed_data/",
    metadata_path="metadata.csv"
)
```

### 2. Quality Control Pipeline

Run automated QC on large datasets:

```bash
ai-neuro-wrangler wrangle raw_data/ qc_results/ \
    -s quality_control \
    -s outlier_detection \
    --report qc_report.md
```

### 3. Harmonization Across Studies

Standardize preprocessing across multiple studies:

```python
config = {
    "normalization": {"method": "zscore"},
    "outlier_detection": {"threshold": 2.5}
}

agent = DataWranglingAgent(config)

for study in ["study1", "study2", "study3"]:
    agent.run_pipeline(f"{study}/raw", f"{study}/processed")
```

## 🔬 Example: Creating an OpenMind-style Dataset

This framework facilitates creating comprehensive, AI-ready datasets similar to [OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind):

```python
from ai_neuro_wrangler import DataWranglingAgent
from pathlib import Path

# Initialize agent
agent = DataWranglingAgent()

# Process multiple datasets
datasets = ["ADNI", "OASIS", "ABIDE"]

for dataset in datasets:
    print(f"Processing {dataset}...")
    
    # Analyze dataset structure
    analysis = agent.analyze_dataset(f"raw/{dataset}")
    
    # Run standardized pipeline
    results = agent.run_pipeline(
        input_path=f"raw/{dataset}",
        output_path=f"processed/{dataset}",
        metadata_path=f"metadata/{dataset}.csv"
    )
    
    # Generate report
    agent.generate_report(results, f"reports/{dataset}_report.md")

print("All datasets processed and ready for upload!")
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=ai_neuro_wrangler tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the need to streamline neuroimaging data preprocessing
- Built to support the creation of comprehensive datasets like [OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind)
- Thanks to the neuroimaging community for standardization efforts

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

## 🗺️ Roadmap

- [ ] Integration with LLM agents for intelligent parameter selection
- [ ] Support for more neuroimaging formats (DICOM, MGH, etc.)
- [ ] Advanced harmonization techniques (ComBat, etc.)
- [ ] Web-based interface for visual QC
- [ ] Integration with HuggingFace datasets
- [ ] Distributed processing for large-scale datasets
- [ ] Pre-trained models for quality assessment

## 📊 Citing

If you use this framework in your research, please cite:

```bibtex
@software{ai_neuro_wrangler2024,
  title={AI-Assisted Neuroimaging Data Harmonization},
  author={AI-assisted Neuroimaging Team},
  year={2024},
  url={https://github.com/stefanches7/AI-assisted-Neuroimaging-harmonization}
}
```
