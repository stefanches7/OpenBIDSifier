"""
Tests for the DataWranglingAgent.
"""

import pytest
import tempfile
import os
from pathlib import Path

from ai_neuro_wrangler.agents.wrangling_agent import DataWranglingAgent


def test_agent_initialization():
    """Test agent initialization with default config."""
    agent = DataWranglingAgent()
    assert agent is not None
    assert agent.volume_normalizer is not None
    assert agent.quality_controller is not None
    assert agent.outlier_detector is not None
    assert agent.label_encoder is not None


def test_agent_with_custom_config():
    """Test agent initialization with custom config."""
    config = {
        "normalization": {"method": "minmax"},
        "outlier_detection": {"threshold": 2.5}
    }
    agent = DataWranglingAgent(config)
    assert agent.config["normalization"]["method"] == "minmax"


def test_analyze_dataset_nonexistent():
    """Test dataset analysis with non-existent path."""
    agent = DataWranglingAgent()
    results = agent.analyze_dataset("/nonexistent/path")
    
    assert results["exists"] is False
    assert results["file_count"] == 0


def test_analyze_dataset_directory():
    """Test dataset analysis with a temporary directory."""
    agent = DataWranglingAgent()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        (Path(tmpdir) / "test1.nii").touch()
        (Path(tmpdir) / "test2.nii.gz").touch()
        (Path(tmpdir) / "other.txt").touch()
        
        results = agent.analyze_dataset(tmpdir)
        
        assert results["exists"] is True
        assert results["file_count"] >= 3
        assert ".nii" in results["file_types"] or ".gz" in results["file_types"]


def test_run_pipeline_basic():
    """Test running basic pipeline."""
    agent = DataWranglingAgent()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        
        # Create a dummy file
        (input_dir / "test.nii").touch()
        
        results = agent.run_pipeline(
            input_path=str(input_dir),
            output_path=str(output_dir)
        )
        
        assert "status" in results
        assert "steps_executed" in results
        assert output_dir.exists()


def test_generate_report():
    """Test report generation."""
    agent = DataWranglingAgent()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.md"
        
        results = {
            "status": "success",
            "input_path": "/test/input",
            "output_path": "/test/output",
            "steps_executed": ["volume_normalization", "quality_control"],
            "details": {}
        }
        
        agent.generate_report(results, str(report_path))
        
        assert report_path.exists()
        content = report_path.read_text()
        assert "Data Wrangling Report" in content
        assert "volume_normalization" in content
