"""
Main AI agent for orchestrating neuroimaging data wrangling tasks.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from ..processors.volume_normalizer import VolumeNormalizer
from ..processors.quality_controller import QualityController
from ..processors.outlier_detector import OutlierDetector
from ..processors.label_encoder import LabelEncoder
from ..utils.logger import setup_logger


class DataWranglingAgent:
    """
    AI-powered agent that orchestrates neuroimaging data wrangling tasks.
    
    This agent uses LLM capabilities to intelligently decide on preprocessing
    strategies and automate the data wrangling pipeline.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data wrangling agent.
        
        Args:
            config: Configuration dictionary for the agent
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or setup_logger("DataWranglingAgent")
        
        # Initialize processors
        self.volume_normalizer = VolumeNormalizer(self.config.get("normalization", {}))
        self.quality_controller = QualityController(self.config.get("quality_control", {}))
        self.outlier_detector = OutlierDetector(self.config.get("outlier_detection", {}))
        self.label_encoder = LabelEncoder(self.config.get("label_encoding", {}))
        
        self.logger.info("DataWranglingAgent initialized")
    
    def analyze_dataset(self, data_path: str) -> Dict[str, Any]:
        """
        Analyze the dataset structure and provide recommendations.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Analysis results and recommendations
        """
        self.logger.info(f"Analyzing dataset at: {data_path}")
        
        path = Path(data_path)
        analysis = {
            "path": str(path),
            "exists": path.exists(),
            "file_count": 0,
            "file_types": {},
            "recommended_pipeline": []
        }
        
        if path.exists():
            if path.is_dir():
                files = list(path.rglob("*"))
                analysis["file_count"] = len([f for f in files if f.is_file()])
                
                # Count file types
                for file in files:
                    if file.is_file():
                        ext = file.suffix.lower()
                        analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1
            else:
                analysis["file_count"] = 1
                analysis["file_types"][path.suffix.lower()] = 1
        
        # AI-assisted recommendation (simplified)
        if ".nii" in analysis["file_types"] or ".gz" in analysis["file_types"]:
            analysis["recommended_pipeline"] = [
                "volume_normalization",
                "quality_control",
                "outlier_detection",
                "label_encoding"
            ]
        
        self.logger.info(f"Analysis complete: {analysis['file_count']} files found")
        return analysis
    
    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        steps: Optional[List[str]] = None,
        metadata_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete data wrangling pipeline.
        
        Args:
            input_path: Path to input data
            output_path: Path for output data
            steps: List of processing steps to execute (if None, uses all)
            metadata_path: Optional path to metadata CSV/JSON file
            
        Returns:
            Pipeline execution results
        """
        self.logger.info("Starting data wrangling pipeline")
        
        if steps is None:
            steps = [
                "volume_normalization",
                "quality_control",
                "outlier_detection",
                "label_encoding"
            ]
        
        results = {
            "input_path": input_path,
            "output_path": output_path,
            "steps_executed": [],
            "status": "success",
            "details": {}
        }
        
        try:
            # Create output directory
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Execute each step
            for step in steps:
                self.logger.info(f"Executing step: {step}")
                
                if step == "volume_normalization":
                    step_results = self.volume_normalizer.process(input_path, output_path)
                    results["details"]["volume_normalization"] = step_results
                    
                elif step == "quality_control":
                    step_results = self.quality_controller.process(input_path, metadata_path)
                    results["details"]["quality_control"] = step_results
                    
                elif step == "outlier_detection":
                    step_results = self.outlier_detector.process(input_path, metadata_path)
                    results["details"]["outlier_detection"] = step_results
                    
                elif step == "label_encoding":
                    step_results = self.label_encoder.process(metadata_path, output_path)
                    results["details"]["label_encoding"] = step_results
                
                results["steps_executed"].append(step)
                self.logger.info(f"Step {step} completed successfully")
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        self.logger.info(f"Pipeline completed with status: {results['status']}")
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Generate a human-readable report of the wrangling results.
        
        Args:
            results: Pipeline results dictionary
            output_path: Path to save the report
        """
        self.logger.info(f"Generating report at: {output_path}")
        
        report_lines = [
            "# Data Wrangling Report",
            "",
            f"**Status**: {results['status']}",
            f"**Input Path**: {results['input_path']}",
            f"**Output Path**: {results['output_path']}",
            "",
            "## Steps Executed",
            ""
        ]
        
        for step in results["steps_executed"]:
            report_lines.append(f"- {step}")
        
        report_lines.extend(["", "## Details", ""])
        
        for step, details in results.get("details", {}).items():
            report_lines.append(f"### {step}")
            report_lines.append(f"```")
            report_lines.append(str(details))
            report_lines.append(f"```")
            report_lines.append("")
        
        if "error" in results:
            report_lines.extend([
                "## Error",
                "",
                f"```",
                results["error"],
                f"```"
            ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, "w") as f:
            f.write(report_content)
        
        self.logger.info("Report generated successfully")
