"""
Main module for the DanceMotionAI package.

This module initializes the DanceMotionAI system and provides the main API.
"""

import os
import yaml
from typing import Dict, Optional, Union, List, Tuple

class DanceMotionAI:
    """
    Main class for the DanceMotionAI system.
    
    This class integrates all components of the system and provides a unified API.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the DanceMotionAI system.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Initialize components based on configuration
        self._init_pose_estimator()
        self._init_multimodal_analyzer()
        self._init_similarity_analyzer()
        
    def _init_pose_estimator(self):
        """Initialize the 3D pose estimator (DanceHRNet)."""
        from .models.dancehrnet import DanceHRNet
        
        pose_config = self.config.get('pose_estimation', {})
        self.pose_estimator = DanceHRNet(pose_config)
        
    def _init_multimodal_analyzer(self):
        """Initialize the multimodal analyzer (DanceFormer)."""
        from .models.danceformer import DanceFormer
        
        multimodal_config = self.config.get('multimodal_analysis', {})
        self.multimodal_analyzer = DanceFormer(multimodal_config)
        
    def _init_similarity_analyzer(self):
        """Initialize the similarity analyzer (DanceDTW)."""
        from .analysis.dance_dtw import DanceDTW
        
        similarity_config = self.config.get('similarity_analysis', {})
        self.similarity_analyzer = DanceDTW(similarity_config)
    
    def process_video(self, video_path: str, music_path: Optional[str] = None) -> Dict:
        """
        Process a dance video and extract analysis results.
        
        Args:
            video_path: Path to the dance video
            music_path: Path to the corresponding music file (optional)
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract 3D poses from video
        pose_results = self.pose_estimator.extract_poses(video_path)
        
        # If music path is provided, perform multimodal analysis
        multimodal_results = {}
        if music_path:
            multimodal_results = self.multimodal_analyzer.analyze(pose_results, music_path)
        
        # Combine results
        results = {
            'pose_results': pose_results,
            'multimodal_results': multimodal_results
        }
        
        return results
    
    def compare_choreographies(self, reference_path: str, comparison_path: str) -> Dict:
        """
        Compare two choreographies and compute similarity.
        
        Args:
            reference_path: Path to the reference video
            comparison_path: Path to the comparison video
            
        Returns:
            Dictionary containing similarity analysis results
        """
        # Extract 3D poses from videos
        reference_poses = self.pose_estimator.extract_poses(reference_path)
        comparison_poses = self.pose_estimator.extract_poses(comparison_path)
        
        # Compute similarity
        similarity_results = self.similarity_analyzer.compute_similarity(
            reference_poses['poses'], comparison_poses['poses']
        )
        
        return similarity_results
    
    def generate_report(self, results: Dict, output_path: str) -> str:
        """
        Generate a detailed analysis report based on results.
        
        Args:
            results: Results dictionary from process_video or compare_choreographies
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        from .utils.report_generator import ReportGenerator
        
        generator = ReportGenerator(self.config.get('report', {}))
        report_path = generator.generate(results, output_path)
        
        return report_path
