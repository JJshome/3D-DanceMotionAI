"""
DanceMotionAI: High-Precision 3D Pose Estimation-based AI Dance Choreography Analysis and Evaluation System

This package provides tools for:
- 3D pose estimation from dance videos
- Multimodal analysis of dance and music
- Similarity analysis of choreographies
- Real-time feedback for dance education

For more information, visit: https://github.com/JJshome/3D-DanceMotionAI
"""

__version__ = '0.1.0'
__author__ = 'JJshome'

# Core modules
from src.dance_dtw import (
    DanceDTW,
    FeatureExtractor,
    WaveletTransformer,
    AdaptiveWeightModel
)
