"""
Utility modules for the DanceMotionAI system.

This package contains various utility functions for visualization,
metrics calculation, and data manipulation for 3D dance motion analysis.
"""

from .visualization import (
    plot_3d_pose,
    plot_pose_sequence,
    plot_pose_comparison,
    plot_similarity_heatmap,
    plot_joint_trajectories,
    plot_dtw_alignment_path,
    plot_radar_metrics,
    save_visualization_to_html,
    export_3d_model
)

from .metrics import (
    mpjpe,
    p_mpjpe,
    n_mpjpe,
    compute_joint_angles,
    compute_pose_velocity,
    compute_pose_acceleration,
    compute_jerk,
    compute_smoothness,
    compute_energy,
    compute_dtw_distance,
    compute_dtw_similarity,
    compute_joint_position_similarity,
    compute_joint_velocity_similarity,
    compute_beat_alignment,
    compute_dance_metrics,
    compute_originality_score,
    compute_style_consistency,
    compute_plagiarism_score
)

__all__ = [
    # Visualization functions
    'plot_3d_pose',
    'plot_pose_sequence',
    'plot_pose_comparison',
    'plot_similarity_heatmap',
    'plot_joint_trajectories',
    'plot_dtw_alignment_path',
    'plot_radar_metrics',
    'save_visualization_to_html',
    'export_3d_model',
    
    # Metrics functions
    'mpjpe',
    'p_mpjpe',
    'n_mpjpe',
    'compute_joint_angles',
    'compute_pose_velocity',
    'compute_pose_acceleration',
    'compute_jerk',
    'compute_smoothness',
    'compute_energy',
    'compute_dtw_distance',
    'compute_dtw_similarity',
    'compute_joint_position_similarity',
    'compute_joint_velocity_similarity',
    'compute_beat_alignment',
    'compute_dance_metrics',
    'compute_originality_score',
    'compute_style_consistency',
    'compute_plagiarism_score'
]
