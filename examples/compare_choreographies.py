"""
Example script for using the 3D-DanceDTW algorithm for choreography similarity analysis.

This script demonstrates how to:
1. Load two dance videos
2. Extract 3D pose sequences
3. Compare the choreographies using the 3D-DanceDTW algorithm
4. Visualize the results

Usage:
    python examples/compare_choreographies.py --reference path/to/reference_video.mp4 --comparison path/to/comparison_video.mp4

Author: JJshome
Date: May 13, 2025
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dance_dtw import DanceDTW

# Mock function to simulate 3D pose extraction (in a real implementation, this would use actual pose estimation)
def extract_3d_poses(video_path, num_frames=100, num_joints=25):
    """
    Simulate extracting 3D poses from a video.
    
    In a real implementation, this would use DanceHRNet to extract actual 3D poses.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to simulate
        num_joints: Number of joints in the skeleton
        
    Returns:
        Numpy array of shape (frames, joints, 3) containing 3D joint coordinates
    """
    print(f"Extracting 3D poses from {video_path}...")
    
    # In a real implementation, this would load the video and process it
    # For demonstration, we'll generate random pose data
    poses = np.random.rand(num_frames, num_joints, 3)
    
    # Add some structure to the random data to make it more realistic
    # Create a walking motion pattern
    t = np.linspace(0, 2*np.pi, num_frames)
    
    # Simulate leg movement
    left_hip_idx, right_hip_idx = 12, 9  # Example indices
    left_knee_idx, right_knee_idx = 13, 10
    left_ankle_idx, right_ankle_idx = 14, 11
    
    # Hip movements
    poses[:, left_hip_idx, 0] = 0.1 * np.sin(t)
    poses[:, right_hip_idx, 0] = 0.1 * np.sin(t + np.pi)
    
    # Knee movements
    poses[:, left_knee_idx, 0] = 0.15 * np.sin(t)
    poses[:, right_knee_idx, 0] = 0.15 * np.sin(t + np.pi)
    
    # Ankle movements
    poses[:, left_ankle_idx, 0] = 0.2 * np.sin(t)
    poses[:, right_ankle_idx, 0] = 0.2 * np.sin(t + np.pi)
    
    print(f"Extracted {num_frames} frames with {num_joints} joints.")
    return poses

def visualize_similarity(similarity_results, output_path=None):
    """
    Visualize the similarity analysis results.
    
    Args:
        similarity_results: Dictionary containing similarity analysis results
        output_path: Path to save the visualization (optional)
    """
    similarity_score = similarity_results['similarity_score']
    is_similar = similarity_results['is_similar']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot similarity score
    ax1.bar(['Similarity Score'], [similarity_score], color='orange')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.set_title('Choreography Similarity Score')
    ax1.text(0, similarity_score/2, f"{similarity_score:.4f}", 
             ha='center', va='center', fontweight='bold')
    
    # Add horizontal line for similarity threshold
    threshold = 0.7  # This should match the threshold in the DanceDTW configuration
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
    ax1.text(-0.2, threshold + 0.02, f"Threshold: {threshold}", color='red')
    
    # Plot similarity verdict
    verdict_color = 'green' if is_similar else 'red'
    verdict_text = 'Similar' if is_similar else 'Not Similar'
    ax2.pie([1], labels=[verdict_text], colors=[verdict_color], autopct='', 
            wedgeprops={'alpha': 0.7})
    ax2.set_title('Similarity Verdict')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare dance choreographies using 3D-DanceDTW')
    parser.add_argument('--reference', required=True, help='Path to reference video')
    parser.add_argument('--comparison', required=True, help='Path to comparison video')
    parser.add_argument('--config', default='../configs/default_config.yaml', help='Path to configuration file')
    parser.add_argument('--output', help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            dtw_config = config.get('similarity_analysis', {})
    except Exception as e:
        print(f"Error loading configuration: {e}")
        dtw_config = {}
    
    # Extract 3D poses from videos
    # In a real implementation, this would use DanceHRNet
    reference_poses = extract_3d_poses(args.reference)
    comparison_poses = extract_3d_poses(args.comparison)
    
    # Initialize DanceDTW with configuration
    dance_dtw = DanceDTW(dtw_config)
    
    # Compute similarity between the poses
    print("Computing choreography similarity...")
    similarity_results = dance_dtw.compute_similarity(reference_poses, comparison_poses)
    
    # Print results
    print("\nSimilarity Analysis Results:")
    print(f"Similarity Score: {similarity_results['similarity_score']:.4f}")
    print(f"Is Similar: {similarity_results['is_similar']}")
    
    # Visualize results
    visualize_similarity(similarity_results, args.output)

if __name__ == "__main__":
    main()
