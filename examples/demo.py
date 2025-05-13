"""
Example script demonstrating how to use the DanceMotionAI system.

This script shows how to:
1. Load 3D pose data
2. Visualize 3D poses
3. Compute metrics between dance sequences
4. Detect potential choreography plagiarism
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils import (
    plot_3d_pose,
    plot_pose_sequence,
    plot_pose_comparison,
    plot_similarity_heatmap,
    plot_dtw_alignment_path,
    plot_radar_metrics,
    
    mpjpe,
    compute_dtw_similarity,
    compute_joint_position_similarity,
    compute_dance_metrics,
    compute_plagiarism_score
)


def generate_sample_pose(num_joints=25):
    """Generate a sample 3D pose for demonstration purposes."""
    pose = np.zeros((num_joints, 3))
    
    # Head
    pose[0] = [0, 0, 0]  # Nose
    pose[1] = [-0.1, 0.1, 0]  # Left eye
    pose[2] = [0.1, 0.1, 0]  # Right eye
    pose[3] = [-0.15, 0.05, -0.1]  # Left ear
    pose[4] = [0.15, 0.05, -0.1]  # Right ear
    
    # Torso
    pose[5] = [-0.2, -0.2, 0]  # Left shoulder
    pose[6] = [0.2, -0.2, 0]  # Right shoulder
    pose[11] = [-0.15, -0.7, 0]  # Left hip
    pose[12] = [0.15, -0.7, 0]  # Right hip
    
    # Left arm
    pose[7] = [-0.4, -0.4, 0]  # Left elbow
    pose[9] = [-0.5, -0.6, 0]  # Left wrist
    
    # Right arm
    pose[8] = [0.4, -0.4, 0]  # Right elbow
    pose[10] = [0.5, -0.6, 0]  # Right wrist
    
    # Left leg
    pose[13] = [-0.2, -1.0, 0]  # Left knee
    pose[15] = [-0.2, -1.3, 0]  # Left ankle
    
    # Right leg
    pose[14] = [0.2, -1.0, 0]  # Right knee
    pose[16] = [0.2, -1.3, 0]  # Right ankle
    
    return pose


def generate_dance_sequence(num_frames=60, num_joints=25, style='wave'):
    """Generate a synthetic dance sequence for demonstration purposes."""
    sequence = np.zeros((num_frames, num_joints, 3))
    
    # Start with a base pose
    base_pose = generate_sample_pose(num_joints)
    
    for i in range(num_frames):
        # Deep copy the base pose
        sequence[i] = base_pose.copy()
        
        if style == 'wave':
            # Add a wave motion to the arms
            t = i / 10.0
            
            # Left arm wave
            sequence[i, 7, 0] = base_pose[7, 0] + 0.1 * np.sin(t)  # Left elbow x
            sequence[i, 7, 1] = base_pose[7, 1] + 0.1 * np.cos(t)  # Left elbow y
            sequence[i, 9, 0] = base_pose[9, 0] + 0.15 * np.sin(t + 0.5)  # Left wrist x
            sequence[i, 9, 1] = base_pose[9, 1] + 0.15 * np.cos(t + 0.5)  # Left wrist y
            
            # Right arm wave (slightly out of phase)
            sequence[i, 8, 0] = base_pose[8, 0] + 0.1 * np.sin(t + 1.0)  # Right elbow x
            sequence[i, 8, 1] = base_pose[8, 1] + 0.1 * np.cos(t + 1.0)  # Right elbow y
            sequence[i, 10, 0] = base_pose[10, 0] + 0.15 * np.sin(t + 1.5)  # Right wrist x
            sequence[i, 10, 1] = base_pose[10, 1] + 0.15 * np.cos(t + 1.5)  # Right wrist y
            
            # Slight body movement
            sequence[i, 0:5, 0] += 0.05 * np.sin(t / 2)  # Head sway
            sequence[i, 5:13, 0] += 0.03 * np.sin(t / 3)  # Torso sway
            
        elif style == 'jump':
            # Add a jumping motion
            t = i / num_frames
            
            # Vertical movement for all joints
            jump_height = 0.2 * np.sin(np.pi * 4 * t)
            sequence[i, :, 1] += jump_height
            
            # Legs bend before and after jump
            leg_bend = -0.1 * np.cos(np.pi * 4 * t)
            sequence[i, 13:15, 1] += leg_bend  # Knees
            
        elif style == 'spin':
            # Add a spinning motion
            angle = i * (2 * np.pi / num_frames) * 2  # 2 full rotations
            
            # Rotate the whole body around the Y axis
            for j in range(num_joints):
                x = base_pose[j, 0]
                z = base_pose[j, 2]
                sequence[i, j, 0] = x * np.cos(angle) - z * np.sin(angle)
                sequence[i, j, 2] = x * np.sin(angle) + z * np.cos(angle)
            
            # Add some up and down movement
            sequence[i, :, 1] += 0.05 * np.sin(4 * angle)
    
    return sequence


def create_similar_sequence(original_sequence, noise_level=0.02, time_warp_factor=0.9):
    """Create a sequence similar to the original but with noise and time warping."""
    num_frames, num_joints, _ = original_sequence.shape
    
    # Apply time warping
    new_num_frames = int(num_frames * time_warp_factor)
    indices = np.linspace(0, num_frames-1, new_num_frames)
    
    # Create sequence with time warping
    similar_sequence = np.zeros((new_num_frames, num_joints, 3))
    for i in range(new_num_frames):
        # Find the two closest frames
        idx_low = int(np.floor(indices[i]))
        idx_high = min(idx_low + 1, num_frames - 1)
        weight = indices[i] - idx_low
        
        # Interpolate between frames
        similar_sequence[i] = (1 - weight) * original_sequence[idx_low] + weight * original_sequence[idx_high]
    
    # Add noise
    noise = np.random.normal(0, noise_level, similar_sequence.shape)
    similar_sequence += noise
    
    return similar_sequence


def main():
    """Main function demonstrating the DanceMotionAI capabilities."""
    print("DanceMotionAI Demo")
    print("=================")
    
    # 1. Visualize a single pose
    print("\n1. Visualizing a single pose...")
    sample_pose = generate_sample_pose()
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')
    plot_3d_pose(sample_pose, ax=ax, title="Sample 3D Pose")
    plt.tight_layout()
    plt.savefig("sample_pose.png")
    print(f"Saved visualization to 'sample_pose.png'")
    
    # 2. Generate and visualize dance sequences
    print("\n2. Generating dance sequences...")
    
    # Generate original sequence
    original_sequence = generate_dance_sequence(num_frames=60, style='wave')
    print(f"Generated original sequence with shape: {original_sequence.shape}")
    
    # Create similar sequence with noise and time warping
    similar_sequence = create_similar_sequence(original_sequence, noise_level=0.02, time_warp_factor=1.1)
    print(f"Generated similar sequence with shape: {similar_sequence.shape}")
    
    # Create a different dance sequence
    different_sequence = generate_dance_sequence(num_frames=60, style='jump')
    print(f"Generated different sequence with shape: {different_sequence.shape}")
    
    # 3. Visualize sequence comparison
    print("\n3. Visualizing sequence comparison...")
    
    # Compare sample frames from original and similar sequences
    frame_idx = 30
    if frame_idx < len(original_sequence) and frame_idx < len(similar_sequence):
        plot_pose_comparison(
            original_sequence[frame_idx], 
            similar_sequence[min(frame_idx, len(similar_sequence)-1)],
            title1="Original Sequence (Frame 30)",
            title2="Similar Sequence (Frame 30)",
        )
        plt.savefig("pose_comparison.png")
        print(f"Saved comparison visualization to 'pose_comparison.png'")
    
    # 4. Compute similarity metrics
    print("\n4. Computing similarity metrics...")
    
    # MPJPE between sample frames
    frame_idx = 30
    if frame_idx < len(original_sequence) and frame_idx < len(similar_sequence):
        original_frame = original_sequence[frame_idx]
        similar_frame = similar_sequence[min(frame_idx, len(similar_sequence)-1)]
        error = mpjpe(similar_frame, original_frame)
        print(f"MPJPE between original and similar frames: {error:.4f}")
    
    # DTW similarity between sequences
    similarity_to_similar = compute_dtw_similarity(original_sequence, similar_sequence)
    similarity_to_different = compute_dtw_similarity(original_sequence, different_sequence)
    
    print(f"DTW similarity between original and similar sequences: {similarity_to_similar:.4f}")
    print(f"DTW similarity between original and different sequences: {similarity_to_different:.4f}")
    
    # 5. Compute similarity matrix and visualize
    print("\n5. Computing similarity matrix...")
    
    # Similarity matrix between original and similar
    similarity_matrix = compute_joint_position_similarity(original_sequence, similar_sequence)
    
    plt.figure(figsize=(10, 8))
    plot_similarity_heatmap(
        similarity_matrix,
        xlabel="Original Sequence Frames",
        ylabel="Similar Sequence Frames",
        title="Pose Similarity Matrix"
    )
    plt.savefig("similarity_matrix.png")
    print(f"Saved similarity matrix to 'similarity_matrix.png'")
    
    # 6. Compute comprehensive dance metrics
    print("\n6. Computing comprehensive dance metrics...")
    
    metrics_similar = compute_dance_metrics(original_sequence, similar_sequence)
    metrics_different = compute_dance_metrics(original_sequence, different_sequence)
    
    print("Metrics for similar sequence:")
    for metric, value in metrics_similar.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMetrics for different sequence:")
    for metric, value in metrics_different.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize radar chart of metrics
    plt.figure(figsize=(10, 10))
    plot_radar_metrics(
        metrics_similar,
        title="Dance Performance Metrics (Similar Sequence)"
    )
    plt.savefig("radar_metrics.png")
    print(f"Saved radar metrics visualization to 'radar_metrics.png'")
    
    # 7. Detect plagiarism
    print("\n7. Detecting potential plagiarism...")
    
    # Create a partially plagiarized sequence
    plagiarized_sequence = different_sequence.copy()
    
    # Insert a segment from the original sequence
    start_orig = 10
    end_orig = 40
    start_plag = 20
    segment_length = end_orig - start_orig
    
    if (start_plag + segment_length <= len(plagiarized_sequence) and 
        end_orig <= len(original_sequence)):
        
        plagiarized_sequence[start_plag:start_plag+segment_length] = original_sequence[start_orig:end_orig]
        
        # Add some noise to disguise the plagiarism
        noise = np.random.normal(0, 0.01, (segment_length, plagiarized_sequence.shape[1], 3))
        plagiarized_sequence[start_plag:start_plag+segment_length] += noise
        
        # Detect plagiarism
        result = compute_plagiarism_score(
            plagiarized_sequence, 
            original_sequence,
            window_size=15,
            stride=5,
            threshold=0.75
        )
        
        print(f"Plagiarism score: {result['overall_score']:.4f}")
        print(f"Number of detected segments: {len(result['detected_segments'])}")
        
        for i, segment in enumerate(result['detected_segments']):
            print(f"  Segment {i+1}:")
            print(f"    Query frames: {segment['query_start']} to {segment['query_end']}")
            print(f"    Reference frames: {segment['reference_start']} to {segment['reference_end']}")
            print(f"    Similarity: {segment['similarity']:.4f}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
