"""
Metrics module for the DanceMotionAI system.

This module provides functions for calculating metrics related to 
3D dance motion analysis, such as joint position errors, 
pose similarity, and dance quality assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.spatial.distance import cdist
from fastdtw import fastdtw


def mpjpe(pred_pose: np.ndarray, gt_pose: np.ndarray) -> float:
    """
    Calculate Mean Per Joint Position Error (MPJPE) between 
    predicted and ground truth 3D poses.
    
    Args:
        pred_pose: Predicted 3D pose with shape [num_joints, 3]
        gt_pose: Ground truth 3D pose with shape [num_joints, 3]
        
    Returns:
        MPJPE value in the same unit as the input poses (typically mm)
    """
    if pred_pose.shape != gt_pose.shape:
        raise ValueError(f"Shape mismatch: pred_pose {pred_pose.shape}, gt_pose {gt_pose.shape}")
    
    return np.mean(np.sqrt(np.sum((pred_pose - gt_pose) ** 2, axis=1)))


def p_mpjpe(pred_pose: np.ndarray, gt_pose: np.ndarray) -> float:
    """
    Calculate Procrustes-aligned Mean Per Joint Position Error (P-MPJPE)
    between predicted and ground truth 3D poses.
    
    Args:
        pred_pose: Predicted 3D pose with shape [num_joints, 3]
        gt_pose: Ground truth 3D pose with shape [num_joints, 3]
        
    Returns:
        P-MPJPE value in the same unit as the input poses (typically mm)
    """
    from scipy.spatial import procrustes
    
    # Center the poses
    pred_centered = pred_pose - np.mean(pred_pose, axis=0)
    gt_centered = gt_pose - np.mean(gt_pose, axis=0)
    
    # Perform Procrustes alignment
    _, pred_aligned, _ = procrustes(gt_centered, pred_centered)
    
    # Calculate MPJPE on aligned poses
    return np.mean(np.sqrt(np.sum((pred_aligned - gt_centered) ** 2, axis=1)))


def n_mpjpe(pred_pose: np.ndarray, gt_pose: np.ndarray) -> float:
    """
    Calculate Normalized Mean Per Joint Position Error (N-MPJPE)
    between predicted and ground truth 3D poses.
    
    Args:
        pred_pose: Predicted 3D pose with shape [num_joints, 3]
        gt_pose: Ground truth 3D pose with shape [num_joints, 3]
        
    Returns:
        N-MPJPE value (scale-invariant)
    """
    # Center the poses
    pred_centered = pred_pose - np.mean(pred_pose, axis=0)
    gt_centered = gt_pose - np.mean(gt_pose, axis=0)
    
    # Scale the poses
    pred_scaled = pred_centered / np.sqrt(np.sum(pred_centered ** 2))
    gt_scaled = gt_centered / np.sqrt(np.sum(gt_centered ** 2))
    
    # Calculate MPJPE on normalized poses
    return np.mean(np.sqrt(np.sum((pred_scaled - gt_scaled) ** 2, axis=1)))


def compute_joint_angles(pose: np.ndarray, joint_connections: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Compute joint angles from 3D pose data.
    
    Args:
        pose: 3D pose data with shape [num_joints, 3]
        joint_connections: List of (joint_idx, parent_idx, child_idx) triplets
            representing the joints for which to compute angles
            
    Returns:
        Array of joint angles in radians with shape [len(joint_connections)]
    """
    angles = []
    
    for joint, parent, child in joint_connections:
        # Compute vectors from joint to parent and child
        v1 = pose[parent] - pose[joint]
        v2 = pose[child] - pose[joint]
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Compute angle using dot product
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        angles.append(angle)
    
    return np.array(angles)


def compute_pose_velocity(pose_sequence: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    Compute velocity of each joint in a pose sequence.
    
    Args:
        pose_sequence: 3D pose sequence with shape [num_frames, num_joints, 3]
        fps: Frames per second of the sequence
        
    Returns:
        Joint velocity array with shape [num_frames-1, num_joints, 3]
    """
    # Compute position differences between consecutive frames
    position_diff = np.diff(pose_sequence, axis=0)
    
    # Convert to velocity in units/second
    velocity = position_diff * fps
    
    return velocity


def compute_pose_acceleration(pose_sequence: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    Compute acceleration of each joint in a pose sequence.
    
    Args:
        pose_sequence: 3D pose sequence with shape [num_frames, num_joints, 3]
        fps: Frames per second of the sequence
        
    Returns:
        Joint acceleration array with shape [num_frames-2, num_joints, 3]
    """
    # Compute velocity
    velocity = compute_pose_velocity(pose_sequence, fps)
    
    # Compute acceleration as derivative of velocity
    acceleration = np.diff(velocity, axis=0) * fps
    
    return acceleration


def compute_jerk(pose_sequence: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    Compute jerk (rate of change of acceleration) for each joint.
    
    Args:
        pose_sequence: 3D pose sequence with shape [num_frames, num_joints, 3]
        fps: Frames per second of the sequence
        
    Returns:
        Joint jerk array with shape [num_frames-3, num_joints, 3]
    """
    # Compute acceleration
    acceleration = compute_pose_acceleration(pose_sequence, fps)
    
    # Compute jerk as derivative of acceleration
    jerk = np.diff(acceleration, axis=0) * fps
    
    return jerk


def compute_smoothness(pose_sequence: np.ndarray, fps: float = 30.0) -> float:
    """
    Compute motion smoothness using mean squared jerk.
    
    Args:
        pose_sequence: 3D pose sequence with shape [num_frames, num_joints, 3]
        fps: Frames per second of the sequence
        
    Returns:
        Smoothness score (lower values indicate smoother motion)
    """
    # Compute jerk
    jerk = compute_jerk(pose_sequence, fps)
    
    # Compute mean squared jerk
    mean_squared_jerk = np.mean(np.sum(jerk ** 2, axis=2))
    
    return mean_squared_jerk


def compute_energy(pose_sequence: np.ndarray, fps: float = 30.0) -> float:
    """
    Compute the energy of the motion.
    
    Args:
        pose_sequence: 3D pose sequence with shape [num_frames, num_joints, 3]
        fps: Frames per second of the sequence
        
    Returns:
        Energy score (higher values indicate more energetic motion)
    """
    # Compute velocity
    velocity = compute_pose_velocity(pose_sequence, fps)
    
    # Compute kinetic energy (proportional to squared velocity)
    energy = np.mean(np.sum(velocity ** 2, axis=2))
    
    return energy


def compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray, radius: int = 10) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Compute Dynamic Time Warping (DTW) distance between two sequences.
    
    Args:
        seq1: First sequence with shape [len1, feature_dim]
        seq2: Second sequence with shape [len2, feature_dim]
        radius: Sakoe-Chiba band radius for DTW
        
    Returns:
        DTW distance and the optimal warping path as a list of (i, j) pairs
    """
    # Flatten each frame to a feature vector if sequences contain 3D poses
    if len(seq1.shape) == 3:  # [num_frames, num_joints, 3]
        seq1_flat = seq1.reshape(seq1.shape[0], -1)
        seq2_flat = seq2.reshape(seq2.shape[0], -1)
    else:
        seq1_flat = seq1
        seq2_flat = seq2
    
    # Compute DTW distance and path
    distance, path = fastdtw(seq1_flat, seq2_flat, dist=lambda x, y: np.linalg.norm(x - y), radius=radius)
    
    return distance, path


def compute_dtw_similarity(seq1: np.ndarray, seq2: np.ndarray, radius: int = 10) -> float:
    """
    Compute similarity between two sequences using DTW.
    
    Args:
        seq1: First sequence with shape [len1, feature_dim]
        seq2: Second sequence with shape [len2, feature_dim]
        radius: Sakoe-Chiba band radius for DTW
        
    Returns:
        Similarity score between 0 and 1 (1 means identical)
    """
    distance, _ = compute_dtw_distance(seq1, seq2, radius)
    
    # Convert distance to similarity
    # Use an exponential decay function to map distance to [0, 1]
    similarity = np.exp(-distance / 1000)  # Scale factor may need adjustment
    
    return similarity


def compute_joint_position_similarity(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
    """
    Compute frame-by-frame joint position similarity between two sequences.
    
    Args:
        seq1: First sequence with shape [len1, num_joints, 3]
        seq2: Second sequence with shape [len2, num_joints, 3]
        
    Returns:
        Similarity matrix with shape [len1, len2]
    """
    # Flatten each frame to a feature vector
    seq1_flat = seq1.reshape(seq1.shape[0], -1)
    seq2_flat = seq2.reshape(seq2.shape[0], -1)
    
    # Compute pairwise distances
    distance_matrix = cdist(seq1_flat, seq2_flat, metric='euclidean')
    
    # Convert distances to similarities (using exponential decay)
    max_dist = np.max(distance_matrix)
    similarity_matrix = np.exp(-distance_matrix / max_dist)
    
    return similarity_matrix


def compute_joint_velocity_similarity(seq1: np.ndarray, seq2: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    Compute frame-by-frame joint velocity similarity between two sequences.
    
    Args:
        seq1: First sequence with shape [len1, num_joints, 3]
        seq2: Second sequence with shape [len2, num_joints, 3]
        fps: Frames per second of the sequences
        
    Returns:
        Similarity matrix with shape [len1-1, len2-1]
    """
    # Compute velocities
    vel1 = compute_pose_velocity(seq1, fps)
    vel2 = compute_pose_velocity(seq2, fps)
    
    # Compute similarity matrix
    return compute_joint_position_similarity(vel1, vel2)


def compute_beat_alignment(motion_energy: np.ndarray, beats: np.ndarray, fps: float = 30.0) -> float:
    """
    Compute alignment between motion energy peaks and music beats.
    
    Args:
        motion_energy: Energy of motion per frame with shape [num_frames]
        beats: Array of beat timestamps in seconds
        fps: Frames per second of the motion sequence
        
    Returns:
        Beat alignment score between 0 and 1
    """
    # Convert beat timestamps to frame indices
    beat_frames = np.round(beats * fps).astype(int)
    beat_frames = beat_frames[beat_frames < len(motion_energy)]
    
    # Compute local energy peaks
    from scipy.signal import find_peaks
    energy_peaks, _ = find_peaks(motion_energy, distance=5)
    
    # For each beat, find the closest energy peak
    alignment_scores = []
    for beat_frame in beat_frames:
        if len(energy_peaks) == 0:
            alignment_scores.append(0)
            continue
            
        # Find the closest energy peak
        distances = np.abs(energy_peaks - beat_frame)
        min_distance = np.min(distances)
        
        # Convert distance to score (closer is better)
        score = np.exp(-min_distance / 5)  # Scale factor determines how quickly score decays with distance
        alignment_scores.append(score)
    
    # Return average alignment score
    if len(alignment_scores) == 0:
        return 0.0
    return np.mean(alignment_scores)


def compute_dance_metrics(reference_seq: np.ndarray, 
                         comparison_seq: np.ndarray,
                         reference_beats: Optional[np.ndarray] = None,
                         fps: float = 30.0) -> Dict[str, float]:
    """
    Compute comprehensive dance quality metrics.
    
    Args:
        reference_seq: Reference dance sequence with shape [len1, num_joints, 3]
        comparison_seq: Comparison dance sequence with shape [len2, num_joints, 3]
        reference_beats: Optional array of beat timestamps for the reference sequence
        fps: Frames per second of the sequences
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Time warp comparison sequences if lengths differ
    if reference_seq.shape[0] != comparison_seq.shape[0]:
        distance, path = compute_dtw_distance(reference_seq, comparison_seq)
        
        # Create warped comparison sequence
        warped_comparison = np.zeros_like(reference_seq)
        for i, j in path:
            if i < reference_seq.shape[0]:
                warped_comparison[i] = comparison_seq[j]
        
        comparison_seq_aligned = warped_comparison
    else:
        comparison_seq_aligned = comparison_seq
    
    # Position accuracy
    metrics['position_accuracy'] = 1.0 - min(1.0, mpjpe(comparison_seq_aligned, reference_seq) / 100.0)
    
    # Smoothness
    ref_smoothness = compute_smoothness(reference_seq, fps)
    comp_smoothness = compute_smoothness(comparison_seq_aligned, fps)
    # Lower is better for smoothness (lower jerk)
    metrics['smoothness'] = max(0, 1.0 - abs(comp_smoothness - ref_smoothness) / ref_smoothness)
    
    # Energy match
    ref_energy = compute_energy(reference_seq, fps)
    comp_energy = compute_energy(comparison_seq_aligned, fps)
    # Energy should match reference
    metrics['energy_match'] = max(0, 1.0 - abs(comp_energy - ref_energy) / ref_energy)
    
    # Tempo match (if beats are provided)
    if reference_beats is not None:
        # Compute motion energy for both sequences
        ref_motion_energy = np.mean(np.sum(compute_pose_velocity(reference_seq, fps) ** 2, axis=(1, 2)), axis=0)
        comp_motion_energy = np.mean(np.sum(compute_pose_velocity(comparison_seq_aligned, fps) ** 2, axis=(1, 2)), axis=0)
        
        ref_beat_alignment = compute_beat_alignment(ref_motion_energy, reference_beats, fps)
        comp_beat_alignment = compute_beat_alignment(comp_motion_energy, reference_beats, fps)
        
        metrics['tempo_match'] = comp_beat_alignment / max(ref_beat_alignment, 1e-5)
    
    # Overall similarity
    metrics['overall_similarity'] = compute_dtw_similarity(reference_seq, comparison_seq)
    
    return metrics


def compute_originality_score(query_seq: np.ndarray, 
                             reference_seqs: List[np.ndarray],
                             threshold: float = 0.8) -> float:
    """
    Compute originality score for a dance sequence compared to reference dataset.
    
    Args:
        query_seq: Query dance sequence with shape [len_q, num_joints, 3]
        reference_seqs: List of reference dance sequences
        threshold: Similarity threshold above which sequences are considered similar
        
    Returns:
        Originality score between 0 and 1 (1 means highly original)
    """
    # Compute similarity to each reference sequence
    similarities = []
    for ref_seq in reference_seqs:
        similarity = compute_dtw_similarity(query_seq, ref_seq)
        similarities.append(similarity)
    
    if not similarities:
        return 1.0  # If no reference sequences, consider completely original
    
    # Count reference sequences that are similar
    similar_count = sum(1 for s in similarities if s >= threshold)
    
    # Calculate originality based on proportion of dissimilar sequences
    originality = 1.0 - (similar_count / len(reference_seqs))
    
    return originality


def compute_style_consistency(sequence: np.ndarray, style_model=None) -> float:
    """
    Compute consistency of dance style throughout the sequence.
    
    Args:
        sequence: Dance sequence with shape [num_frames, num_joints, 3]
        style_model: Optional model for style classification
        
    Returns:
        Style consistency score between 0 and 1
    """
    # If no style model is provided, use simple velocity-based approach
    if style_model is None:
        # Compute velocities
        velocities = compute_pose_velocity(sequence)
        
        # Compute variance of velocity magnitudes over time
        vel_magnitudes = np.sqrt(np.sum(velocities ** 2, axis=(1, 2)))
        
        # Higher variance means less consistency
        consistency = np.exp(-np.var(vel_magnitudes) / 10.0)
        
        return consistency
    else:
        # Use style model for more sophisticated assessment
        # This is a placeholder for a real style model implementation
        raise NotImplementedError("Style model-based consistency not implemented yet")


def compute_plagiarism_score(query_seq: np.ndarray, 
                           reference_seq: np.ndarray,
                           window_size: int = 30,
                           stride: int = 15,
                           threshold: float = 0.85) -> Dict[str, Any]:
    """
    Detect potential plagiarism in dance choreography.
    
    Args:
        query_seq: Query dance sequence with shape [len_q, num_joints, 3]
        reference_seq: Reference dance sequence with shape [len_r, num_joints, 3]
        window_size: Size of sliding window for comparison (frames)
        stride: Step size for sliding window
        threshold: Similarity threshold above which segments are considered plagiarized
        
    Returns:
        Dictionary containing plagiarism score and detected segments
    """
    result = {
        'overall_score': 0.0,
        'detected_segments': []
    }
    
    # Get sequence lengths
    len_q = query_seq.shape[0]
    len_r = reference_seq.shape[0]
    
    # Skip if sequences are too short
    if len_q < window_size or len_r < window_size:
        return result
    
    # Create windows
    query_windows = []
    query_indices = []
    for i in range(0, len_q - window_size + 1, stride):
        query_windows.append(query_seq[i:i+window_size])
        query_indices.append((i, i+window_size))
    
    reference_windows = []
    reference_indices = []
    for i in range(0, len_r - window_size + 1, stride):
        reference_windows.append(reference_seq[i:i+window_size])
        reference_indices.append((i, i+window_size))
    
    # Compare each query window with each reference window
    plagiarized_segments = []
    
    for q_idx, q_window in enumerate(query_windows):
        for r_idx, r_window in enumerate(reference_windows):
            similarity = compute_dtw_similarity(q_window, r_window)
            
            if similarity >= threshold:
                # This segment is potentially plagiarized
                plagiarized_segments.append({
                    'query_start': query_indices[q_idx][0],
                    'query_end': query_indices[q_idx][1],
                    'reference_start': reference_indices[r_idx][0],
                    'reference_end': reference_indices[r_idx][1],
                    'similarity': similarity
                })
    
    # Merge overlapping segments
    if plagiarized_segments:
        plagiarized_segments.sort(key=lambda x: x['query_start'])
        
        merged_segments = [plagiarized_segments[0]]
        
        for segment in plagiarized_segments[1:]:
            last = merged_segments[-1]
            
            # Check if segments overlap
            if segment['query_start'] <= last['query_end']:
                # Extend the last segment
                last['query_end'] = max(last['query_end'], segment['query_end'])
                last['reference_end'] = max(last['reference_end'], segment['reference_end'])
                last['similarity'] = max(last['similarity'], segment['similarity'])
            else:
                # Add as a new segment
                merged_segments.append(segment)
        
        # Calculate overall plagiarism score based on coverage and similarity
        total_frames = sum(seg['query_end'] - seg['query_start'] for seg in merged_segments)
        weighted_sim = sum((seg['query_end'] - seg['query_start']) * seg['similarity'] for seg in merged_segments)
        
        coverage_ratio = min(1.0, total_frames / len_q)
        avg_similarity = weighted_sim / total_frames if total_frames > 0 else 0
        
        result['overall_score'] = coverage_ratio * avg_similarity
        result['detected_segments'] = merged_segments
    
    return result
