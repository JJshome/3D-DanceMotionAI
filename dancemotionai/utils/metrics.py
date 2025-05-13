"""
Metrics module for the DanceMotionAI system.

This module provides functions for computing various metrics for pose analysis
including joint angles, velocities, accelerations, and quantitative evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


def compute_joint_angles(pose: np.ndarray) -> np.ndarray:
    """
    Compute joint angles for a single pose.
    
    Args:
        pose: 3D pose data [num_joints, 3]
        
    Returns:
        Joint angles in radians [num_angles]
    """
    # Define pairs of joints to compute angles for
    # This is a simplified example - in practice, would define specific triplets
    # of joints for anatomically meaningful angles
    joint_triplets = [
        # Right arm
        (2, 3, 4),    # Shoulder, elbow, wrist
        # Left arm
        (5, 6, 7),    # Shoulder, elbow, wrist
        # Right leg
        (9, 10, 11),  # Hip, knee, ankle
        # Left leg
        (12, 13, 14), # Hip, knee, ankle
        # Spine
        (0, 1, 8),    # Pelvis, spine, neck
    ]
    
    angles = []
    
    for a, b, c in joint_triplets:
        if max(a, b, c) >= pose.shape[0]:
            # Skip if any joint index is out of bounds
            continue
            
        # Get vectors
        v1 = pose[a] - pose[b]
        v2 = pose[c] - pose[b]
        
        # Normalize
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            # Skip if vectors are too small
            angles.append(0.0)
            continue
            
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Compute angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    return np.array(angles)


def compute_velocities(sequence: np.ndarray) -> np.ndarray:
    """
    Compute velocities for a pose sequence using finite differences.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Velocities [seq_len-1, num_joints, 3]
    """
    return np.diff(sequence, axis=0)


def compute_accelerations(sequence: np.ndarray) -> np.ndarray:
    """
    Compute accelerations for a pose sequence using finite differences.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Accelerations [seq_len-2, num_joints, 3]
    """
    velocities = compute_velocities(sequence)
    return np.diff(velocities, axis=0)


def compute_jerk(sequence: np.ndarray) -> np.ndarray:
    """
    Compute jerk (rate of change of acceleration) for a pose sequence.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Jerk [seq_len-3, num_joints, 3]
    """
    accelerations = compute_accelerations(sequence)
    return np.diff(accelerations, axis=0)


def compute_speed(sequence: np.ndarray) -> np.ndarray:
    """
    Compute speed (magnitude of velocity) for a pose sequence.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Speed [seq_len-1, num_joints]
    """
    velocities = compute_velocities(sequence)
    return np.linalg.norm(velocities, axis=2)


def compute_joint_trajectories(sequence: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute various trajectory metrics for each joint.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Dictionary with trajectory metrics for each joint
    """
    trajectories = {}
    
    # Compute speed
    if sequence.shape[0] > 1:
        speeds = compute_speed(sequence)
        trajectories['speed'] = speeds
        trajectories['mean_speed'] = np.mean(speeds, axis=0)
        trajectories['max_speed'] = np.max(speeds, axis=0)
    
    # Compute acceleration magnitude
    if sequence.shape[0] > 2:
        accel = compute_accelerations(sequence)
        accel_mag = np.linalg.norm(accel, axis=2)
        trajectories['accel_magnitude'] = accel_mag
        trajectories['mean_accel'] = np.mean(accel_mag, axis=0)
        trajectories['max_accel'] = np.max(accel_mag, axis=0)
    
    # Compute jerk magnitude
    if sequence.shape[0] > 3:
        jerk = compute_jerk(sequence)
        jerk_mag = np.linalg.norm(jerk, axis=2)
        trajectories['jerk_magnitude'] = jerk_mag
        trajectories['mean_jerk'] = np.mean(jerk_mag, axis=0)
        trajectories['max_jerk'] = np.max(jerk_mag, axis=0)
    
    # Compute trajectory length
    if sequence.shape[0] > 1:
        velocities = compute_velocities(sequence)
        trajectory_length = np.sum(np.linalg.norm(velocities, axis=2), axis=0)
        trajectories['trajectory_length'] = trajectory_length
    
    # Compute trajectory straightness (ratio of end-to-end distance to path length)
    if sequence.shape[0] > 1:
        end_to_end = np.linalg.norm(sequence[-1] - sequence[0], axis=1)
        straightness = np.zeros_like(end_to_end)
        nonzero_idx = trajectory_length > 1e-8
        straightness[nonzero_idx] = end_to_end[nonzero_idx] / trajectory_length[nonzero_idx]
        trajectories['straightness'] = straightness
    
    return trajectories


def compute_energy(sequence: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Compute kinetic and potential energy-like metrics for a pose sequence.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        dt: Time step between frames
        
    Returns:
        Dictionary with energy metrics
    """
    energy = {}
    
    # Compute velocities and speeds
    if sequence.shape[0] > 1:
        velocities = compute_velocities(sequence) / dt
        speeds = np.linalg.norm(velocities, axis=2)
        
        # Kinetic energy (proportional to v^2)
        kinetic_energy = 0.5 * speeds**2
        energy['kinetic_energy'] = kinetic_energy
        energy['total_kinetic_energy'] = np.sum(kinetic_energy, axis=1)
        energy['mean_kinetic_energy'] = np.mean(kinetic_energy, axis=1)
    
    # Compute "potential energy" (height above ground)
    # Assuming y-axis is up
    height = sequence[:, :, 1]  # [seq_len, num_joints]
    min_height = np.min(height, axis=1, keepdims=True)
    relative_height = height - min_height
    potential_energy = 9.8 * relative_height  # g * h
    energy['potential_energy'] = potential_energy
    energy['total_potential_energy'] = np.sum(potential_energy, axis=1)
    energy['mean_potential_energy'] = np.mean(potential_energy, axis=1)
    
    # Compute total energy (kinetic + potential)
    if 'total_kinetic_energy' in energy:
        energy['total_energy'] = energy['total_kinetic_energy'] + energy['total_potential_energy']
    
    return energy


def compute_smoothness(sequence: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute smoothness metrics for a pose sequence.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Dictionary with smoothness metrics
    """
    smoothness = {}
    
    # Compute jerk-based smoothness (lower jerk = smoother)
    if sequence.shape[0] > 3:
        jerk = compute_jerk(sequence)
        jerk_mag = np.linalg.norm(jerk, axis=2)
        
        # Dimensionless jerk (normalized by speed and duration)
        if sequence.shape[0] > 1:
            speeds = compute_speed(sequence)
            mean_speed = np.mean(speeds, axis=0)
            duration = sequence.shape[0]
            
            # Avoid division by zero
            mean_speed = np.maximum(mean_speed, 1e-8)
            dimensionless_jerk = jerk_mag / mean_speed[:, np.newaxis] * (duration**2)
            
            smoothness['dimensionless_jerk'] = dimensionless_jerk
            smoothness['mean_dimensionless_jerk'] = np.mean(dimensionless_jerk, axis=0)
    
    # Compute spectral arc length (SPARC) - frequency domain smoothness metric
    if sequence.shape[0] > 8:  # Need enough points for meaningful FFT
        from scipy.fft import fft
        
        # Compute for each joint separately
        sparc = np.zeros(sequence.shape[1])
        
        for joint in range(sequence.shape[1]):
            # Get joint trajectory
            trajectory = sequence[:, joint, :]
            
            # Compute velocity norm
            if sequence.shape[0] > 1:
                velocity = np.diff(trajectory, axis=0)
                speed = np.linalg.norm(velocity, axis=1)
                
                # Compute FFT
                fft_result = fft(speed)
                magnitude = np.abs(fft_result[:len(speed)//2])
                
                # Normalize
                if np.max(magnitude) > 0:
                    normalized = magnitude / np.max(magnitude)
                    
                    # Compute arc length in frequency domain
                    # Only use frequencies up to 10Hz (assuming 30fps)
                    freq_cutoff = min(10, len(normalized))
                    arc_length = 0
                    
                    for i in range(1, freq_cutoff):
                        arc_length += np.sqrt(1 + (normalized[i] - normalized[i-1])**2)
                    
                    sparc[joint] = -arc_length
        
        smoothness['sparc'] = sparc
    
    return smoothness


def compute_periodicity(sequence: np.ndarray) -> Dict[str, Any]:
    """
    Compute periodicity metrics for a pose sequence.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Dictionary with periodicity metrics
    """
    periodicity = {}
    
    # Need enough frames for meaningful periodicity analysis
    if sequence.shape[0] < 60:  # Arbitrary threshold
        return periodicity
    
    # Compute auto-correlation for selected joints
    # Focus on extremities (hands, feet) as they show more periodic behavior
    key_joints = [3, 4, 10, 13]  # Example indices for hands and feet
    
    # Get trajectories for key joints
    trajectories = []
    for joint in key_joints:
        if joint < sequence.shape[1]:
            # Use speed as the signal for autocorrelation
            if sequence.shape[0] > 1:
                velocity = np.diff(sequence[:, joint, :], axis=0)
                speed = np.linalg.norm(velocity, axis=1)
                trajectories.append(speed)
    
    if not trajectories:
        return periodicity
    
    # Combine trajectories
    combined = np.mean(trajectories, axis=0)
    
    # Compute auto-correlation (up to half sequence length)
    max_lag = min(sequence.shape[0] // 2, 100)
    auto_corr = np.zeros(max_lag)
    
    # Normalize signal
    signal = combined - np.mean(combined)
    if np.std(signal) > 0:
        signal = signal / np.std(signal)
    
    # Compute auto-correlation
    for lag in range(1, max_lag):
        corr = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
        auto_corr[lag] = corr
    
    periodicity['auto_correlation'] = auto_corr
    
    # Find peaks in auto-correlation
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(auto_corr, height=0.2, distance=10)
    
    if len(peaks) > 0:
        # Main period is the first significant peak
        main_period = peaks[0]
        periodicity['main_period'] = int(main_period)
        periodicity['main_period_strength'] = float(auto_corr[main_period])
        
        # If multiple peaks, compute regularity
        if len(peaks) > 1:
            peak_periods = np.diff(peaks)
            periodicity['period_regularity'] = float(1.0 - np.std(peak_periods) / np.mean(peak_periods))
            periodicity['all_periods'] = peaks.tolist()
    
    # Spectral analysis for dominant frequencies
    if sequence.shape[0] > 64:  # Power of 2 for efficient FFT
        from scipy.fft import fft, fftfreq
        
        # Compute FFT
        fft_result = fft(signal)
        magnitude = np.abs(fft_result[:len(signal)//2])
        
        # Get frequencies
        freq = fftfreq(len(signal))[:len(signal)//2]
        
        # Find dominant frequencies (excluding DC component)
        if len(magnitude) > 1:
            dominant_idx = np.argsort(magnitude[1:])[-3:] + 1  # Top 3 frequencies
            dominant_freq = freq[dominant_idx]
            dominant_mag = magnitude[dominant_idx]
            
            periodicity['dominant_frequencies'] = dominant_freq.tolist()
            periodicity['dominant_magnitudes'] = dominant_mag.tolist()
    
    return periodicity


def compute_synchronization(sequences: List[np.ndarray]) -> Dict[str, Any]:
    """
    Compute synchronization metrics between multiple dance sequences.
    
    Args:
        sequences: List of pose sequences, each [seq_len, num_joints, 3]
        
    Returns:
        Dictionary with synchronization metrics
    """
    synchronization = {}
    
    if len(sequences) < 2:
        return synchronization
    
    # Ensure all sequences have the same length
    min_length = min(seq.shape[0] for seq in sequences)
    truncated_sequences = [seq[:min_length] for seq in sequences]
    
    # Compute velocities for each sequence
    velocities = []
    for seq in truncated_sequences:
        if seq.shape[0] > 1:
            vel = compute_velocities(seq)
            velocities.append(vel)
    
    if not velocities or len(velocities) < 2:
        return synchronization
    
    # Compute pairwise correlation between sequences
    num_sequences = len(velocities)
    correlation_matrix = np.zeros((num_sequences, num_sequences))
    
    for i in range(num_sequences):
        for j in range(i, num_sequences):
            # Flatten velocities for correlation
            vel_i_flat = velocities[i].reshape(velocities[i].shape[0], -1)
            vel_j_flat = velocities[j].reshape(velocities[j].shape[0], -1)
            
            # Compute correlation for each dimension
            correlations = []
            for dim in range(vel_i_flat.shape[1]):
                if np.std(vel_i_flat[:, dim]) > 0 and np.std(vel_j_flat[:, dim]) > 0:
                    corr = np.corrcoef(vel_i_flat[:, dim], vel_j_flat[:, dim])[0, 1]
                    correlations.append(corr)
            
            # Average correlation across dimensions
            if correlations:
                avg_corr = np.mean(correlations)
                correlation_matrix[i, j] = avg_corr
                correlation_matrix[j, i] = avg_corr
    
    synchronization['correlation_matrix'] = correlation_matrix
    
    # Compute average synchronization across all pairs
    upper_indices = np.triu_indices(num_sequences, k=1)
    synchronization['mean_correlation'] = float(np.mean(correlation_matrix[upper_indices]))
    
    # Compute time lag correlation to find temporal offsets
    if min_length > 20:  # Need enough frames for meaningful cross-correlation
        max_lag = min(20, min_length // 5)  # Limit maximum lag
        lag_correlation = np.zeros((num_sequences, num_sequences, 2*max_lag+1))
        optimal_lags = np.zeros((num_sequences, num_sequences), dtype=int)
        
        for i in range(num_sequences):
            for j in range(i+1, num_sequences):
                # Flatten velocities
                vel_i_flat = velocities[i].reshape(velocities[i].shape[0], -1)
                vel_j_flat = velocities[j].reshape(velocities[j].shape[0], -1)
                
                # Average velocity across all dimensions
                vel_i_avg = np.mean(vel_i_flat, axis=1)
                vel_j_avg = np.mean(vel_j_flat, axis=1)
                
                # Normalize
                vel_i_norm = (vel_i_avg - np.mean(vel_i_avg)) / (np.std(vel_i_avg) + 1e-8)
                vel_j_norm = (vel_j_avg - np.mean(vel_j_avg)) / (np.std(vel_j_avg) + 1e-8)
                
                # Compute cross-correlation for different lags
                for lag in range(-max_lag, max_lag+1):
                    if lag < 0:
                        corr = np.corrcoef(vel_i_norm[:lag], vel_j_norm[-lag:])[0, 1]
                    elif lag > 0:
                        corr = np.corrcoef(vel_i_norm[lag:], vel_j_norm[:-lag])[0, 1]
                    else:
                        corr = np.corrcoef(vel_i_norm, vel_j_norm)[0, 1]
                    
                    lag_correlation[i, j, lag+max_lag] = corr
                    lag_correlation[j, i, lag+max_lag] = corr
                
                # Find optimal lag (maximum correlation)
                opt_lag_idx = np.argmax(lag_correlation[i, j])
                optimal_lags[i, j] = opt_lag_idx - max_lag
                optimal_lags[j, i] = -(opt_lag_idx - max_lag)
        
        synchronization['lag_correlation'] = lag_correlation
        synchronization['optimal_lags'] = optimal_lags
        
        # Compute maximum correlation at optimal lag
        max_correlations = np.zeros((num_sequences, num_sequences))
        for i in range(num_sequences):
            for j in range(i+1, num_sequences):
                max_correlations[i, j] = np.max(lag_correlation[i, j])
                max_correlations[j, i] = max_correlations[i, j]
        
        synchronization['max_correlation'] = max_correlations
        synchronization['mean_max_correlation'] = float(np.mean(max_correlations[upper_indices]))
    
    # Compute phase synchronization using Hilbert transform for key joints
    try:
        from scipy.signal import hilbert
        
        # Select key joints for phase analysis (e.g., extremities)
        key_joints = [3, 4, 10, 13]  # Example indices for hands and feet
        phase_sync = np.zeros((num_sequences, num_sequences))
        
        # For each key joint
        for joint in key_joints:
            if all(joint < seq.shape[1] for seq in truncated_sequences):
                # Get joint trajectories (use y-coordinate as example)
                trajectories = [seq[:, joint, 1] for seq in truncated_sequences]
                
                # Compute phases using Hilbert transform
                phases = []
                for traj in trajectories:
                    # Normalize
                    traj_norm = traj - np.mean(traj)
                    if np.std(traj_norm) > 0:
                        traj_norm = traj_norm / np.std(traj_norm)
                    
                    # Compute analytic signal and phase
                    analytic_signal = hilbert(traj_norm)
                    phase = np.angle(analytic_signal)
                    phases.append(phase)
                
                # Compute phase synchronization between all pairs
                for i in range(num_sequences):
                    for j in range(i+1, num_sequences):
                        # Phase difference
                        phase_diff = phases[i] - phases[j]
                        
                        # Phase Synchronization Index (PSI)
                        psi = np.abs(np.mean(np.exp(1j * phase_diff)))
                        
                        phase_sync[i, j] += psi / len(key_joints)
                        phase_sync[j, i] += psi / len(key_joints)
        
        synchronization['phase_synchronization'] = phase_sync
        synchronization['mean_phase_sync'] = float(np.mean(phase_sync[upper_indices]))
        
    except ImportError:
        # Skip phase synchronization if scipy.signal is not available
        pass
    
    # Compute overall synchronization score (weighted average of different metrics)
    synchronization_score = 0.0
    weight_sum = 0.0
    
    if 'mean_correlation' in synchronization:
        synchronization_score += 0.4 * synchronization['mean_correlation']
        weight_sum += 0.4
    
    if 'mean_max_correlation' in synchronization:
        synchronization_score += 0.4 * synchronization['mean_max_correlation']
        weight_sum += 0.4
    
    if 'mean_phase_sync' in synchronization:
        synchronization_score += 0.2 * synchronization['mean_phase_sync']
        weight_sum += 0.2
    
    if weight_sum > 0:
        synchronization['overall_score'] = float(synchronization_score / weight_sum)
    
    return synchronization


def compute_similarity(sequence1: np.ndarray, sequence2: np.ndarray, weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Compute similarity between two dance sequences using multiple metrics.
    
    Args:
        sequence1: First pose sequence [seq_len1, num_joints, 3]
        sequence2: Second pose sequence [seq_len2, num_joints, 3]
        weights: Optional dictionary with weights for different metrics
        
    Returns:
        Dictionary with similarity metrics
    """
    if weights is None:
        weights = {
            'joint_positions': 0.3,
            'joint_velocities': 0.3,
            'joint_angles': 0.2,
            'energy': 0.1,
            'smoothness': 0.1
        }
    
    similarity = {}
    
    # Ensure sequences have the same number of joints
    if sequence1.shape[1] != sequence2.shape[1]:
        return {'error': 'Sequences have different number of joints'}
    
    # Dynamic Time Warping for sequence alignment
    from scipy.spatial.distance import euclidean
    try:
        from fastdtw import fastdtw
        
        # Reshape sequences for DTW
        seq1_reshaped = sequence1.reshape(sequence1.shape[0], -1)
        seq2_reshaped = sequence2.reshape(sequence2.shape[0], -1)
        
        # Compute DTW distance
        distance, path = fastdtw(seq1_reshaped, seq2_reshaped, dist=euclidean)
        
        # Normalize by sequence lengths
        normalized_distance = distance / (sequence1.shape[0] + sequence2.shape[0])
        similarity['dtw_distance'] = float(normalized_distance)
        
        # Convert to similarity score (0-1)
        similarity['dtw_similarity'] = float(np.exp(-normalized_distance))
        
        # Extract aligned sequences
        aligned_idx1 = [p[0] for p in path]
        aligned_idx2 = [p[1] for p in path]
        
        aligned_seq1 = sequence1[aligned_idx1]
        aligned_seq2 = sequence2[aligned_idx2]
        
    except ImportError:
        # If fastdtw is not available, use simple resampling to make sequences the same length
        from scipy.interpolate import interp1d
        
        # Determine target length (average of both sequences)
        target_length = int((sequence1.shape[0] + sequence2.shape[0]) / 2)
        
        # Resample sequences
        aligned_seq1 = np.zeros((target_length, sequence1.shape[1], sequence1.shape[2]))
        aligned_seq2 = np.zeros((target_length, sequence2.shape[1], sequence2.shape[2]))
        
        for joint in range(sequence1.shape[1]):
            for dim in range(sequence1.shape[2]):
                # Create interpolation functions
                interp_func1 = interp1d(np.linspace(0, 1, sequence1.shape[0]), 
                                       sequence1[:, joint, dim], 
                                       kind='linear')
                interp_func2 = interp1d(np.linspace(0, 1, sequence2.shape[0]), 
                                       sequence2[:, joint, dim], 
                                       kind='linear')
                
                # Resample at target length
                aligned_seq1[:, joint, dim] = interp_func1(np.linspace(0, 1, target_length))
                aligned_seq2[:, joint, dim] = interp_func2(np.linspace(0, 1, target_length))
        
        # Compute simple Euclidean distance
        distance = np.mean(np.linalg.norm(aligned_seq1 - aligned_seq2, axis=2))
        similarity['euclidean_distance'] = float(distance)
        
        # Convert to similarity score (0-1)
        similarity['euclidean_similarity'] = float(np.exp(-distance))
    
    # Compute position-based similarity
    position_diff = np.mean(np.linalg.norm(aligned_seq1 - aligned_seq2, axis=2))
    position_similarity = np.exp(-position_diff)
    similarity['position_similarity'] = float(position_similarity)
    
    # Compute velocity-based similarity
    if aligned_seq1.shape[0] > 1 and aligned_seq2.shape[0] > 1:
        vel1 = np.diff(aligned_seq1, axis=0)
        vel2 = np.diff(aligned_seq2, axis=0)
        
        vel_diff = np.mean(np.linalg.norm(vel1 - vel2, axis=2))
        vel_similarity = np.exp(-vel_diff)
        similarity['velocity_similarity'] = float(vel_similarity)
    
    # Compute angle-based similarity
    angles1 = np.array([compute_joint_angles(aligned_seq1[i]) for i in range(aligned_seq1.shape[0])])
    angles2 = np.array([compute_joint_angles(aligned_seq2[i]) for i in range(aligned_seq2.shape[0])])
    
    if angles1.size > 0 and angles2.size > 0 and angles1.shape == angles2.shape:
        angle_diff = np.mean(np.abs(angles1 - angles2))
        angle_similarity = np.exp(-angle_diff)
        similarity['angle_similarity'] = float(angle_similarity)
    
    # Compute energy profile similarity
    energy1 = compute_energy(aligned_seq1)
    energy2 = compute_energy(aligned_seq2)
    
    if 'total_energy' in energy1 and 'total_energy' in energy2:
        # Normalize energies
        norm_energy1 = energy1['total_energy'] / np.max(energy1['total_energy'])
        norm_energy2 = energy2['total_energy'] / np.max(energy2['total_energy'])
        
        # Compute correlation
        energy_corr = np.corrcoef(norm_energy1, norm_energy2)[0, 1]
        similarity['energy_correlation'] = float(energy_corr)
    
    # Compute smoothness similarity
    smoothness1 = compute_smoothness(aligned_seq1)
    smoothness2 = compute_smoothness(aligned_seq2)
    
    if 'mean_dimensionless_jerk' in smoothness1 and 'mean_dimensionless_jerk' in smoothness2:
        jerk_diff = np.mean(np.abs(smoothness1['mean_dimensionless_jerk'] - 
                                  smoothness2['mean_dimensionless_jerk']))
        smoothness_similarity = np.exp(-jerk_diff)
        similarity['smoothness_similarity'] = float(smoothness_similarity)
    
    # Compute overall similarity score (weighted average)
    overall_score = 0.0
    weight_sum = 0.0
    
    metric_mapping = {
        'joint_positions': 'position_similarity',
        'joint_velocities': 'velocity_similarity',
        'joint_angles': 'angle_similarity',
        'energy': 'energy_correlation',
        'smoothness': 'smoothness_similarity'
    }
    
    for metric, weight in weights.items():
        if metric_mapping[metric] in similarity:
            overall_score += weight * similarity[metric_mapping[metric]]
            weight_sum += weight
    
    if weight_sum > 0:
        similarity['overall_similarity'] = float(overall_score / weight_sum)
    
    return similarity


def compute_complexity(sequence: np.ndarray) -> Dict[str, float]:
    """
    Compute complexity metrics for a dance sequence.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Dictionary with complexity metrics
    """
    complexity = {}
    
    # Need enough frames for meaningful analysis
    if sequence.shape[0] < 10:
        return complexity
    
    # Compute velocity and acceleration
    if sequence.shape[0] > 2:
        velocities = compute_velocities(sequence)
        accelerations = compute_accelerations(sequence)
        
        # Velocity variation (standard deviation of speed)
        speeds = np.linalg.norm(velocities, axis=2)
        complexity['velocity_variation'] = float(np.mean(np.std(speeds, axis=0)))
        
        # Spatial complexity (average distance traveled by all joints)
        complexity['spatial_complexity'] = float(np.sum(speeds) / (sequence.shape[0] * sequence.shape[1]))
        
        # Temporal complexity (variation in acceleration)
        accel_mag = np.linalg.norm(accelerations, axis=2)
        complexity['temporal_complexity'] = float(np.mean(np.std(accel_mag, axis=0)))
        
        # Direction changes (using velocity vector angles)
        direction_changes = 0
        for joint in range(velocities.shape[1]):
            for t in range(1, velocities.shape[0]):
                v1 = velocities[t-1, joint]
                v2 = velocities[t, joint]
                
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 1e-5 and v2_norm > 1e-5:
                    cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    # Count significant direction changes (> 45 degrees)
                    if angle > np.pi/4:
                        direction_changes += 1
        
        complexity['direction_changes'] = float(direction_changes / (velocities.shape[0] * velocities.shape[1]))
    
    # Compute dimensionality using PCA
    try:
        from sklearn.decomposition import PCA
        
        # Reshape sequence for PCA
        reshaped_seq = sequence.reshape(sequence.shape[0], -1)
        
        # Apply PCA
        pca = PCA()
        pca.fit(reshaped_seq)
        
        # Compute effective dimensionality (number of components needed to explain 95% variance)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        complexity['effective_dimensionality'] = float(n_components)
        
        # Compute entropy of PCA eigenvalues as another measure of complexity
        eigenvalues = pca.explained_variance_
        normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)
        entropy = -np.sum(normalized_eigenvalues * np.log2(normalized_eigenvalues + 1e-10))
        
        complexity['eigenvalue_entropy'] = float(entropy)
        
    except ImportError:
        # Skip PCA-based metrics if sklearn is not available
        pass
    
    # Sample Entropy as a measure of movement predictability
    try:
        from nolds import sampen
        
        # Compute Sample Entropy for key joints
        key_joints = [3, 4, 10, 13]  # Example indices for hands and feet
        entropy_values = []
        
        for joint in key_joints:
            if joint < sequence.shape[1]:
                # Use position of joint as signal
                for dim in range(sequence.shape[2]):
                    signal = sequence[:, joint, dim]
                    
                    # Normalize signal
                    if np.std(signal) > 0:
                        signal = (signal - np.mean(signal)) / np.std(signal)
                        
                        # Compute Sample Entropy (m=2, r=0.2)
                        try:
                            entropy = sampen(signal, emb_dim=2, tolerance=0.2 * np.std(signal))
                            entropy_values.append(entropy)
                        except:
                            # Skip if entropy calculation fails
                            pass
        
        if entropy_values:
            complexity['sample_entropy'] = float(np.mean(entropy_values))
            
    except ImportError:
        # Skip Sample Entropy if nolds is not available
        pass
    
    # Compute overall complexity score (weighted average of metrics)
    complexity_score = 0.0
    weight_sum = 0.0
    
    # Define weights for each metric
    weights = {
        'velocity_variation': 0.2,
        'spatial_complexity': 0.2,
        'temporal_complexity': 0.2,
        'direction_changes': 0.2,
        'effective_dimensionality': 0.1,
        'eigenvalue_entropy': 0.05,
        'sample_entropy': 0.05
    }
    
    for metric, weight in weights.items():
        if metric in complexity:
            # Normalize metric to 0-1 range (using empirical thresholds)
            if metric == 'velocity_variation':
                normalized = min(1.0, complexity[metric] / 5.0)
            elif metric == 'spatial_complexity':
                normalized = min(1.0, complexity[metric] / 10.0)
            elif metric == 'temporal_complexity':
                normalized = min(1.0, complexity[metric] / 5.0)
            elif metric == 'direction_changes':
                normalized = min(1.0, complexity[metric] / 0.5)
            elif metric == 'effective_dimensionality':
                normalized = min(1.0, complexity[metric] / (sequence.shape[1] * sequence.shape[2]))
            elif metric == 'eigenvalue_entropy':
                normalized = min(1.0, complexity[metric] / 5.0)
            elif metric == 'sample_entropy':
                normalized = min(1.0, complexity[metric] / 2.0)
            else:
                normalized = 0.5  # Default value
            
            complexity_score += weight * normalized
            weight_sum += weight
    
    if weight_sum > 0:
        complexity['overall_complexity'] = float(complexity_score / weight_sum)
    
    return complexity


def compute_style_metrics(sequence: np.ndarray) -> Dict[str, Any]:
    """
    Compute style-related metrics for a dance sequence.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Dictionary with style metrics
    """
    style = {}
    
    # Need enough frames for meaningful analysis
    if sequence.shape[0] < 20:
        return style
    
    # Compute basic statistics
    if sequence.shape[0] > 1:
        # Compute velocities and accelerations
        velocities = compute_velocities(sequence)
        accelerations = compute_accelerations(sequence) if sequence.shape[0] > 2 else None
        
        # Speed statistics
        speeds = np.linalg.norm(velocities, axis=2)
        style['mean_speed'] = float(np.mean(speeds))
        style['std_speed'] = float(np.std(speeds))
        style['max_speed'] = float(np.max(speeds))
        
        # Acceleration statistics if available
        if accelerations is not None:
            accel_mag = np.linalg.norm(accelerations, axis=2)
            style['mean_acceleration'] = float(np.mean(accel_mag))
            style['std_acceleration'] = float(np.std(accel_mag))
            style['max_acceleration'] = float(np.max(accel_mag))
        
        # Movement range (space utilization)
        pos_range = np.max(sequence, axis=0) - np.min(sequence, axis=0)
        style['movement_range'] = float(np.mean(pos_range))
        
        # Center of mass (COM) movement
        com = np.mean(sequence, axis=1)  # [seq_len, 3]
        com_velocity = np.diff(com, axis=0)
        com_speed = np.linalg.norm(com_velocity, axis=1)
        style['com_mean_speed'] = float(np.mean(com_speed))
        style['com_max_speed'] = float(np.max(com_speed))
        
        # Movement symmetry
        left_joints = [5, 6, 7, 12, 13, 14]  # Example left side joints
        right_joints = [2, 3, 4, 9, 10, 11]  # Example right side joints
        
        # Ensure indices are within bounds
        left_joints = [j for j in left_joints if j < sequence.shape[1]]
        right_joints = [j for j in right_joints if j < sequence.shape[1]]
        
        if len(left_joints) == len(right_joints) and len(left_joints) > 0:
            left_avg_speed = np.mean(speeds[:, left_joints])
            right_avg_speed = np.mean(speeds[:, right_joints])
            
            # Symmetry ratio (1.0 = perfect symmetry)
            if left_avg_speed > 0 and right_avg_speed > 0:
                symmetry_ratio = min(left_avg_speed, right_avg_speed) / max(left_avg_speed, right_avg_speed)
                style['movement_symmetry'] = float(symmetry_ratio)
        
        # Periodicity-based style features
        periodicity = compute_periodicity(sequence)
        if 'main_period' in periodicity:
            style['main_period'] = periodicity['main_period']
            style['period_strength'] = periodicity['main_period_strength']
        
        if 'period_regularity' in periodicity:
            style['movement_regularity'] = periodicity['period_regularity']
        
        # Smoothness-based style features
        smoothness = compute_smoothness(sequence)
        if 'mean_dimensionless_jerk' in smoothness:
            # Invert jerk to get smoothness (higher = smoother)
            style['movement_smoothness'] = float(1.0 / (1.0 + np.mean(smoothness['mean_dimensionless_jerk'])))
        
        # Joint correlations (coordination patterns)
        joint_correlations = np.zeros((sequence.shape[1], sequence.shape[1]))
        
        for i in range(sequence.shape[1]):
            for j in range(i+1, sequence.shape[1]):
                # Compute correlation of joint speeds
                if i < speeds.shape[1] and j < speeds.shape[1]:
                    correlation = np.corrcoef(speeds[:, i], speeds[:, j])[0, 1]
                    joint_correlations[i, j] = correlation
                    joint_correlations[j, i] = correlation
        
        style['joint_correlations'] = joint_correlations
        style['mean_joint_correlation'] = float(np.mean(joint_correlations[np.triu_indices(joint_correlations.shape[0], k=1)]))
        
        # Compute upper body vs lower body activity ratio
        upper_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Example upper body joints
        lower_joints = [9, 10, 11, 12, 13, 14]  # Example lower body joints
        
        # Ensure indices are within bounds
        upper_joints = [j for j in upper_joints if j < speeds.shape[1]]
        lower_joints = [j for j in lower_joints if j < speeds.shape[1]]
        
        if upper_joints and lower_joints:
            upper_activity = np.mean(speeds[:, upper_joints])
            lower_activity = np.mean(speeds[:, lower_joints])
            
            if lower_activity > 0:
                style['upper_lower_ratio'] = float(upper_activity / lower_activity)
    
    # Compute dynamic signature
    # This is a simplified version that captures speed and acceleration patterns
    num_bins = 10
    if sequence.shape[0] > 1:
        # Create speed histogram
        speed_range = [0, np.max(speeds) + 1e-6]
        speed_hist, _ = np.histogram(speeds, bins=num_bins, range=speed_range)
        speed_hist = speed_hist / np.sum(speed_hist)
        style['speed_distribution'] = speed_hist.tolist()
        
        # Create acceleration histogram if available
        if accelerations is not None:
            accel_mag = np.linalg.norm(accelerations, axis=2)
            accel_range = [0, np.max(accel_mag) + 1e-6]
            accel_hist, _ = np.histogram(accel_mag, bins=num_bins, range=accel_range)
            accel_hist = accel_hist / np.sum(accel_hist)
            style['acceleration_distribution'] = accel_hist.tolist()
    
    # Compute style-based classification features
    # These features can be used for style classification (e.g., dance genre)
    style_features = []
    
    if 'mean_speed' in style:
        style_features.append(style['mean_speed'])
    
    if 'std_speed' in style:
        style_features.append(style['std_speed'])
    
    if 'movement_range' in style:
        style_features.append(style['movement_range'])
    
    if 'movement_symmetry' in style:
        style_features.append(style['movement_symmetry'])
    
    if 'movement_smoothness' in style:
        style_features.append(style['movement_smoothness'])
    
    if 'mean_joint_correlation' in style:
        style_features.append(style['mean_joint_correlation'])
    
    if 'upper_lower_ratio' in style:
        style_features.append(style['upper_lower_ratio'])
    
    if style_features:
        style['style_feature_vector'] = np.array(style_features).tolist()
    
    return style


def compute_music_dance_alignment(dance_sequence: np.ndarray, music_features: np.ndarray, 
                               fps: float = 30.0, music_sr: float = 44100.0) -> Dict[str, Any]:
    """
    Compute alignment between dance movements and music features.
    
    Args:
        dance_sequence: Dance pose sequence [seq_len, num_joints, 3]
        music_features: Music features array [num_frames, num_features]
        fps: Frames per second of dance sequence
        music_sr: Sample rate of music features (or feature frame rate)
        
    Returns:
        Dictionary with alignment metrics
    """
    alignment = {}
    
    # Ensure both sequences have data
    if dance_sequence.shape[0] < 5 or music_features.shape[0] < 5:
        return alignment
    
    # Compute dance energy profile
    if dance_sequence.shape[0] > 1:
        # Compute velocities and speed
        velocities = compute_velocities(dance_sequence)
        speeds = np.linalg.norm(velocities, axis=2)
        
        # Compute kinetic energy for each frame
        dance_energy = np.sum(speeds**2, axis=1)  # Sum over all joints
        
        # Normalize energy
        if np.max(dance_energy) > 0:
            dance_energy = dance_energy / np.max(dance_energy)
    else:
        return alignment
    
    # If music features are multi-dimensional, extract relevant ones
    # Assuming first column might be onset strength or energy
    music_energy = music_features[:, 0] if music_features.shape[1] > 1 else music_features
    
    # Normalize music energy
    if np.max(music_energy) > 0:
        music_energy = music_energy / np.max(music_energy)
    
    # Resample to match lengths if needed
    from scipy.interpolate import interp1d
    
    # Create time arrays for both sequences
    dance_time = np.arange(dance_energy.shape[0]) / fps
    music_time = np.arange(music_energy.shape[0]) / music_sr
    
    # Determine common time range
    common_duration = min(dance_time[-1], music_time[-1])
    
    # Create resampling functions
    dance_interp = interp1d(dance_time, dance_energy, bounds_error=False, fill_value='extrapolate')
    music_interp = interp1d(music_time, music_energy, bounds_error=False, fill_value='extrapolate')
    
    # Create common time base (use dance fps for simplicity)
    common_time = np.arange(0, common_duration, 1/fps)
    
    # Resample both signals to common time base
    dance_resampled = dance_interp(common_time)
    music_resampled = music_interp(common_time)
    
    # Compute correlation
    correlation = np.corrcoef(dance_resampled, music_resampled)[0, 1]
    alignment['energy_correlation'] = float(correlation)
    
    # Compute time lag correlation to find temporal offsets
    max_lag_seconds = 2.0  # Maximum 2 second lag
    max_lag_frames = int(max_lag_seconds * fps)
    max_lag = min(max_lag_frames, len(common_time) // 4)  # Limit maximum lag
    
    lag_correlation = np.zeros(2*max_lag+1)
    
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            corr = np.corrcoef(dance_resampled[:lag], music_resampled[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(dance_resampled[lag:], music_resampled[:-lag])[0, 1]
        else:
            corr = correlation
        
        lag_correlation[lag+max_lag] = corr
    
    alignment['lag_correlation'] = lag_correlation.tolist()
    
    # Find optimal lag (maximum correlation)
    opt_lag_idx = np.argmax(lag_correlation)
    opt_lag_frames = opt_lag_idx - max_lag
    opt_lag_seconds = opt_lag_frames / fps
    
    alignment['optimal_lag_frames'] = int(opt_lag_frames)
    alignment['optimal_lag_seconds'] = float(opt_lag_seconds)
    alignment['max_correlation'] = float(lag_correlation[opt_lag_idx])
    
    # Compute beat alignment if music features include beat information
    if music_features.shape[1] > 1:
        # Assume second column might contain beat information (0 or 1)
        beat_info = music_features[:, 1] if music_features.shape[1] > 1 else None
        
        if beat_info is not None and np.max(beat_info) > 0:
            # Resample beat info to match common time base
            beat_time = np.arange(beat_info.shape[0]) / music_sr
            beat_interp = interp1d(beat_time, beat_info, bounds_error=False, fill_value=0)
            beats_resampled = beat_interp(common_time)
            
            # Threshold to get binary beats
            beats_binary = beats_resampled > 0.5
            
            # Find frames with significant movement changes
            movement_changes = np.zeros_like(dance_resampled, dtype=bool)
            
            # Compute acceleration magnitude
            if len(dance_resampled) > 2:
                accel = np.diff(np.diff(dance_resampled))
                accel_padded = np.pad(accel, (1, 1), 'constant')
                
                # Threshold acceleration to detect significant changes
                threshold = np.std(accel) * 1.5
                movement_changes[1:-1] = np.abs(accel) > threshold
            
            # Compute beat alignment
            # For each beat, check if there's a movement change within a small window
            window_size = int(0.1 * fps)  # 100ms window
            
            beat_indices = np.where(beats_binary)[0]
            aligned_beats = 0
            
            for beat_idx in beat_indices:
                start_idx = max(0, beat_idx - window_size)
                end_idx = min(len(movement_changes), beat_idx + window_size + 1)
                
                if np.any(movement_changes[start_idx:end_idx]):
                    aligned_beats += 1
            
            if len(beat_indices) > 0:
                beat_alignment_score = aligned_beats / len(beat_indices)
                alignment['beat_alignment_score'] = float(beat_alignment_score)
    
    # Calculate phase synchronization using Hilbert transform
    try:
        from scipy.signal import hilbert
        
        # Apply Hilbert transform to get analytic signals
        analytic_dance = hilbert(dance_resampled - np.mean(dance_resampled))
        analytic_music = hilbert(music_resampled - np.mean(music_resampled))
        
        # Extract instantaneous phases
        dance_phase = np.angle(analytic_dance)
        music_phase = np.angle(analytic_music)
        
        # Phase difference
        phase_diff = dance_phase - music_phase
        
        # Phase Synchronization Index (PSI)
        psi = np.abs(np.mean(np.exp(1j * phase_diff)))
        alignment['phase_synchronization'] = float(psi)
        
    except ImportError:
        # Skip phase synchronization if scipy.signal is not available
        pass
    
    # Compute overall alignment score (weighted average of different metrics)
    alignment_score = 0.0
    weight_sum = 0.0
    
    if 'max_correlation' in alignment:
        alignment_score += 0.4 * alignment['max_correlation']
        weight_sum += 0.4
    
    if 'beat_alignment_score' in alignment:
        alignment_score += 0.4 * alignment['beat_alignment_score']
        weight_sum += 0.4
    
    if 'phase_synchronization' in alignment:
        alignment_score += 0.2 * alignment['phase_synchronization']
        weight_sum += 0.2
    
    if weight_sum > 0:
        alignment['overall_score'] = float(alignment_score / weight_sum)
    
    return alignment
