"""
3D-DanceDTW Algorithm Implementation

This module implements the 3D-DanceDTW (Dynamic Time Warping) algorithm for 
calculating the similarity between two 3D dance motion sequences.

The algorithm extends traditional DTW by incorporating:
1. 3D joint angles, velocities, and accelerations
2. Multi-scale analysis through wavelet transform
3. Adaptive weighting of different motion features

Author: JJshome
Date: May 13, 2025
"""

import numpy as np
from scipy.signal import wavelets
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union


class FeatureExtractor:
    """
    Extracts relevant features from 3D pose sequences for similarity analysis.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the feature extractor with configuration parameters.
        
        Args:
            config: Configuration dictionary with parameters for feature extraction
        """
        self.config = config or {}
        self.joint_angle_weight = self.config.get('joint_angle_weight', 0.4)
        self.velocity_weight = self.config.get('velocity_weight', 0.3)
        self.acceleration_weight = self.config.get('acceleration_weight', 0.2)
        self.position_weight = self.config.get('position_weight', 0.1)
    
    def compute_joint_angles(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Compute joint angles from 3D pose sequence.
        
        Args:
            pose_sequence: 3D pose sequence with shape (frames, joints, 3)
            
        Returns:
            Joint angles with shape (frames, n_angles)
        """
        # Number of frames and joints
        n_frames, n_joints, _ = pose_sequence.shape
        
        # Define connections between joints (parent-child relationships)
        # Example: [(0,1), (1,2), ...] where (parent, child)
        connections = self._get_skeleton_connections()
        n_angles = len(connections)
        
        # Initialize joint angles array
        joint_angles = np.zeros((n_frames, n_angles))
        
        # For each frame, compute the angles between connected joints
        for t in range(n_frames):
            for i, (parent, child) in enumerate(connections):
                # Vector from parent to child
                vector = pose_sequence[t, child] - pose_sequence[t, parent]
                
                # Normalize
                vector_norm = np.linalg.norm(vector)
                if vector_norm > 1e-6:  # Avoid division by zero
                    vector = vector / vector_norm
                
                # Compute angle with reference vector (e.g., y-axis [0, 1, 0])
                # This is simplified; in practice, you'd compute meaningful angles
                reference = np.array([0, 1, 0])
                cos_angle = np.dot(vector, reference)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                
                joint_angles[t, i] = angle
        
        return joint_angles
    
    def compute_velocities(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Compute velocities from 3D pose sequence.
        
        Args:
            pose_sequence: 3D pose sequence with shape (frames, joints, 3)
            
        Returns:
            Velocities with shape (frames-1, joints, 3)
        """
        # Simple velocity computation: displacement between consecutive frames
        velocities = np.diff(pose_sequence, axis=0)
        return velocities
    
    def compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """
        Compute accelerations from velocities.
        
        Args:
            velocities: Velocity data with shape (frames-1, joints, 3)
            
        Returns:
            Accelerations with shape (frames-2, joints, 3)
        """
        # Simple acceleration computation: change in velocity between consecutive frames
        accelerations = np.diff(velocities, axis=0)
        return accelerations
    
    def extract_features(self, pose_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all relevant features from a 3D pose sequence.
        
        Args:
            pose_sequence: 3D pose sequence with shape (frames, joints, 3)
            
        Returns:
            Dictionary containing extracted features
        """
        # Compute individual features
        joint_angles = self.compute_joint_angles(pose_sequence)
        velocities = self.compute_velocities(pose_sequence)
        accelerations = self.compute_accelerations(velocities)
        
        # Ensure all features have the same number of frames by padding or truncating
        min_frames = min(joint_angles.shape[0], velocities.shape[0], accelerations.shape[0])
        
        # Return features
        return {
            'joint_angles': joint_angles[:min_frames],
            'velocities': velocities[:min_frames],
            'accelerations': accelerations[:min_frames],
            'positions': pose_sequence[:min_frames]
        }
    
    def _get_skeleton_connections(self) -> List[Tuple[int, int]]:
        """
        Get the connections between joints in the skeleton.
        
        Returns:
            List of tuples, each containing (parent_joint_idx, child_joint_idx)
        """
        # Example skeleton connections based on a standard 25-joint skeleton
        # This should be adjusted based on your specific skeleton model
        return [
            (0, 1),   # Spine to neck
            (1, 2),   # Neck to head
            (1, 3),   # Neck to right shoulder
            (3, 4),   # Right shoulder to right elbow
            (4, 5),   # Right elbow to right wrist
            (1, 6),   # Neck to left shoulder
            (6, 7),   # Left shoulder to left elbow
            (7, 8),   # Left elbow to left wrist
            (0, 9),   # Spine to right hip
            (9, 10),  # Right hip to right knee
            (10, 11), # Right knee to right ankle
            (0, 12),  # Spine to left hip
            (12, 13), # Left hip to left knee
            (13, 14)  # Left knee to left ankle
        ]


class WaveletTransformer:
    """
    Applies wavelet transform to extract multi-scale representations of motion data.
    """
    
    def __init__(self, wavelet: str = 'db4', levels: int = 3):
        """
        Initialize the wavelet transformer.
        
        Args:
            wavelet: Wavelet type to use ('db4', 'haar', etc.)
            levels: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.levels = levels
    
    def transform(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Apply wavelet transform to data.
        
        Args:
            data: Input data array with shape (frames, features)
            
        Returns:
            List of transformed data at different scales
        """
        # Initialize list to store multi-scale representations
        multi_scale_data = [data]  # Original data (scale 1)
        
        # Apply wavelet transform for each level
        for i in range(self.levels - 1):
            # Get approximation coefficients for current level
            approx_coef = self._wavelet_approx(multi_scale_data[i])
            
            # Add to multi-scale data
            multi_scale_data.append(approx_coef)
        
        return multi_scale_data
    
    def _wavelet_approx(self, data: np.ndarray) -> np.ndarray:
        """
        Get the approximation coefficients from wavelet transform.
        
        Args:
            data: Input data array
            
        Returns:
            Approximation coefficients
        """
        # This is a simplified implementation
        # For each feature dimension, get approximation coefficients
        result = np.zeros((data.shape[0] // 2, data.shape[1]))
        
        for i in range(data.shape[1]):
            # Simple approximation: average of consecutive pairs
            result[:, i] = (data[0::2, i] + data[1::2, i]) / 2
        
        return result


class AdaptiveWeightModel(nn.Module):
    """
    Neural network model to learn adaptive weights for different features.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize the adaptive weight model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
        """
        super(AdaptiveWeightModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 4)  # 4 weights for joint angles, velocities, accelerations, positions
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input features
            
        Returns:
            Adaptive weights for different features
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        
        return x


class DanceDTW:
    """
    3D-DanceDTW algorithm for computing similarity between dance motion sequences.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the DanceDTW algorithm.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(self.config.get('feature_extractor', {}))
        self.wavelet_transformer = WaveletTransformer(
            wavelet=self.config.get('wavelet', 'db4'),
            levels=self.config.get('wavelet_levels', 3)
        )
        
        # Initialize adaptive weight model if using learned weights
        use_learned_weights = self.config.get('use_learned_weights', False)
        if use_learned_weights:
            # Define input dimension based on feature representation
            input_dim = self.config.get('weight_model_input_dim', 128)
            self.weight_model = AdaptiveWeightModel(input_dim)
            
            # Load pre-trained weights if available
            weight_path = self.config.get('weight_model_path', None)
            if weight_path:
                self.weight_model.load_state_dict(torch.load(weight_path))
            
            self.weight_model.eval()
        
        # Default weights for each scale
        self.scale_weights = self.config.get('scale_weights', [0.6, 0.3, 0.1])
    
    def compute_feature_distance(self, features1: Dict[str, np.ndarray], features2: Dict[str, np.ndarray], 
                                t1: int, t2: int) -> float:
        """
        Compute the distance between two feature sets at specific time points.
        
        Args:
            features1: Features of the first sequence
            features2: Features of the second sequence
            t1: Time index in the first sequence
            t2: Time index in the second sequence
            
        Returns:
            Distance value
        """
        # Calculate distances for each feature type
        angle_dist = np.mean((features1['joint_angles'][t1] - features2['joint_angles'][t2]) ** 2)
        vel_dist = np.mean((features1['velocities'][t1] - features2['velocities'][t2]) ** 2)
        acc_dist = np.mean((features1['accelerations'][t1] - features2['accelerations'][t2]) ** 2)
        pos_dist = np.mean((features1['positions'][t1] - features2['positions'][t2]) ** 2)
        
        # Apply weights
        total_dist = (
            self.feature_extractor.joint_angle_weight * angle_dist +
            self.feature_extractor.velocity_weight * vel_dist +
            self.feature_extractor.acceleration_weight * acc_dist +
            self.feature_extractor.position_weight * pos_dist
        )
        
        return total_dist
    
    def dtw(self, features1: Dict[str, np.ndarray], features2: Dict[str, np.ndarray]) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Perform DTW algorithm on two feature sets.
        
        Args:
            features1: Features of the first sequence
            features2: Features of the second sequence
            
        Returns:
            Tuple containing (DTW distance, optimal warping path)
        """
        # Get sequence lengths
        n = len(features1['joint_angles'])
        m = len(features2['joint_angles'])
        
        # Initialize DTW matrix
        dtw_matrix = np.zeros((n + 1, m + 1))
        dtw_matrix[0, 1:] = np.inf
        dtw_matrix[1:, 0] = np.inf
        
        # Compute DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self.compute_feature_distance(features1, features2, i - 1, j - 1)
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],     # Insertion
                    dtw_matrix[i, j - 1],     # Deletion
                    dtw_matrix[i - 1, j - 1]  # Match
                )
        
        # Retrieve the optimal warping path
        path = []
        i, j = n, m
        
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            
            # Find the next move (insertion, deletion, or match)
            min_val = min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
            
            if min_val == dtw_matrix[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif min_val == dtw_matrix[i - 1, j]:
                i -= 1
            else:
                j -= 1
        
        # Reverse path to get it in the right order
        path.reverse()
        
        # Return DTW distance and path
        return dtw_matrix[n, m], path
    
    def multi_scale_dtw(self, pose_seq1: np.ndarray, pose_seq2: np.ndarray) -> Tuple[float, List[List[Tuple[int, int]]]]:
        """
        Perform multi-scale DTW on two pose sequences.
        
        Args:
            pose_seq1: First 3D pose sequence with shape (frames, joints, 3)
            pose_seq2: Second 3D pose sequence with shape (frames, joints, 3)
            
        Returns:
            Tuple containing (final similarity score, list of warping paths at each scale)
        """
        # Extract features
        features1 = self.feature_extractor.extract_features(pose_seq1)
        features2 = self.feature_extractor.extract_features(pose_seq2)
        
        # Apply wavelet transform to get multi-scale representations
        multi_scale_features1 = []
        multi_scale_features2 = []
        
        for feature_name in features1.keys():
            scale_data1 = self.wavelet_transformer.transform(features1[feature_name])
            scale_data2 = self.wavelet_transformer.transform(features2[feature_name])
            
            multi_scale_features1.append(scale_data1)
            multi_scale_features2.append(scale_data2)
        
        # Perform DTW at each scale
        dtw_distances = []
        warping_paths = []
        
        for scale in range(self.wavelet_transformer.levels):
            # Construct features at this scale
            scale_features1 = {
                feature_name: multi_scale_features1[i][scale] 
                for i, feature_name in enumerate(features1.keys())
            }
            
            scale_features2 = {
                feature_name: multi_scale_features2[i][scale] 
                for i, feature_name in enumerate(features2.keys())
            }
            
            # Apply DTW
            dist, path = self.dtw(scale_features1, scale_features2)
            
            dtw_distances.append(dist)
            warping_paths.append(path)
        
        # Compute final similarity score
        # Normalize distances
        max_dist = max(dtw_distances)
        if max_dist > 0:
            normalized_distances = [d / max_dist for d in dtw_distances]
        else:
            normalized_distances = dtw_distances
        
        # Apply scale weights and compute final score
        similarity_score = sum(w * (1 - d) for w, d in zip(self.scale_weights, normalized_distances))
        
        return similarity_score, warping_paths
    
    def compute_similarity(self, pose_seq1: np.ndarray, pose_seq2: np.ndarray) -> Dict:
        """
        Compute similarity between two pose sequences.
        
        Args:
            pose_seq1: First 3D pose sequence with shape (frames, joints, 3)
            pose_seq2: Second 3D pose sequence with shape (frames, joints, 3)
            
        Returns:
            Dictionary containing similarity results
        """
        # Perform multi-scale DTW
        similarity_score, warping_paths = self.multi_scale_dtw(pose_seq1, pose_seq2)
        
        # Prepare results
        results = {
            'similarity_score': similarity_score,
            'warping_paths': warping_paths,
            'is_similar': similarity_score > self.config.get('similarity_threshold', 0.7)
        }
        
        return results


if __name__ == "__main__":
    # Example usage
    # Generate random pose sequences for demonstration
    n_frames1, n_frames2 = 100, 120
    n_joints = 25
    
    # Create random pose sequences
    pose_seq1 = np.random.rand(n_frames1, n_joints, 3)
    pose_seq2 = np.random.rand(n_frames2, n_joints, 3)
    
    # Initialize DanceDTW
    dance_dtw = DanceDTW()
    
    # Compute similarity
    results = dance_dtw.compute_similarity(pose_seq1, pose_seq2)
    
    # Print results
    print(f"Similarity Score: {results['similarity_score']:.4f}")
    print(f"Is Similar: {results['is_similar']}")
    print(f"Number of Warping Paths: {len(results['warping_paths'])}")
