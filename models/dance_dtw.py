"""
3D-DanceDTW: Advanced similarity analysis for dance choreographies.

This module implements the 3D-DanceDTW algorithm for comparing dance 
sequences in 3D space, using multi-scale DTW and adaptive weighting
to properly evaluate choreography similarity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import pywt  # PyWavelets for wavelet transforms


class PoseFeatureExtractor(nn.Module):
    """
    Extracts meaningful features from 3D pose sequences for similarity analysis.
    """
    def __init__(
        self, 
        input_dim: int = 3,
        feature_dim: int = 128,
        num_joints: int = 25,
        joint_weights: Optional[torch.Tensor] = None,
        normalize: bool = True,
        compute_velocity: bool = True,
        compute_acceleration: bool = True,
        compute_jerk: bool = False,
        compute_angles: bool = True,
    ):
        """
        Initialize PoseFeatureExtractor.
        
        Args:
            input_dim: Input dimension per joint (typically 3 for x,y,z)
            feature_dim: Output feature dimension
            num_joints: Number of joints in the skeleton
            joint_weights: Optional weights for each joint
            normalize: Whether to normalize poses
            compute_velocity: Whether to compute velocity features
            compute_acceleration: Whether to compute acceleration features
            compute_jerk: Whether to compute jerk features
            compute_angles: Whether to compute joint angle features
        """
        super(PoseFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_joints = num_joints
        self.normalize = normalize
        
        # Set up computation flags
        self.compute_velocity = compute_velocity
        self.compute_acceleration = compute_acceleration
        self.compute_jerk = compute_jerk
        self.compute_angles = compute_angles
        
        # Calculate input feature dimension
        self.computed_feat_dim = input_dim  # Position
        if compute_velocity:
            self.computed_feat_dim += input_dim
        if compute_acceleration:
            self.computed_feat_dim += input_dim
        if compute_jerk:
            self.computed_feat_dim += input_dim
        if compute_angles:
            self.computed_feat_dim += 3  # Joint angles (3 angles per joint)
        
        # Register joint weights or create default ones
        if joint_weights is None:
            # Default: all joints have equal importance
            joint_weights = torch.ones(num_joints)
        self.register_buffer('joint_weights', joint_weights)
        
        # Define joint connections for angle computation
        # This is a simplified example - actual implementation would use
        # a proper skeleton definition
        self.joint_connections = [
            # Head and neck
            (0, 1), 
            # Shoulders
            (1, 5), (1, 6),
            # Arms
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            # Torso
            (1, 11), (1, 12),
            # Legs
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]
        
        # Feature projection network
        self.feature_projector = nn.Sequential(
            nn.Linear(self.computed_feat_dim * num_joints, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Extract features from pose sequences.
        
        Args:
            poses: Pose sequence tensor [batch_size, seq_len, num_joints, input_dim]
            
        Returns:
            Pose features [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len, num_joints, input_dim = poses.shape
        
        # Normalize poses if required
        if self.normalize:
            # Center poses around root joint (typically joint 0)
            root = poses[:, :, 0:1, :]  # [batch, seq, 1, input_dim]
            poses = poses - root
            
            # Scale by the maximum distance from root
            dist_from_root = torch.norm(poses, dim=-1, keepdim=True)  # [batch, seq, joints, 1]
            max_dist = torch.max(dist_from_root, dim=2, keepdim=True)[0]  # [batch, seq, 1, 1]
            poses = poses / (max_dist + 1e-8)
        
        # Initialize feature list with position data
        features = [poses]
        
        # Compute velocity
        if self.compute_velocity:
            # Calculate temporal difference
            # Pad with zeros at the beginning to maintain sequence length
            pad = torch.zeros_like(poses[:, 0:1, :, :])
            velocity = torch.cat([pad, poses[:, 1:, :, :] - poses[:, :-1, :, :]], dim=1)
            features.append(velocity)
        
        # Compute acceleration
        if self.compute_acceleration:
            if self.compute_velocity:
                # Use velocity to compute acceleration
                pad = torch.zeros_like(velocity[:, 0:1, :, :])
                acceleration = torch.cat([pad, velocity[:, 1:, :, :] - velocity[:, :-1, :, :]], dim=1)
            else:
                # Compute second-order difference directly
                pad = torch.zeros_like(poses[:, 0:1, :, :])
                acceleration = torch.cat([
                    pad, 
                    pad, 
                    poses[:, 2:, :, :] - 2 * poses[:, 1:-1, :, :] + poses[:, :-2, :, :]
                ], dim=1)
            features.append(acceleration)
        
        # Compute jerk (derivative of acceleration)
        if self.compute_jerk:
            if self.compute_acceleration:
                # Use acceleration to compute jerk
                pad = torch.zeros_like(acceleration[:, 0:1, :, :])
                jerk = torch.cat([pad, acceleration[:, 1:, :, :] - acceleration[:, :-1, :, :]], dim=1)
            else:
                # Compute third-order difference directly
                pad = torch.zeros_like(poses[:, 0:1, :, :])
                jerk = torch.cat([
                    pad, pad, pad,
                    poses[:, 3:, :, :] - 3 * poses[:, 2:-1, :, :] + 3 * poses[:, 1:-2, :, :] - poses[:, :-3, :, :]
                ], dim=1)
            features.append(jerk)
        
        # Compute joint angles
        if self.compute_angles:
            angles = []
            
            for b in range(batch_size):
                batch_angles = []
                
                for t in range(seq_len):
                    # Compute angles for each joint connection
                    joint_angles = []
                    
                    for parent, child in self.joint_connections:
                        # Get joint vectors
                        v1 = poses[b, t, child] - poses[b, t, parent]  # [input_dim]
                        
                        # Normalize vector
                        v1_norm = F.normalize(v1, dim=0)
                        
                        # Compute angles with world axes
                        x_angle = torch.acos(torch.clamp(v1_norm[0], -1.0, 1.0))
                        y_angle = torch.acos(torch.clamp(v1_norm[1], -1.0, 1.0))
                        z_angle = torch.acos(torch.clamp(v1_norm[2], -1.0, 1.0))
                        
                        joint_angles.extend([x_angle, y_angle, z_angle])
                    
                    # For joints without connections, use default values
                    while len(joint_angles) < num_joints * 3:
                        joint_angles.append(torch.tensor(0.0, device=poses.device))
                    
                    batch_angles.append(torch.stack(joint_angles))
                
                angles.append(torch.stack(batch_angles))
            
            # Stack batch dimension
            angles = torch.stack(angles)  # [batch, seq, joints*3]
            
            # Reshape to match other features
            angles = angles.reshape(batch_size, seq_len, num_joints, 3)
            features.append(angles)
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=-1)  # [batch, seq, joints, combined_dim]
        
        # Apply joint weights
        weighted_features = combined_features * self.joint_weights.view(1, 1, num_joints, 1)
        
        # Reshape for feature projection
        flat_features = weighted_features.reshape(batch_size * seq_len, -1)
        
        # Project to lower dimension
        projected_features = self.feature_projector(flat_features)
        
        # Reshape back to sequence
        output_features = projected_features.reshape(batch_size, seq_len, self.feature_dim)
        
        return output_features


class StyleAdaptiveWeights(nn.Module):
    """
    Learns adaptive weights for different dance styles.
    """
    def __init__(
        self,
        feature_dim: int = 128,
        style_dim: int = 64,
        num_styles: int = 5,
        num_metrics: int = 4  # position, velocity, acceleration, angles
    ):
        """
        Initialize StyleAdaptiveWeights module.
        
        Args:
            feature_dim: Input feature dimension
            style_dim: Style embedding dimension
            num_styles: Number of dance styles to recognize
            num_metrics: Number of distance metrics to weight
        """
        super(StyleAdaptiveWeights, self).__init__()
        
        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(feature_dim, style_dim * 2),
            nn.BatchNorm1d(style_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(style_dim * 2, style_dim)
        )
        
        # Style classifier
        self.style_classifier = nn.Linear(style_dim, num_styles)
        
        # Weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, num_metrics),
            nn.Softmax(dim=1)  # Ensure weights sum to 1
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate adaptive weights based on dance style.
        
        Args:
            features: Dance sequence features [batch_size, seq_len, feature_dim]
            
        Returns:
            Tuple of (metric weights, style embedding, style probabilities)
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Global pooling over sequence
        global_features = features.mean(dim=1)  # [batch, feature_dim]
        
        # Encode style
        style_embedding = self.style_encoder(global_features)
        
        # Classify style
        style_logits = self.style_classifier(style_embedding)
        style_probs = F.softmax(style_logits, dim=1)
        
        # Generate weights for different metrics
        metric_weights = self.weight_generator(style_embedding)
        
        return metric_weights, style_embedding, style_probs


class MultiScaleDTW(nn.Module):
    """
    Multi-scale Dynamic Time Warping for dance sequence comparison.
    """
    def __init__(
        self,
        feature_dim: int = 128,
        num_scales: int = 3,
        wavelet: str = "db4",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize MultiScaleDTW module.
        
        Args:
            feature_dim: Input feature dimension
            num_scales: Number of time scales to analyze
            wavelet: Wavelet type for multi-scale decomposition
            device: Device to use (cuda or cpu)
        """
        super(MultiScaleDTW, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.wavelet = wavelet
        self.device = device
        
        # Scale weights (trainable parameters)
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
    
    def forward(
        self, 
        seq1: torch.Tensor, 
        seq2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
        metric_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute multi-scale DTW between two sequences.
        
        Args:
            seq1: First sequence [batch_size, seq_len1, feature_dim]
            seq2: Second sequence [batch_size, seq_len2, feature_dim]
            mask1: Optional mask for seq1 [batch_size, seq_len1]
            mask2: Optional mask for seq2 [batch_size, seq_len2]
            metric_weights: Optional weights for distance metrics [batch_size, num_metrics]
            
        Returns:
            Tuple of (similarity scores, optimal paths, DTW matrices)
        """
        batch_size = seq1.size(0)
        
        # Use normalized scale weights
        norm_weights = F.softmax(self.scale_weights, dim=0)
        
        # Process each scale
        dtw_matrices = []
        optimal_paths = []
        scale_similarities = []
        
        for scale in range(self.num_scales):
            # Apply wavelet decomposition to downsample sequences
            if scale == 0:
                # Original scale
                seq1_scale = seq1
                seq2_scale = seq2
                mask1_scale = mask1
                mask2_scale = mask2
            else:
                # Use wavelet transform to get lower resolution
                seq1_scale, mask1_scale = self._wavelet_downsample(seq1, mask1, level=scale)
                seq2_scale, mask2_scale = self._wavelet_downsample(seq2, mask2, level=scale)
            
            # Compute DTW for this scale
            dtw_matrix, path, similarity = self._compute_dtw(
                seq1_scale, seq2_scale, mask1_scale, mask2_scale, metric_weights
            )
            
            dtw_matrices.append(dtw_matrix)
            optimal_paths.append(path)
            scale_similarities.append(similarity)
        
        # Combine similarities from different scales
        combined_similarity = torch.zeros(batch_size, device=self.device)
        for i in range(self.num_scales):
            combined_similarity += norm_weights[i] * scale_similarities[i]
        
        return combined_similarity, optimal_paths, dtw_matrices
    
    def _wavelet_downsample(
        self, 
        sequence: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        level: int = 1
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Downsample sequence using wavelet transform.
        
        Args:
            sequence: Input sequence [batch_size, seq_len, feature_dim]
            mask: Optional mask [batch_size, seq_len]
            level: Wavelet decomposition level
            
        Returns:
            Tuple of (downsampled sequence, downsampled mask)
        """
        batch_size, seq_len, feature_dim = sequence.shape
        
        # Process each batch and feature dimension separately
        downsampled_seq = []
        
        for b in range(batch_size):
            feature_list = []
            
            for f in range(feature_dim):
                # Extract feature sequence
                feat_seq = sequence[b, :, f].cpu().numpy()
                
                # Apply wavelet transform
                coeffs = pywt.wavedec(feat_seq, self.wavelet, level=level)
                # Take approximation coefficients
                approx = coeffs[0]
                
                # Convert back to tensor
                downsampled_feat = torch.tensor(
                    approx, dtype=sequence.dtype, device=sequence.device
                )
                
                feature_list.append(downsampled_feat)
            
            # Stack features
            downsampled_seq.append(torch.stack(feature_list, dim=1))
        
        # Stack batches
        downsampled_seq = torch.stack(downsampled_seq)  # [batch, down_seq, feature_dim]
        
        # Downsample mask if provided
        downsampled_mask = None
        if mask is not None:
            # Simple downsampling for mask
            down_size = downsampled_seq.size(1)
            downsampled_mask = F.interpolate(
                mask.float().unsqueeze(1),  # [batch, 1, seq_len]
                size=down_size,
                mode='nearest'
            ).squeeze(1).bool()
        
        return downsampled_seq, downsampled_mask
    
    def _compute_dtw(
        self, 
        seq1: torch.Tensor, 
        seq2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
        metric_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute DTW between two sequences.
        
        Args:
            seq1: First sequence [batch_size, seq_len1, feature_dim]
            seq2: Second sequence [batch_size, seq_len2, feature_dim]
            mask1: Optional mask for seq1 [batch_size, seq_len1]
            mask2: Optional mask for seq2 [batch_size, seq_len2]
            metric_weights: Optional weights for distance metrics [batch_size, num_metrics]
            
        Returns:
            Tuple of (DTW distance matrix, optimal path, similarity score)
        """
        batch_size, seq_len1, feature_dim = seq1.shape
        seq_len2 = seq2.size(1)
        
        # Initialize DTW matrix [batch, seq_len1+1, seq_len2+1]
        # Add extra row and column for initialization
        dtw_matrix = torch.zeros(
            batch_size, seq_len1 + 1, seq_len2 + 1, 
            device=self.device
        )
        
        # Initialize first row and column to infinity
        dtw_matrix[:, 0, 1:] = float('inf')
        dtw_matrix[:, 1:, 0] = float('inf')
        
        # Main DTW computation loop
        for i in range(1, seq_len1 + 1):
            for j in range(1, seq_len2 + 1):
                # Get local cost
                cost = self._compute_distance(
                    seq1[:, i-1, :], seq2[:, j-1, :], metric_weights
                )
                
                # Apply mask if provided
                if mask1 is not None and mask2 is not None:
                    mask_val = mask1[:, i-1].float() * mask2[:, j-1].float()
                    cost = cost * mask_val + (1 - mask_val) * float('inf')
                
                # Find minimum of three adjacent cells
                dtw_matrix[:, i, j] = cost + torch.min(
                    dtw_matrix[:, i-1, j-1],  # diagonal
                    torch.min(
                        dtw_matrix[:, i-1, j],    # vertical
                        dtw_matrix[:, i, j-1]     # horizontal
                    )
                )
        
        # Get normalized DTW distance
        dtw_distance = dtw_matrix[:, -1, -1] / (seq_len1 + seq_len2)
        
        # Convert to similarity score (inverse of distance)
        similarity = 1.0 / (1.0 + dtw_distance)
        
        # Backtrack to find optimal path
        paths = []
        for b in range(batch_size):
            path = []
            i, j = seq_len1, seq_len2
            
            while i > 0 and j > 0:
                path.append((i-1, j-1))
                
                # Find the minimum direction
                diag = dtw_matrix[b, i-1, j-1]
                left = dtw_matrix[b, i, j-1]
                up = dtw_matrix[b, i-1, j]
                
                min_val = min(diag, left, up)
                
                if min_val == diag:
                    i, j = i-1, j-1
                elif min_val == left:
                    j = j-1
                else:
                    i = i-1
            
            # Reverse path to get correct order
            path.reverse()
            paths.append(path)
        
        # Convert paths to tensor representation
        # For simplicity, we'll just return the DTW matrix
        # A more complete implementation would encode the paths as tensors
        
        return dtw_matrix, paths, similarity
    
    def _compute_distance(
        self, 
        feat1: torch.Tensor, 
        feat2: torch.Tensor,
        metric_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distance between feature vectors.
        
        Args:
            feat1: First feature vector [batch_size, feature_dim]
            feat2: Second feature vector [batch_size, feature_dim]
            metric_weights: Optional weights for distance metrics [batch_size, num_metrics]
            
        Returns:
            Distance between features [batch_size]
        """
        # Euclidean distance
        distance = torch.norm(feat1 - feat2, dim=1)
        
        # If metric weights provided, we could split features into different components
        # and weight them accordingly. For simplicity, we just return the Euclidean distance.
        
        return distance


class DanceDTW(nn.Module):
    """
    Complete 3D-DanceDTW module for dance similarity analysis.
    """
    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 128,
        num_joints: int = 25,
        num_styles: int = 5,
        num_scales: int = 3,
        wavelet: str = "db4",
        joint_weights: Optional[torch.Tensor] = None,
        similarity_threshold: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize DanceDTW module.
        
        Args:
            input_dim: Input dimension per joint (typically 3 for x,y,z)
            feature_dim: Feature dimension for comparison
            num_joints: Number of joints in the skeleton
            num_styles: Number of dance styles to recognize
            num_scales: Number of time scales to analyze
            wavelet: Wavelet type for multi-scale decomposition
            joint_weights: Optional weights for each joint
            similarity_threshold: Threshold for determining similarity
            device: Device to use (cuda or cpu)
        """
        super(DanceDTW, self).__init__()
        
        # Configuration
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # Feature extractor
        self.feature_extractor = PoseFeatureExtractor(
            input_dim=input_dim,
            feature_dim=feature_dim,
            num_joints=num_joints,
            joint_weights=joint_weights,
            normalize=True,
            compute_velocity=True,
            compute_acceleration=True,
            compute_angles=True
        )
        
        # Style adaptive weights
        self.adaptive_weights = StyleAdaptiveWeights(
            feature_dim=feature_dim,
            style_dim=64,
            num_styles=num_styles,
            num_metrics=4  # position, velocity, acceleration, angles
        )
        
        # Multi-scale DTW
        self.ms_dtw = MultiScaleDTW(
            feature_dim=feature_dim,
            num_scales=num_scales,
            wavelet=wavelet,
            device=device
        )
    
    def forward(
        self, 
        seq1: torch.Tensor, 
        seq2: torch.Tensor,
        compute_path: bool = False
    ) -> Dict[str, Any]:
        """
        Compute similarity between two dance sequences.
        
        Args:
            seq1: First dance sequence [batch_size, seq_len1, num_joints, input_dim]
            seq2: Second dance sequence [batch_size, seq_len2, num_joints, input_dim]
            compute_path: Whether to compute and return the optimal alignment path
            
        Returns:
            Dictionary with similarity results
        """
        # Extract features
        features1 = self.feature_extractor(seq1)
        features2 = self.feature_extractor(seq2)
        
        # Get adaptive weights based on dance style
        # We combine features from both sequences to determine style
        combined_features = torch.cat([features1, features2], dim=1)
        combined_features = combined_features.mean(dim=1, keepdim=True).expand(-1, features1.size(1), -1)
        
        metric_weights, style_embedding, style_probs = self.adaptive_weights(combined_features)
        
        # Compute multi-scale DTW
        similarity, paths, dtw_matrices = self.ms_dtw(
            features1, features2, metric_weights=metric_weights
        )
        
        # Determine if sequences are similar based on threshold
        is_similar = similarity > self.similarity_threshold
        
        # Prepare results
        results = {
            "similarity_score": similarity,
            "is_similar": is_similar,
            "style_embedding": style_embedding,
            "style_probabilities": style_probs,
            "metric_weights": metric_weights
        }
        
        # Include path information if requested
        if compute_path:
            results["optimal_paths"] = paths
            results["dtw_matrices"] = dtw_matrices
        
        return results


def create_model(config):
    """
    Create a DanceDTW model instance based on config.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Initialized DanceDTW model
    """
    # Get joint weights from config if available
    joint_weights = None
    if 'joint_weights' in config:
        joint_weights = torch.tensor(config['joint_weights'])
    
    model = DanceDTW(
        input_dim=config.get('input_dim', 3),
        feature_dim=config.get('feature_dim', 128),
        num_joints=config.get('joint_count', 25),
        num_styles=config.get('num_styles', 5),
        num_scales=config.get('wavelet_levels', 3),
        wavelet=config.get('wavelet', 'db4'),
        joint_weights=joint_weights,
        similarity_threshold=config.get('similarity_threshold', 0.7)
    )
    
    return model