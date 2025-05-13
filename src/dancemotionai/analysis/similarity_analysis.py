"""
Similarity Analysis Module for DanceMotionAI.

This module provides functionality for comparing choreographies and analyzing 
similarity between dance sequences. It implements the 3D-DanceDTW algorithm 
which considers multiple aspects of dance movements including joint positions, 
angles, velocities, and accelerations.

Key components:
- SimilarityAnalyzer: Main class for analyzing choreography similarity
- 3D-DanceDTW: Multi-scale dynamic time warping with adaptive weights
- StyleAdaptiveWeighting: Dynamic weight adjustment based on dance style
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import pywt  # PyWavelets for wavelet transforms

from ..utils.visualization import create_similarity_heatmap, plot_alignment_path
from ..utils.metrics import compute_joint_angles, compute_velocities, compute_accelerations


class StyleClassifier(nn.Module):
    """
    Neural network to classify dance styles from pose sequences.
    Used to inform the adaptive weighting system.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_styles: int):
        """
        Initialize the style classifier.
        
        Args:
            input_dim: Dimension of input features (joints * 3)
            hidden_dim: Dimension of hidden layers
            num_styles: Number of dance styles to classify
        """
        super(StyleClassifier, self).__init__()
        
        # 3D ResNet-like architecture
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim*4,
            dropout=0.1
        )
        
        # Final classification layers
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_styles)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for style classification.
        
        Args:
            x: Input pose sequence [batch_size, sequence_length, num_joints*3]
            
        Returns:
            Style classification logits [batch_size, num_styles]
        """
        # Reshape for 1D convolution [batch, channels, length]
        x = x.permute(0, 2, 1)
        
        # ResNet-like processing
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Reshape for transformer [batch, length, features]
        x = x.permute(0, 2, 1)
        
        # Transformer processing
        x = self.transformer_layer(x)
        
        # Self-attention pooling
        attn_output, _ = self.attn(x, x, x)
        
        # Global average pooling
        x = attn_output.mean(dim=1)
        
        # Classification
        return self.fc(x)


class WeightGenerator(nn.Module):
    """
    Generates adaptive weights for the distance measure based on style embedding.
    """
    
    def __init__(self, style_dim: int, num_weights: int = 7):
        """
        Initialize weight generator.
        
        Args:
            style_dim: Dimension of style embedding
            num_weights: Number of weights to generate (for different distance components)
        """
        super(WeightGenerator, self).__init__()
        
        self.num_weights = num_weights
        
        # MLP layers
        self.layers = nn.Sequential(
            nn.Linear(style_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_weights)
        )
        
        # GRU for temporal consistency
        self.gru = nn.GRU(num_weights, num_weights, batch_first=True)
        
    def forward(self, style_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate weights based on style embedding.
        
        Args:
            style_embedding: Style embedding vector [batch_size, style_dim]
            
        Returns:
            Weights for distance components [batch_size, num_weights]
        """
        # Generate initial weights
        weights = self.layers(style_embedding)
        
        # Apply GRU for temporal consistency (if batched sequence)
        if len(weights.shape) > 1 and weights.shape[0] > 1:
            weights, _ = self.gru(weights.unsqueeze(1))
            weights = weights.squeeze(1)
        
        # Normalize weights with softmax
        weights = F.softmax(weights, dim=-1)
        
        return weights


class SimilarityAnalyzer:
    """
    Main class for analyzing similarity between dance sequences.
    Implements the 3D-DanceDTW algorithm.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the similarity analyzer.
        
        Args:
            config: Configuration dictionary
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        
        # Initialize style classifier if configured
        if config.get("use_style_adaptive_weights", True):
            self.style_classifier = StyleClassifier(
                input_dim=config.get("input_dim", 75),  # 25 joints * 3 coordinates
                hidden_dim=config.get("hidden_dim", 512),
                num_styles=config.get("num_styles", 8)  # Number of dance styles to classify
            ).to(device)
            
            # Load pre-trained weights if available
            if "style_classifier_weights" in config:
                self.style_classifier.load_state_dict(
                    torch.load(config["style_classifier_weights"], map_location=device)
                )
            
            # Initialize weight generator
            self.weight_generator = WeightGenerator(
                style_dim=config.get("hidden_dim", 512),
                num_weights=config.get("num_weights", 7)
            ).to(device)
            
            # Load pre-trained weights if available
            if "weight_generator_weights" in config:
                self.weight_generator.load_state_dict(
                    torch.load(config["weight_generator_weights"], map_location=device)
                )
        
        # Default weights if not using adaptive weighting
        self.default_weights = config.get("default_weights", {
            "position": 0.35,
            "angle": 0.25,
            "velocity": 0.15,
            "acceleration": 0.1,
            "relative_distance": 0.05,
            "direction": 0.05,
            "quaternion": 0.05
        })
    
    def compute_distance(
        self, 
        pose1: np.ndarray, 
        pose2: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute weighted distance between two 3D poses.
        
        Args:
            pose1: First 3D pose [num_joints, 3]
            pose2: Second 3D pose [num_joints, 3]
            weights: Optional weights for distance components
            
        Returns:
            Weighted distance between poses
        """
        # Use default weights if not provided
        if weights is None:
            weights = self.default_weights
            
        # Convert to numpy arrays if tensors
        if isinstance(pose1, torch.Tensor):
            pose1 = pose1.detach().cpu().numpy()
        if isinstance(pose2, torch.Tensor):
            pose2 = pose2.detach().cpu().numpy()
            
        # Position distance (Euclidean)
        position_dist = np.mean(np.sqrt(np.sum((pose1 - pose2) ** 2, axis=1)))
        
        # Joint angle distance
        angles1 = compute_joint_angles(pose1)
        angles2 = compute_joint_angles(pose2)
        angle_dist = np.mean(np.abs(angles1 - angles2))
        
        # Velocity distance (using finite differences if sequences provided)
        vel_dist = 0.0
        if pose1.ndim > 2 and pose2.ndim > 2 and pose1.shape[0] > 1 and pose2.shape[0] > 1:
            vel1 = compute_velocities(pose1)
            vel2 = compute_velocities(pose2)
            vel_dist = np.mean(np.sqrt(np.sum((vel1 - vel2) ** 2, axis=(1, 2))))
        
        # Acceleration distance
        acc_dist = 0.0
        if pose1.ndim > 2 and pose2.ndim > 2 and pose1.shape[0] > 2 and pose2.shape[0] > 2:
            acc1 = compute_accelerations(pose1)
            acc2 = compute_accelerations(pose2)
            acc_dist = np.mean(np.sqrt(np.sum((acc1 - acc2) ** 2, axis=(1, 2))))
        
        # Relative distance between joints
        rel_dist1 = np.sqrt(np.sum((pose1[:, None, :] - pose1[None, :, :]) ** 2, axis=2))
        rel_dist2 = np.sqrt(np.sum((pose2[:, None, :] - pose2[None, :, :]) ** 2, axis=2))
        relative_dist = np.mean(np.abs(rel_dist1 - rel_dist2))
        
        # Movement direction
        direction_dist = 0.0
        if pose1.ndim > 2 and pose2.ndim > 2 and pose1.shape[0] > 1 and pose2.shape[0] > 1:
            dir1 = pose1[1:] - pose1[:-1]
            dir2 = pose2[1:] - pose2[:-1]
            # Normalize
            dir1 = dir1 / (np.linalg.norm(dir1, axis=2, keepdims=True) + 1e-8)
            dir2 = dir2 / (np.linalg.norm(dir2, axis=2, keepdims=True) + 1e-8)
            # Compute cosine distance
            dot_product = np.sum(dir1 * dir2, axis=2)
            direction_dist = np.mean(1 - dot_product)
        
        # Quaternion distance (rotation)
        quaternion_dist = 0.0
        # Note: Implementation would require quaternion computation
        # which is omitted here for simplicity
        
        # Compute weighted sum
        total_dist = (
            weights["position"] * position_dist +
            weights["angle"] * angle_dist +
            weights["velocity"] * vel_dist +
            weights["acceleration"] * acc_dist +
            weights["relative_distance"] * relative_dist +
            weights["direction"] * direction_dist +
            weights["quaternion"] * quaternion_dist
        )
        
        return total_dist
    
    def compute_distance_matrix(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute distance matrix between two sequences.
        
        Args:
            seq1: First sequence of 3D poses [seq_len1, num_joints, 3]
            seq2: Second sequence of 3D poses [seq_len2, num_joints, 3]
            weights: Optional weights for distance components
            
        Returns:
            Distance matrix [seq_len1, seq_len2]
        """
        # Use default weights if not provided
        if weights is None:
            weights = self.default_weights
            
        # Initialize distance matrix
        dist_matrix = np.zeros((len(seq1), len(seq2)))
        
        # Compute distance for each pair of frames
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                dist_matrix[i, j] = self.compute_distance(seq1[i], seq2[j], weights)
                
        return dist_matrix
    
    def wavelet_decompose(
        self, 
        sequence: np.ndarray, 
        wavelet: str = 'db4', 
        level: int = 3
    ) -> List[np.ndarray]:
        """
        Decompose sequence using wavelet transform for multi-scale analysis.
        
        Args:
            sequence: Sequence to decompose [seq_len, num_joints, 3]
            wavelet: Wavelet type to use
            level: Decomposition level
            
        Returns:
            List of approximation coefficients at different scales
        """
        # Reshape for wavelet transform
        orig_shape = sequence.shape
        sequence_flat = sequence.reshape(orig_shape[0], -1)
        
        # Perform wavelet decomposition
        coeffs = []
        for i in range(sequence_flat.shape[1]):
            # Get coefficients for each joint coordinate
            c = pywt.wavedec(sequence_flat[:, i], wavelet, level=level)
            if i == 0:
                # Initialize list of lists for coefficients
                coeffs = [[] for _ in range(len(c))]
            
            # Append coefficients
            for j, coeff in enumerate(c):
                coeffs[j].append(coeff)
        
        # Convert to numpy arrays with proper shape
        multi_scale_seq = []
        for i, c in enumerate(coeffs):
            c_array = np.array(c).T  # Transpose to get [seq_len, num_joints*3]
            
            # Only use approximation coefficients (first element)
            if i == 0:
                # Reshape back to original format
                seq_len = c_array.shape[0]
                c_reshaped = c_array.reshape(seq_len, orig_shape[1], orig_shape[2])
                multi_scale_seq.append(c_reshaped)
                
        return multi_scale_seq
    
    def compute_dtw(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, np.ndarray, List[Tuple[int, int]]]:
        """
        Compute DTW distance between two sequences.
        
        Args:
            seq1: First sequence [seq_len1, num_joints, 3]
            seq2: Second sequence [seq_len2, num_joints, 3]
            weights: Optional weights for distance components
            
        Returns:
            DTW distance, accumulated cost matrix, and optimal path
        """
        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(seq1, seq2, weights)
        
        # Initialize accumulated cost matrix
        acc_matrix = np.zeros_like(dist_matrix)
        acc_matrix[0, 0] = dist_matrix[0, 0]
        
        # Fill first row and column
        for i in range(1, len(seq1)):
            acc_matrix[i, 0] = acc_matrix[i-1, 0] + dist_matrix[i, 0]
        for j in range(1, len(seq2)):
            acc_matrix[0, j] = acc_matrix[0, j-1] + dist_matrix[0, j]
        
        # Fill the rest of the matrix
        for i in range(1, len(seq1)):
            for j in range(1, len(seq2)):
                acc_matrix[i, j] = dist_matrix[i, j] + min(
                    acc_matrix[i-1, j],    # Insertion
                    acc_matrix[i, j-1],    # Deletion
                    acc_matrix[i-1, j-1]   # Match
                )
        
        # Backtracking to find optimal path
        path = []
        i, j = len(seq1) - 1, len(seq2) - 1
        path.append((i, j))
        
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                min_cost = min(
                    acc_matrix[i-1, j],
                    acc_matrix[i, j-1],
                    acc_matrix[i-1, j-1]
                )
                
                if min_cost == acc_matrix[i-1, j-1]:
                    i -= 1
                    j -= 1
                elif min_cost == acc_matrix[i-1, j]:
                    i -= 1
                else:
                    j -= 1
                    
            path.append((i, j))
            
        # Reverse path
        path.reverse()
        
        # Calculate normalized DTW distance
        dtw_distance = acc_matrix[-1, -1] / len(path)
        
        return dtw_distance, acc_matrix, path
    
    def compute_multi_scale_dtw(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
        wavelet: str = 'db4',
        level: int = 3,
        scale_weights: Optional[List[float]] = None
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Compute DTW with multi-scale analysis.
        
        Args:
            seq1: First sequence [seq_len1, num_joints, 3]
            seq2: Second sequence [seq_len2, num_joints, 3]
            weights: Optional weights for distance components
            wavelet: Wavelet type for decomposition
            level: Decomposition level
            scale_weights: Weights for different scales
            
        Returns:
            Multi-scale DTW distance and optimal path
        """
        # Default scale weights if not provided
        if scale_weights is None:
            scale_weights = [0.6, 0.3, 0.1][:level]
            # Normalize weights
            scale_weights = [w / sum(scale_weights) for w in scale_weights]
        
        # Wavelet decomposition for multi-scale analysis
        seq1_scales = self.wavelet_decompose(seq1, wavelet, level)
        seq2_scales = self.wavelet_decompose(seq2, wavelet, level)
        
        # Compute DTW at each scale
        dtw_distances = []
        dtw_paths = []
        
        for i in range(min(len(seq1_scales), len(seq2_scales))):
            dtw_dist, _, path = self.compute_dtw(seq1_scales[i], seq2_scales[i], weights)
            dtw_distances.append(dtw_dist)
            dtw_paths.append(path)
        
        # Compute weighted average of DTW distances
        ms_dtw_distance = sum(d * w for d, w in zip(dtw_distances, scale_weights))
        
        # Use path from the finest scale (most detailed)
        optimal_path = dtw_paths[0]
        
        return ms_dtw_distance, optimal_path
    
    def classify_style(self, sequence: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Classify the dance style of a sequence.
        
        Args:
            sequence: Pose sequence [seq_len, num_joints, 3]
            
        Returns:
            Predicted style index and style embedding
        """
        # Only use if style classifier is initialized
        if not hasattr(self, 'style_classifier'):
            return 0, torch.zeros(1, self.config.get("hidden_dim", 512))
        
        # Prepare input
        x = torch.from_numpy(sequence).float()
        
        # Reshape to [batch, seq_len, features]
        if len(x.shape) == 3:
            x = x.reshape(1, x.shape[0], -1)  # [1, seq_len, num_joints*3]
        elif len(x.shape) == 2:
            x = x.reshape(1, 1, -1)  # [1, 1, num_joints*3]
            
        x = x.to(self.device)
        
        # Get style embedding (from before the final classification layer)
        with torch.no_grad():
            # Forward pass through most of the network
            x_reshaped = x.permute(0, 2, 1)  # [batch, channels, length]
            x_conv = F.relu(self.style_classifier.bn1(self.style_classifier.conv1(x_reshaped)))
            x_trans = x_conv.permute(0, 2, 1)  # [batch, length, features]
            features = self.style_classifier.transformer_layer(x_trans)
            
            # Self-attention pooling
            attn_output, _ = self.style_classifier.attn(features, features, features)
            
            # Global average pooling for embedding
            embedding = attn_output.mean(dim=1)
            
            # Get classification logits
            logits = self.style_classifier.fc(embedding)
            
            # Get predicted style
            pred_style = torch.argmax(logits, dim=1).item()
            
        return pred_style, embedding
    
    def generate_weights(self, style_embedding: torch.Tensor) -> Dict[str, float]:
        """
        Generate distance weights based on style embedding.
        
        Args:
            style_embedding: Style embedding from the classifier
            
        Returns:
            Dictionary of weights for distance components
        """
        # Only use if weight generator is initialized
        if not hasattr(self, 'weight_generator'):
            return self.default_weights
        
        # Generate weights
        with torch.no_grad():
            weights_tensor = self.weight_generator(style_embedding)
            weights_np = weights_tensor.cpu().numpy().flatten()
        
        # Convert to dictionary
        weight_names = list(self.default_weights.keys())
        weights_dict = {name: float(weights_np[i]) for i, name in enumerate(weight_names)}
        
        return weights_dict
    
    def compute_similarity(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray,
        use_adaptive_weights: bool = True,
        use_multi_scale: bool = True
    ) -> Dict[str, Any]:
        """
        Compute similarity between two choreography sequences.
        
        Args:
            seq1: First sequence [seq_len1, num_joints, 3]
            seq2: Second sequence [seq_len2, num_joints, 3]
            use_adaptive_weights: Whether to use style-based adaptive weights
            use_multi_scale: Whether to use multi-scale DTW
            
        Returns:
            Dictionary with similarity results
        """
        weights = self.default_weights
        
        # Use adaptive weights if configured
        if use_adaptive_weights and hasattr(self, 'style_classifier'):
            # Classify styles
            style1, embedding1 = self.classify_style(seq1)
            style2, embedding2 = self.classify_style(seq2)
            
            # Use average embedding for weight generation
            avg_embedding = (embedding1 + embedding2) / 2
            weights = self.generate_weights(avg_embedding)
            
            # Include style information in results
            style_info = {
                "sequence1_style": int(style1),
                "sequence2_style": int(style2)
            }
        else:
            style_info = {}
        
        # Compute DTW
        if use_multi_scale:
            dtw_distance, path = self.compute_multi_scale_dtw(seq1, seq2, weights)
            method = "multi_scale_dtw"
        else:
            dtw_distance, acc_matrix, path = self.compute_dtw(seq1, seq2, weights)
            method = "standard_dtw"
        
        # Compute similarity score (0-1, where 1 is identical)
        # Convert DTW distance to similarity score with exponential mapping
        similarity_score = np.exp(-dtw_distance)
        
        # Prepare results
        results = {
            "similarity_score": float(similarity_score),
            "dtw_distance": float(dtw_distance),
            "optimal_path": path,
            "method": method,
            "weights_used": weights,
            **style_info
        }
        
        # Add similarity classification based on thresholds
        if similarity_score >= 0.85:
            results["similarity_category"] = "Very Similar"
        elif similarity_score >= 0.7:
            results["similarity_category"] = "Similar"
        elif similarity_score >= 0.55:
            results["similarity_category"] = "Partially Similar"
        else:
            results["similarity_category"] = "Not Similar"
            
        # Identify similar segments
        similar_segments = self.identify_similar_segments(
            seq1, seq2, path, similarity_score, threshold=0.8
        )
        results["similar_segments"] = similar_segments
        
        return results
    
    def identify_similar_segments(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray, 
        path: List[Tuple[int, int]],
        global_similarity: float,
        threshold: float = 0.8,
        min_segment_length: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify contiguous segments with high similarity.
        
        Args:
            seq1: First sequence [seq_len1, num_joints, 3]
            seq2: Second sequence [seq_len2, num_joints, 3]
            path: Optimal alignment path from DTW
            global_similarity: Overall similarity score
            threshold: Similarity threshold to identify segments
            min_segment_length: Minimum length of segments to report
            
        Returns:
            List of similar segments with start/end frames and similarity scores
        """
        similar_segments = []
        current_segment = None
        
        # Iterate through alignment path
        for i, (idx1, idx2) in enumerate(path):
            # Calculate local similarity for this aligned pair
            local_distance = self.compute_distance(seq1[idx1], seq2[idx2])
            local_similarity = np.exp(-local_distance)
            
            # Check if this pair is part of a similar segment
            if local_similarity >= threshold:
                if current_segment is None:
                    # Start a new segment
                    current_segment = {
                        "start_idx1": idx1,
                        "start_idx2": idx2,
                        "end_idx1": idx1,
                        "end_idx2": idx2,
                        "similarities": [local_similarity]
                    }
                else:
                    # Continue current segment
                    current_segment["end_idx1"] = idx1
                    current_segment["end_idx2"] = idx2
                    current_segment["similarities"].append(local_similarity)
            else:
                # End current segment if it exists
                if current_segment is not None:
                    # Only add if segment is long enough
                    if (current_segment["end_idx1"] - current_segment["start_idx1"] + 1) >= min_segment_length:
                        avg_similarity = np.mean(current_segment["similarities"])
                        similar_segments.append({
                            "sequence1_start": int(current_segment["start_idx1"]),
                            "sequence1_end": int(current_segment["end_idx1"]),
                            "sequence2_start": int(current_segment["start_idx2"]),
                            "sequence2_end": int(current_segment["end_idx2"]),
                            "segment_similarity": float(avg_similarity),
                            "segment_length1": int(current_segment["end_idx1"] - current_segment["start_idx1"] + 1),
                            "segment_length2": int(current_segment["end_idx2"] - current_segment["start_idx2"] + 1)
                        })
                    current_segment = None
        
        # Handle last segment if exists
        if current_segment is not None:
            if (current_segment["end_idx1"] - current_segment["start_idx1"] + 1) >= min_segment_length:
                avg_similarity = np.mean(current_segment["similarities"])
                similar_segments.append({
                    "sequence1_start": int(current_segment["start_idx1"]),
                    "sequence1_end": int(current_segment["end_idx1"]),
                    "sequence2_start": int(current_segment["start_idx2"]),
                    "sequence2_end": int(current_segment["end_idx2"]),
                    "segment_similarity": float(avg_similarity),
                    "segment_length1": int(current_segment["end_idx1"] - current_segment["start_idx1"] + 1),
                    "segment_length2": int(current_segment["end_idx2"] - current_segment["start_idx2"] + 1)
                })
        
        return similar_segments


class SimilarityReportGenerator:
    """
    Generates detailed reports for choreography similarity analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def generate_report(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray,
        similarity_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed similarity report.
        
        Args:
            seq1: First sequence [seq_len1, num_joints, 3]
            seq2: Second sequence [seq_len2, num_joints, 3]
            similarity_results: Results from SimilarityAnalyzer
            output_path: Path to save visualizations (if None, only returns data)
            
        Returns:
            Dictionary with report information and visualization paths
        """
        report = {
            "overall_similarity": similarity_results["similarity_score"],
            "similarity_category": similarity_results["similarity_category"],
            "similar_segments": similarity_results["similar_segments"],
            "method_used": similarity_results["method"],
            "visualizations": {}
        }
        
        # Generate visualizations if output path is provided
        if output_path is not None:
            # Create similarity heatmap
            heatmap_path = f"{output_path}/similarity_heatmap.png"
            dist_matrix = self.compute_distance_matrix_for_vis(seq1, seq2, similarity_results["weights_used"])
            create_similarity_heatmap(dist_matrix, similarity_results["optimal_path"], heatmap_path)
            report["visualizations"]["heatmap"] = heatmap_path
            
            # Create alignment path visualization
            path_vis_path = f"{output_path}/alignment_path.png"
            plot_alignment_path(similarity_results["optimal_path"], seq1.shape[0], seq2.shape[0], path_vis_path)
            report["visualizations"]["alignment_path"] = path_vis_path
            
            # Add more visualizations as needed
        
        return report
    
    def compute_distance_matrix_for_vis(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray,
        weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute distance matrix for visualization purposes.
        This is a simplified version for visualization only.
        
        Args:
            seq1: First sequence [seq_len1, num_joints, 3]
            seq2: Second sequence [seq_len2, num_joints, 3]
            weights: Weights for distance components
            
        Returns:
            Distance matrix [seq_len1, seq_len2]
        """
        # Initialize distance matrix
        dist_matrix = np.zeros((len(seq1), len(seq2)))
        
        # Simple Euclidean distance for visualization
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                # Compute Euclidean distance between poses
                pose_dist = np.mean(np.sqrt(np.sum((seq1[i] - seq2[j]) ** 2, axis=1)))
                dist_matrix[i, j] = pose_dist
                
        return dist_matrix
