"""
DanceHRNet: High-precision 3D pose estimation network for dance motion analysis.

This module implements the DanceHRNet with Adaptive Graph Transformer (AGT) blocks,
Global-Local Adaptive Graph Convolutional Network (GLA-GCN),
and Keypoint Information-based Rotation Optimization (KITRO).

It also includes the 4DHands module for detailed hand gesture recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class AGTBlock(nn.Module):
    """
    Adaptive Graph Transformer (AGT) block.
    
    Combines Transformer for global context modeling with Graph Convolutional
    Network (GCN) for skeletal structure awareness.
    """
    
    def __init__(self, 
                 input_dim: int = 256, 
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_gcn_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize the AGT block.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            num_heads: Number of attention heads in Transformer
            num_gcn_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super(AGTBlock, self).__init__()
        
        # Transformer module
        self.transformer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # GCN module
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            if i == 0:
                self.gcn_layers.append(GraphConv(input_dim, hidden_dim))
            elif i == num_gcn_layers - 1:
                self.gcn_layers.append(GraphConv(hidden_dim, input_dim))
            else:
                self.gcn_layers.append(GraphConv(hidden_dim, hidden_dim))
                
        # Adaptive fusion
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.fusion_layer = nn.Linear(input_dim * 2, input_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AGT block.
        
        Args:
            x: Input features of shape (batch_size, num_joints, input_dim)
            adj: Adjacency matrix of shape (batch_size, num_joints, num_joints)
            
        Returns:
            Output features of shape (batch_size, num_joints, input_dim)
        """
        # Transformer path
        transformer_out = self.transformer(x)
        
        # GCN path
        gcn_out = x
        for gcn_layer in self.gcn_layers:
            gcn_out = gcn_layer(gcn_out, adj)
            
        # Normalize fusion weights
        norm_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Fusion (version 1 - weighted sum)
        # fused = norm_weights[0] * transformer_out + norm_weights[1] * gcn_out
        
        # Fusion (version 2 - concatenation followed by projection)
        concat = torch.cat([transformer_out, gcn_out], dim=2)
        fused = self.fusion_layer(concat)
        
        return fused


class GraphConv(nn.Module):
    """
    Graph Convolutional Network layer.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the GraphConv layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
        """
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset learnable parameters."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GraphConv layer.
        
        Args:
            x: Input features of shape (batch_size, num_nodes, input_dim)
            adj: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            
        Returns:
            Output features of shape (batch_size, num_nodes, output_dim)
        """
        # Normalize adjacency matrix
        batch_size, num_nodes = adj.size(0), adj.size(1)
        
        # Add self-loops to adjacency matrix
        adj = adj + torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute degree matrix
        degree = torch.sum(adj, dim=2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        
        # Normalize adjacency matrix
        norm_adj = torch.zeros_like(adj)
        for i in range(batch_size):
            deg_inv_sqrt_i = torch.diag(degree_inv_sqrt[i])
            norm_adj[i] = torch.matmul(torch.matmul(deg_inv_sqrt_i, adj[i]), deg_inv_sqrt_i)
        
        # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(norm_adj, support)
        output = output + self.bias
        
        return F.relu(output)


class GLAGCNModule(nn.Module):
    """
    Global-Local Adaptive Graph Convolutional Network (GLA-GCN) module.
    
    Processes skeletal data at multiple resolutions for both fine-grained
    joint details and overall body structure.
    """
    
    def __init__(self, 
                 input_dim: int = 256, 
                 hidden_dim: int = 512,
                 num_joints: int = 25):
        """
        Initialize the GLA-GCN module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            num_joints: Number of joints in the skeleton
        """
        super(GLAGCNModule, self).__init__()
        
        # Define the hierarchical joint groups based on human skeleton
        # For example, we can define 5 groups: full body, upper body, lower body, left limbs, right limbs
        self.joint_groups = {
            'full_body': list(range(num_joints)),
            'upper_body': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Adjust these indices based on your skeleton
            'lower_body': [0, 9, 10, 11, 12, 13, 14],   # Adjust these indices based on your skeleton
            'left_limbs': [6, 7, 8, 12, 13, 14],        # Adjust these indices based on your skeleton
            'right_limbs': [3, 4, 5, 9, 10, 11]         # Adjust these indices based on your skeleton
        }
        
        # Create GCNs for each joint group
        self.group_gcns = nn.ModuleDict()
        for group_name, _ in self.joint_groups.items():
            self.group_gcns[group_name] = nn.Sequential(
                GraphConv(input_dim, hidden_dim),
                GraphConv(hidden_dim, input_dim)
            )
            
        # Adaptive fusion module
        self.fusion_weights = nn.Parameter(torch.ones(len(self.joint_groups)))
        self.fusion_layer = nn.Linear(input_dim * len(self.joint_groups), input_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GLA-GCN module.
        
        Args:
            x: Input features of shape (batch_size, num_joints, input_dim)
            adj: Adjacency matrix of shape (batch_size, num_joints, num_joints)
            
        Returns:
            Output features of shape (batch_size, num_joints, input_dim)
        """
        batch_size, num_joints, feat_dim = x.size()
        
        # Process each joint group
        group_outputs = []
        for group_name, joint_indices in self.joint_groups.items():
            # Extract features and adjacency matrix for the current group
            idx = torch.tensor(joint_indices, device=x.device)
            x_group = x[:, idx, :]
            
            # Create group adjacency matrix (only connections between joints in this group)
            adj_group = torch.zeros(batch_size, len(joint_indices), len(joint_indices), device=adj.device)
            for b in range(batch_size):
                for i, i_global in enumerate(joint_indices):
                    for j, j_global in enumerate(joint_indices):
                        adj_group[b, i, j] = adj[b, i_global, j_global]
            
            # Apply GCN for this group
            out_group = self.group_gcns[group_name][0](x_group, adj_group)
            out_group = self.group_gcns[group_name][1](out_group, adj_group)
            
            # Prepare for fusion (expand to full joint set)
            out_full = torch.zeros(batch_size, num_joints, feat_dim, device=x.device)
            out_full[:, idx, :] = out_group
            
            group_outputs.append(out_full)
            
        # Normalize fusion weights
        norm_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Adaptive fusion using learned weights
        # Step 1: Weighted sum approach
        fused = torch.zeros_like(x)
        for i, out in enumerate(group_outputs):
            fused += norm_weights[i] * out
            
        # Alternative fusion approach (concatenation followed by projection)
        # concat = torch.cat(group_outputs, dim=2)
        # fused = self.fusion_layer(concat)
        
        return fused


class KITROModule(nn.Module):
    """
    Keypoint Information-based Rotation Optimization (KITRO) module.
    
    Uses quaternion-based rotation representation for accurate 3D pose estimation.
    """
    
    def __init__(self, num_joints: int = 25):
        """
        Initialize the KITRO module.
        
        Args:
            num_joints: Number of joints in the skeleton
        """
        super(KITROModule, self).__init__()
        
        # Quaternion prediction network
        self.quat_predictor = nn.Sequential(
            nn.Linear(num_joints * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints * 4)  # 4 components for each quaternion (w, x, y, z)
        )
        
        # Rotation matrix conversion layer
        self.rotation_layer = nn.Linear(num_joints * 9, num_joints * 3)  # 9 elements in 3x3 rotation matrix
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KITRO module.
        
        Args:
            x: Input 3D keypoints of shape (batch_size, num_joints, 3)
            
        Returns:
            Rotation-optimized 3D keypoints of shape (batch_size, num_joints, 3)
        """
        batch_size, num_joints, _ = x.size()
        
        # Flatten input
        x_flat = x.reshape(batch_size, -1)
        
        # Predict quaternions
        quats = self.quat_predictor(x_flat).reshape(batch_size, num_joints, 4)
        
        # Normalize quaternions
        quats_norm = F.normalize(quats, p=2, dim=2)
        
        # Convert quaternions to rotation matrices
        rot_matrices = self._quaternion_to_rotation_matrix(quats_norm)
        rot_matrices_flat = rot_matrices.reshape(batch_size, -1)
        
        # Apply rotation optimization
        x_rotated = self.rotation_layer(rot_matrices_flat).reshape(batch_size, num_joints, 3)
        
        # Residual connection
        x_output = x + x_rotated
        
        return x_output
    
    def _quaternion_to_rotation_matrix(self, quaternion: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternions to rotation matrices.
        
        Args:
            quaternion: Quaternions of shape (batch_size, num_joints, 4) where each quaternion
                       is represented as (w, x, y, z)
            
        Returns:
            Rotation matrices of shape (batch_size, num_joints, 3, 3)
        """
        batch_size, num_joints, _ = quaternion.size()
        
        # Extract quaternion components
        w, x, y, z = quaternion[:, :, 0], quaternion[:, :, 1], quaternion[:, :, 2], quaternion[:, :, 3]
        
        # Compute rotation matrix elements
        # First row
        r00 = 1 - 2 * (y**2 + z**2)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
        
        # Second row
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x**2 + z**2)
        r12 = 2 * (y * z - x * w)
        
        # Third row
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x**2 + y**2)
        
        # Stack to form rotation matrices
        rotation_matrices = torch.stack([
            torch.stack([r00, r01, r02], dim=2),
            torch.stack([r10, r11, r12], dim=2),
            torch.stack([r20, r21, r22], dim=2)
        ], dim=3)
        
        # Reshape to (batch_size, num_joints, 3, 3)
        rotation_matrices = rotation_matrices.permute(1, 2, 0, 3)
        
        return rotation_matrices


class HandsRAT(nn.Module):
    """
    Relation-aware Two-Hand Tokenization (RAT) module for 4DHands.
    
    Models the relationships between finger joints for precise hand tracking.
    """
    
    def __init__(self, input_dim: int = 63, hidden_dim: int = 128, output_dim: int = 256):
        """
        Initialize the HandsRAT module.
        
        Args:
            input_dim: Input dimension (21 hand joints * 3 coordinates = 63)
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super(HandsRAT, self).__init__()
        
        # Graph attention network for hand joint relationships
        self.gat = GraphAttentionLayer(input_dim, hidden_dim)
        
        # Temporal convolution for capturing hand motion dynamics
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=3,
            padding=1
        )
        
        # Final projection
        self.projection = nn.Linear(output_dim, output_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                hand_adjacency: torch.Tensor, 
                hand_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the HandsRAT module.
        
        Args:
            x: Hand joint coordinates of shape (batch_size, sequence_length, 21, 3)
            hand_adjacency: Hand joint adjacency matrix of shape (21, 21)
            hand_mask: Optional mask for valid hand detection of shape (batch_size, sequence_length)
            
        Returns:
            Hand features of shape (batch_size, sequence_length, output_dim)
        """
        batch_size, seq_len, num_joints, coords = x.size()
        
        # Reshape input for GAT
        x_reshaped = x.reshape(batch_size * seq_len, num_joints, coords)
        
        # Apply GAT for joint relationships
        gat_output = self.gat(x_reshaped, hand_adjacency)
        
        # Reshape for temporal convolution
        gat_output = gat_output.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
        
        # Apply temporal convolution
        temp_output = self.temporal_conv(gat_output)
        
        # Reshape and project
        output = temp_output.permute(0, 2, 1)
        output = self.projection(output)
        
        # Apply mask if provided
        if hand_mask is not None:
            output = output * hand_mask.unsqueeze(-1)
        
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for modeling relationships between joints.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        """
        Initialize the Graph Attention Layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Transformation matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LeakyReLU and Dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Graph Attention Layer.
        
        Args:
            h: Input features of shape (batch_size, num_nodes, in_features)
            adj: Adjacency matrix of shape (num_nodes, num_nodes)
            
        Returns:
            Output features of shape (batch_size, num_nodes, out_features)
        """
        batch_size, N = h.size(0), h.size(1)
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (batch_size, N, out_features)
        
        # Compute attention coefficients
        a_input = torch.cat([Wh.repeat(1, 1, N).view(batch_size, N * N, self.out_features),
                             Wh.repeat(1, N, 1)], dim=2).view(batch_size, N, N, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (batch_size, N, N)
        
        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).expand(batch_size, -1, -1)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # (batch_size, N, out_features)
        
        # Add bias
        h_prime = h_prime + self.bias
        
        return F.elu(h_prime)


class HandsSIR(nn.Module):
    """
    Spatio-temporal Interaction Reasoning (SIR) module for 4DHands.
    
    Analyzes interactions between hands for gesture recognition.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 256):
        """
        Initialize the HandsSIR module.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super(HandsSIR, self).__init__()
        
        # Cross-attention for hand interaction
        self.cross_attention = CrossAttention(input_dim, num_heads=4)
        
        # LSTM-CRF for gesture reasoning
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Projection layer
        self.projection = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        
    def forward(self, 
                left_hand: torch.Tensor, 
                right_hand: torch.Tensor, 
                hand_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the HandsSIR module.
        
        Args:
            left_hand: Left hand features of shape (batch_size, sequence_length, input_dim)
            right_hand: Right hand features of shape (batch_size, sequence_length, input_dim)
            hand_mask: Optional mask for valid hand detection of shape (batch_size, sequence_length, 2)
            
        Returns:
            Tuple of (left_hand_interaction, right_hand_interaction) features
        """
        # Apply cross-attention for hand interaction
        left_attended = self.cross_attention(left_hand, right_hand, right_hand)
        right_attended = self.cross_attention(right_hand, left_hand, left_hand)
        
        # Apply mask if provided
        if hand_mask is not None:
            left_mask = hand_mask[:, :, 0].unsqueeze(-1)
            right_mask = hand_mask[:, :, 1].unsqueeze(-1)
            left_attended = left_attended * left_mask
            right_attended = right_attended * right_mask
        
        # Concatenate for joint processing
        combined = torch.cat([left_attended, right_attended], dim=1)  # (batch_size, 2*seq_len, input_dim)
        
        # Apply LSTM for temporal reasoning
        lstm_out, _ = self.lstm(combined)
        
        # Split back to left and right hands
        seq_len = left_attended.size(1)
        left_lstm = lstm_out[:, :seq_len, :]
        right_lstm = lstm_out[:, seq_len:, :]
        
        # Final projection
        left_output = self.projection(left_lstm)
        right_output = self.projection(right_lstm)
        
        return left_output, right_output


class CrossAttention(nn.Module):
    """
    Cross-attention module for interactions between two sequences.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize the CrossAttention module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossAttention, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the CrossAttention module.
        
        Args:
            query: Query tensor of shape (batch_size, query_len, embed_dim)
            key: Key tensor of shape (batch_size, key_len, embed_dim)
            value: Value tensor of shape (batch_size, value_len, embed_dim)
            key_padding_mask: Optional mask for padding in keys
            
        Returns:
            Attended features of shape (batch_size, query_len, embed_dim)
        """
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        return attn_output


class Hands4D(nn.Module):
    """
    4DHands module for detailed hand gesture recognition.
    
    Integrates RAT (Relation-aware Two-Hand Tokenization) and 
    SIR (Spatio-temporal Interaction Reasoning) modules.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, output_dim: int = 256):
        """
        Initialize the 4DHands module.
        
        Args:
            input_dim: Input dimension per joint coordinate (default: 3 for x, y, z)
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super(Hands4D, self).__init__()
        
        # Hand adjacency matrix (predefined based on finger connections)
        self.register_buffer('hand_adjacency', self._get_hand_adjacency_matrix())
        
        # RAT module for each hand
        self.left_rat = HandsRAT(input_dim * 21, hidden_dim, output_dim)
        self.right_rat = HandsRAT(input_dim * 21, hidden_dim, output_dim)
        
        # SIR module for hand interaction
        self.sir = HandsSIR(output_dim, hidden_dim * 2, output_dim)
        
        # Final classification layer (for gesture classification)
        self.classifier = nn.Linear(output_dim * 2, 64)  # 64 common hand gesture classes
        
    def forward(self, 
                left_hand: torch.Tensor, 
                right_hand: torch.Tensor, 
                hand_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the 4DHands module.
        
        Args:
            left_hand: Left hand joint coordinates of shape (batch_size, seq_len, 21, 3)
            right_hand: Right hand joint coordinates of shape (batch_size, seq_len, 21, 3)
            hand_mask: Optional mask for valid hand detection of shape (batch_size, seq_len, 2)
            
        Returns:
            Dictionary containing hand features and gesture logits
        """
        # Process each hand with RAT
        left_features = self.left_rat(left_hand, self.hand_adjacency, 
                                    hand_mask[:, :, 0] if hand_mask is not None else None)
        right_features = self.right_rat(right_hand, self.hand_adjacency,
                                       hand_mask[:, :, 1] if hand_mask is not None else None)
        
        # Process hand interactions with SIR
        left_interaction, right_interaction = self.sir(left_features, right_features, hand_mask)
        
        # Combine hand features for gesture classification
        combined_features = torch.cat([left_interaction, right_interaction], dim=2)
        gesture_logits = self.classifier(combined_features)
        
        # Return all intermediate features for potential use by other modules
        return {
            'left_features': left_features,
            'right_features': right_features,
            'left_interaction': left_interaction,
            'right_interaction': right_interaction,
            'gesture_logits': gesture_logits
        }
        
    def _get_hand_adjacency_matrix(self) -> torch.Tensor:
        """
        Create hand adjacency matrix based on finger connections.
        
        Returns:
            Adjacency matrix of shape (21, 21)
        """
        # Define the hand skeleton structure
        # 0: Wrist
        # 1-4: Thumb (from base to tip)
        # 5-8: Index finger
        # 9-12: Middle finger
        # 13-16: Ring finger
        # 17-20: Pinky
        
        # Initialize adjacency matrix
        adjacency = torch.zeros(21, 21)
        
        # Define connections
        connections = [
            # Wrist to finger bases
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            # Thumb connections
            (1, 2), (2, 3), (3, 4),
            # Index finger connections
            (5, 6), (6, 7), (7, 8),
            # Middle finger connections
            (9, 10), (10, 11), (11, 12),
            # Ring finger connections
            (13, 14), (14, 15), (15, 16),
            # Pinky connections
            (17, 18), (18, 19), (19, 20),
            # Cross-finger connections (optional)
            (1, 5), (5, 9), (9, 13), (13, 17)
        ]
        
        # Fill adjacency matrix
        for i, j in connections:
            adjacency[i, j] = 1
            adjacency[j, i] = 1  # Undirected graph
            
        # Add self-loops
        for i in range(21):
            adjacency[i, i] = 1
            
        return adjacency


class DanceHRNet(nn.Module):
    """
    DanceHRNet: High-precision 3D pose estimation network for dance motion analysis.
    
    Integrates AGT blocks, GLA-GCN, KITRO, and 4DHands modules.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the DanceHRNet.
        
        Args:
            config: Configuration dictionary
        """
        super(DanceHRNet, self).__init__()
        
        self.config = config or {}
        
        # Get configuration parameters
        self.input_dim = self.config.get('input_dim', 2048)
        self.feature_dim = self.config.get('feature_dim', 512)
        self.num_agt_blocks = self.config.get('agt_blocks', 6)
        self.num_attention_heads = self.config.get('attention_heads', 8)
        self.num_gcn_layers = self.config.get('gcn_layers', 3)
        self.joint_count = self.config.get('joint_count', 25)
        self.use_kitro = self.config.get('use_kitro', True)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.feature_dim)
        
        # AGT blocks
        self.agt_blocks = nn.ModuleList([
            AGTBlock(
                input_dim=self.feature_dim,
                hidden_dim=self.feature_dim * 2,
                num_heads=self.num_attention_heads,
                num_gcn_layers=self.num_gcn_layers
            ) for _ in range(self.num_agt_blocks)
        ])
        
        # GLA-GCN module
        self.gla_gcn = GLAGCNModule(
            input_dim=self.feature_dim,
            hidden_dim=self.feature_dim * 2,
            num_joints=self.joint_count
        )
        
        # KITRO module (optional)
        if self.use_kitro:
            self.kitro = KITROModule(num_joints=self.joint_count)
        
        # Output projection for 3D coordinates
        self.output_projection = nn.Linear(self.feature_dim, 3)
        
        # 4DHands module (optional)
        self.use_4dhands = self.config.get('use_4dhands', False)
        if self.use_4dhands:
            self.hands_4d = Hands4D(input_dim=3, hidden_dim=128, output_dim=256)
        
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the DanceHRNet.
        
        Args:
            x: Input features of shape (batch_size, num_joints, input_dim)
            adj: Optional adjacency matrix of shape (batch_size, num_joints, num_joints)
            
        Returns:
            Dictionary containing 3D pose estimates and additional features
        """
        batch_size, num_joints = x.size(0), x.size(1)
        
        # Create adjacency matrix if not provided
        if adj is None:
            adj = self._get_default_adjacency_matrix(batch_size, num_joints, x.device)
        
        # Input projection
        features = self.input_projection(x)
        
        # Apply AGT blocks
        for agt_block in self.agt_blocks:
            features = agt_block(features, adj)
        
        # Apply GLA-GCN
        features = self.gla_gcn(features, adj)
        
        # Output projection to get 3D coordinates
        poses_3d = self.output_projection(features)
        
        # Apply KITRO for rotation optimization (optional)
        if self.use_kitro:
            poses_3d = self.kitro(poses_3d)
        
        # Process hand regions with 4DHands (optional)
        hand_features = None
        if self.use_4dhands:
            # Extract hand regions (assuming specific joint indices for hands)
            left_hand_idx = torch.tensor([6, 7, 8] + list(range(25, 46)), device=x.device)  # Example indices
            right_hand_idx = torch.tensor([3, 4, 5] + list(range(46, 67)), device=x.device)  # Example indices
            
            left_hand = poses_3d[:, left_hand_idx, :]
            right_hand = poses_3d[:, right_hand_idx, :]
            
            # Apply 4DHands module
            hand_features = self.hands_4d(left_hand, right_hand)
        
        # Return results
        results = {
            'poses_3d': poses_3d,
            'features': features
        }
        
        if hand_features is not None:
            results['hand_features'] = hand_features
        
        return results
    
    def _get_default_adjacency_matrix(self, batch_size: int, num_joints: int, device: torch.device) -> torch.Tensor:
        """
        Create default adjacency matrix based on human skeleton.
        
        Args:
            batch_size: Batch size
            num_joints: Number of joints
            device: Device to create tensor on
            
        Returns:
            Adjacency matrix of shape (batch_size, num_joints, num_joints)
        """
        # Initialize adjacency matrix
        adjacency = torch.zeros(num_joints, num_joints, device=device)
        
        # Define connections based on human skeleton
        # Example for a 25-joint skeleton
        connections = [
            (0, 1),    # Spine to neck
            (1, 2),    # Neck to head
            (1, 3),    # Neck to right shoulder
            (3, 4),    # Right shoulder to right elbow
            (4, 5),    # Right elbow to right wrist
            (1, 6),    # Neck to left shoulder
            (6, 7),    # Left shoulder to left elbow
            (7, 8),    # Left elbow to left wrist
            (0, 9),    # Spine to right hip
            (9, 10),   # Right hip to right knee
            (10, 11),  # Right knee to right ankle
            (0, 12),   # Spine to left hip
            (12, 13),  # Left hip to left knee
            (13, 14),  # Left knee to left ankle
            # Add more connections for hands and feet if needed
        ]
        
        # Fill adjacency matrix
        for i, j in connections:
            if i < num_joints and j < num_joints:  # Check if joint indices are valid
                adjacency[i, j] = 1
                adjacency[j, i] = 1  # Undirected graph
                
        # Add self-loops
        for i in range(num_joints):
            adjacency[i, i] = 1
            
        # Expand to batch dimension
        adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        
        return adjacency
    
    def extract_poses(self, video_path: str) -> Dict:
        """
        Extract 3D poses from a video.
        
        This is a high-level function that would be used in practice to process videos.
        For simplicity, we'll just implement a placeholder that returns random poses.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing extracted poses and metadata
        """
        import cv2
        import numpy as np
        from tqdm import tqdm
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # For demonstration purposes, generate random poses
        # In practice, this would involve running 2D pose detection on each frame,
        # then lifting to 3D using the DanceHRNet model
        print(f"Extracting 3D poses from {video_path}...")
        print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Random poses for demonstration
        poses = np.random.rand(total_frames, self.joint_count, 3) - 0.5  # Center around 0
        
        # Add a simple motion pattern
        t = np.linspace(0, 2*np.pi, total_frames)
        
        # Global up-down motion
        poses[:, :, 1] += 0.2 * np.sin(t)[:, np.newaxis]
        
        # Arm swinging motion (assuming joints 3-8 are arms)
        for i in range(3, 9):
            freq = 1 + 0.2 * (i - 3)  # Different frequency for each arm joint
            poses[:, i, 0] += 0.3 * np.sin(freq * t)
            poses[:, i, 1] += 0.2 * np.cos(freq * t)
        
        # Leg motion (assuming joints 9-14 are legs)
        for i in range(9, 15):
            freq = 0.5 + 0.1 * (i - 9)  # Different frequency for each leg joint
            poses[:, i, 0] += 0.2 * np.sin(freq * t + np.pi)
            poses[:, i, 2] += 0.1 * np.cos(freq * t + np.pi)
        
        # Return the results with metadata
        return {
            'poses': poses,
            'fps': fps,
            'num_frames': total_frames,
            'video_path': video_path,
            'joint_count': self.joint_count
        }
