"""
DanceFormer model for multimodal analysis of dance and music.

This module implements the DanceFormer architecture with cross-modal
attention mechanisms to correlate dance movements with music features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for correlating dance and music features.
    """
    def __init__(
        self,
        dance_dim: int = 512,
        music_dim: int = 128,
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize CrossModalAttention module.
        
        Args:
            dance_dim: Dimension of dance features
            music_dim: Dimension of music features
            output_dim: Dimension of output features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossModalAttention, self).__init__()
        
        # Project dance and music features to the same dimension
        self.dance_proj = nn.Linear(dance_dim, output_dim)
        self.music_proj = nn.Linear(music_dim, output_dim)
        
        # Multi-head attention
        self.dance2music_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.music2dance_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
    
    def forward(
        self, 
        dance_features: torch.Tensor, 
        music_features: torch.Tensor,
        dance_mask: Optional[torch.Tensor] = None,
        music_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for cross-modal attention.
        
        Args:
            dance_features: Dance features [batch_size, seq_len_dance, dance_dim]
            music_features: Music features [batch_size, seq_len_music, music_dim]
            dance_mask: Optional mask for dance features [batch_size, seq_len_dance]
            music_mask: Optional mask for music features [batch_size, seq_len_music]
            
        Returns:
            Fused dance-music features [batch_size, seq_len_dance, output_dim]
        """
        # Project features to the same dimension
        dance_proj = self.dance_proj(dance_features)  # [batch, dance_seq, dim]
        music_proj = self.music_proj(music_features)  # [batch, music_seq, dim]
        
        # Apply cross-attention from dance to music
        d2m_output, d2m_weights = self.dance2music_attn(
            query=dance_proj,
            key=music_proj,
            value=music_proj,
            key_padding_mask=music_mask
        )
        
        # Apply cross-attention from music to dance
        m2d_output, m2d_weights = self.music2dance_attn(
            query=music_proj,
            key=dance_proj,
            value=dance_proj,
            key_padding_mask=dance_mask
        )
        
        # Reshape m2d_output to match dance sequence length
        # This is a simplification - a more sophisticated approach would use
        # interpolation or dynamic time warping to align sequences
        m2d_output = F.interpolate(
            m2d_output.transpose(1, 2),  # [batch, dim, music_seq]
            size=dance_proj.size(1),     # dance_seq
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # [batch, dance_seq, dim]
        
        # Concatenate and project
        concat_features = torch.cat([d2m_output, m2d_output], dim=-1)
        output = self.output_proj(concat_features)
        
        return output, d2m_weights


class MultiScaleTransformer(nn.Module):
    """
    Multi-scale transformer for hierarchical analysis of dance choreography.
    """
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        num_scales: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize MultiScaleTransformer module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_scales: Number of time scales to process
            dropout: Dropout probability
        """
        super(MultiScaleTransformer, self).__init__()
        
        # Scale-specific transformers
        self.scale_transformers = nn.ModuleList([
            self._create_transformer_encoder(input_dim, hidden_dim, num_heads, num_layers, dropout)
            for _ in range(num_scales)
        ])
        
        # Downsampling and upsampling layers
        self.downsamplers = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size=2**i, stride=2**i, padding=0)
            for i in range(1, num_scales)
        ])
        
        # Cross-scale attention
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim * num_scales, input_dim),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        
        # Number of scales
        self.num_scales = num_scales
    
    def _create_transformer_encoder(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_heads: int, 
        num_layers: int, 
        dropout: float
    ) -> nn.TransformerEncoder:
        """Helper method to create a transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-scale transformer.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Multi-scale features [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Process at different scales
        scale_outputs = [self.scale_transformers[0](x, src_key_padding_mask=mask)]
        
        # Process lower resolution scales
        for i in range(1, self.num_scales):
            # Downsample
            x_down = x.transpose(1, 2)  # [batch, input_dim, seq_len]
            x_down = self.downsamplers[i-1](x_down)  # [batch, input_dim, seq_len / 2^i]
            x_down = x_down.transpose(1, 2)  # [batch, seq_len / 2^i, input_dim]
            
            # Create mask for downsampled sequence
            down_mask = None
            if mask is not None:
                # Downsample the mask - this is a simplification
                down_mask = mask[:, ::2**i][:, :x_down.size(1)]
            
            # Apply transformer
            scale_output = self.scale_transformers[i](x_down, src_key_padding_mask=down_mask)
            
            # Upsample back to original sequence length
            scale_output = F.interpolate(
                scale_output.transpose(1, 2),  # [batch, input_dim, seq_len / 2^i]
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, seq_len, input_dim]
            
            scale_outputs.append(scale_output)
        
        # Cross-scale attention for feature fusion
        fused_scales = []
        for i in range(self.num_scales):
            # Use the first scale as query, and each scale as key/value
            fused, _ = self.cross_scale_attn(
                query=scale_outputs[0],
                key=scale_outputs[i],
                value=scale_outputs[i]
            )
            fused_scales.append(fused)
        
        # Concatenate and project
        concat_scales = torch.cat(fused_scales, dim=-1)
        output = self.output_proj(concat_scales)
        
        return output


class DynamicGraphGenerator(nn.Module):
    """
    Dynamic graph generation for modeling dancer interactions.
    """
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_dancers: int = 5,
        threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        """
        Initialize DynamicGraphGenerator module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            num_dancers: Maximum number of dancers to model
            threshold: Threshold for edge creation
            dropout: Dropout probability
        """
        super(DynamicGraphGenerator, self).__init__()
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge prediction network
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Graph convolution
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.threshold = threshold
        self.num_dancers = num_dancers
    
    def forward(
        self, 
        dancer_features: torch.Tensor,
        valid_dancers: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for dynamic graph generation.
        
        Args:
            dancer_features: Features for each dancer [batch_size, num_dancers, input_dim]
            valid_dancers: Optional boolean mask for valid dancers [batch_size, num_dancers]
            
        Returns:
            Tuple of (updated dancer features, adjacency matrix)
        """
        batch_size, num_dancers, input_dim = dancer_features.shape
        
        # Transform features
        features = self.feature_transform(dancer_features)  # [batch, dancers, hidden]
        
        # Generate all pairs of dancer features
        dancer_i = features.unsqueeze(2).expand(-1, -1, num_dancers, -1)  # [batch, dancers, dancers, hidden]
        dancer_j = features.unsqueeze(1).expand(-1, num_dancers, -1, -1)  # [batch, dancers, dancers, hidden]
        
        # Concatenate pairs
        pairs = torch.cat([dancer_i, dancer_j], dim=-1)  # [batch, dancers, dancers, hidden*2]
        
        # Predict edges
        edges = self.edge_predictor(pairs).squeeze(-1)  # [batch, dancers, dancers]
        
        # Apply threshold to create binary adjacency matrix
        adj_matrix = (edges > self.threshold).float()
        
        # Set diagonal to 1 (self-connections)
        diag_mask = torch.eye(num_dancers, device=adj_matrix.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_matrix = adj_matrix + diag_mask
        adj_matrix = (adj_matrix > 0).float()  # Ensure binary values
        
        # Apply valid dancers mask if provided
        if valid_dancers is not None:
            valid_mask = valid_dancers.unsqueeze(1) & valid_dancers.unsqueeze(2)  # [batch, dancers, dancers]
            adj_matrix = adj_matrix * valid_mask
        
        # Normalize adjacency matrix for graph convolution
        deg = torch.sum(adj_matrix, dim=2, keepdim=True)  # [batch, dancers, 1]
        deg = torch.clamp(deg, min=1)  # Avoid division by zero
        norm_adj = adj_matrix / deg
        
        # Apply graph convolution
        graph_out = norm_adj @ self.graph_conv(features)  # [batch, dancers, hidden]
        
        # Project back to original dimension
        output = self.output_proj(graph_out)
        
        return output, adj_matrix


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output with positional encoding added [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DanceFormer(nn.Module):
    """
    DanceFormer model for multimodal analysis of dance and music.
    """
    def __init__(
        self,
        dance_dim: int = 512,
        music_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_scales: int = 3,
        max_dancers: int = 5,
        dropout: float = 0.1,
        fusion_type: str = "cross_attention",
    ):
        """
        Initialize DanceFormer model.
        
        Args:
            dance_dim: Dimension of dance features
            music_dim: Dimension of music features
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_scales: Number of time scales to process
            max_dancers: Maximum number of dancers to model
            dropout: Dropout probability
            fusion_type: Type of fusion mechanism ("cross_attention", "concat", "sum")
        """
        super(DanceFormer, self).__init__()
        
        self.fusion_type = fusion_type
        
        # Cross-modal attention for dance-music correlation
        self.cross_modal_attn = CrossModalAttention(
            dance_dim=dance_dim,
            music_dim=music_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Multi-scale transformer for hierarchical analysis
        self.multi_scale_transformer = MultiScaleTransformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            num_heads=num_heads,
            num_layers=num_layers,
            num_scales=num_scales,
            dropout=dropout
        )
        
        # Dynamic graph generator for dancer interactions
        self.dynamic_graph = DynamicGraphGenerator(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_dancers=max_dancers,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Alternative fusion modules
        if fusion_type == "concat":
            self.concat_fusion = nn.Sequential(
                nn.Linear(dance_dim + music_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        elif fusion_type == "sum":
            self.dance_proj = nn.Linear(dance_dim, hidden_dim)
            self.music_proj = nn.Linear(music_dim, hidden_dim)
    
    def forward(
        self, 
        dance_features: torch.Tensor,
        music_features: torch.Tensor,
        dance_mask: Optional[torch.Tensor] = None,
        music_mask: Optional[torch.Tensor] = None,
        dancer_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for DanceFormer model.
        
        Args:
            dance_features: Dance features 
                            [batch_size, seq_len, num_dancers, dance_dim] for multiple dancers or
                            [batch_size, seq_len, dance_dim] for single dancer
            music_features: Music features [batch_size, music_seq_len, music_dim]
            dance_mask: Optional mask for dance features [batch_size, seq_len]
            music_mask: Optional mask for music features [batch_size, music_seq_len]
            dancer_ids: Optional IDs for valid dancers [batch_size, num_dancers]
            
        Returns:
            Dictionary of outputs including fused features, attention weights, etc.
        """
        # Handle single dancer case
        multiple_dancers = len(dance_features.shape) == 4
        
        if not multiple_dancers:
            # Add dancer dimension [batch, seq, dance_dim] -> [batch, seq, 1, dance_dim]
            dance_features = dance_features.unsqueeze(2)
        
        batch_size, seq_len, num_dancers, dance_feat_dim = dance_features.shape
        
        # Process each dancer separately
        dancer_outputs = []
        dancer_attn_weights = []
        
        for i in range(num_dancers):
            dancer_feat = dance_features[:, :, i]  # [batch, seq, dance_dim]
            
            # Skip invalid dancers
            if dancer_ids is not None:
                valid_mask = dancer_ids[:, i].bool()  # [batch]
                if not valid_mask.any():
                    # Create dummy outputs for invalid dancers
                    dummy_output = torch.zeros(
                        batch_size, seq_len, dance_feat_dim, 
                        device=dance_features.device
                    )
                    dummy_attn = torch.zeros(
                        batch_size, seq_len, music_features.size(1),
                        device=dance_features.device
                    )
                    dancer_outputs.append(dummy_output)
                    dancer_attn_weights.append(dummy_attn)
                    continue
            
            # Perform fusion based on specified type
            if self.fusion_type == "cross_attention":
                # Cross-modal attention fusion
                fused_features, attn_weights = self.cross_modal_attn(
                    dancer_feat, music_features, dance_mask, music_mask
                )
            elif self.fusion_type == "concat":
                # Concatenation fusion
                # First, align sequence lengths
                music_feat_aligned = F.interpolate(
                    music_features.transpose(1, 2),  # [batch, music_dim, music_seq]
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [batch, seq, music_dim]
                
                # Concatenate and project
                concat_feat = torch.cat([dancer_feat, music_feat_aligned], dim=-1)
                fused_features = self.concat_fusion(concat_feat)
                attn_weights = None  # No attention weights for concat fusion
            elif self.fusion_type == "sum":
                # Sum fusion
                dance_proj = self.dance_proj(dancer_feat)
                
                # Align music sequence length
                music_feat_aligned = F.interpolate(
                    music_features.transpose(1, 2),  # [batch, music_dim, music_seq]
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [batch, seq, music_dim]
                
                music_proj = self.music_proj(music_feat_aligned)
                fused_features = dance_proj + music_proj
                attn_weights = None  # No attention weights for sum fusion
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
            # Apply multi-scale transformer
            fused_features = self.multi_scale_transformer(fused_features, dance_mask)
            
            dancer_outputs.append(fused_features)
            dancer_attn_weights.append(attn_weights)
        
        # Stack dancer outputs
        dancer_features = torch.stack(dancer_outputs, dim=2)  # [batch, seq, dancers, hidden]
        
        # Create mask for valid dancers
        valid_dancers = None
        if dancer_ids is not None:
            valid_dancers = dancer_ids.bool()  # [batch, dancers]
        
        # Apply dynamic graph modeling for dancer interactions
        if multiple_dancers and num_dancers > 1:
            # For each time step
            graph_outputs = []
            graph_adj_matrices = []
            
            for t in range(seq_len):
                # Get features at this time step
                time_features = dancer_features[:, t]  # [batch, dancers, hidden]
                
                # Apply dynamic graph generator
                graph_out, adj_matrix = self.dynamic_graph(time_features, valid_dancers)
                graph_outputs.append(graph_out)
                graph_adj_matrices.append(adj_matrix)
            
            # Stack time steps
            graph_features = torch.stack(graph_outputs, dim=1)  # [batch, seq, dancers, hidden]
            graph_adj = torch.stack(graph_adj_matrices, dim=1)  # [batch, seq, dancers, dancers]
            
            # Combine with individual features
            dancer_features = 0.7 * dancer_features + 0.3 * graph_features
        else:
            # No graph modeling for single dancer
            graph_adj = None
        
        # Final projection
        output_features = self.output_proj(dancer_features)
        
        # Remove dancer dimension for single dancer case
        if not multiple_dancers:
            output_features = output_features.squeeze(2)
            if graph_adj is not None:
                graph_adj = graph_adj.squeeze(2)
        
        # Prepare attention weights output
        if self.fusion_type == "cross_attention":
            attn_weights = torch.stack(dancer_attn_weights, dim=2)  # [batch, seq, dancers, music_seq]
            if not multiple_dancers:
                attn_weights = attn_weights.squeeze(2)
        else:
            attn_weights = None
        
        return {
            "fused_features": output_features,
            "attention_weights": attn_weights,
            "graph_adjacency": graph_adj
        }


def create_model(config):
    """
    Create a DanceFormer model instance based on config.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Initialized DanceFormer model
    """
    model = DanceFormer(
        dance_dim=config.get('dance_feature_dim', 512),
        music_dim=config.get('music_feature_dim', 128),
        hidden_dim=config.get('hidden_dim', 512),
        num_heads=config.get('attention_heads', 8),
        num_layers=config.get('transformer_layers', 4),
        num_scales=config.get('num_scales', 3),
        max_dancers=config.get('max_dancers', 5),
        dropout=config.get('dropout', 0.1),
        fusion_type=config.get('fusion_type', "cross_attention")
    )
    
    return model