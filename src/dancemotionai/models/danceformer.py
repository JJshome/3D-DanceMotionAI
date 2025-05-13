"""
DanceFormer: Multimodal analysis network for dance and music integration.

This module implements the DanceFormer with cross-modal attention mechanism,
multi-scale transformer, and dynamic graph generation for dance analysis.

Key Components:
1. Cross-Modal Attention: Aligns dance movements with musical elements
2. Multi-Scale Transformer: Processes dance at frame, phrase, and sequence levels
3. Dynamic Graph Generation: Creates adaptive relationship graphs for choreography analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union

class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention module for aligning dance movements with musical elements.
    """
    
    def __init__(self, 
                 dance_dim: int = 512, 
                 music_dim: int = 128, 
                 output_dim: int = 512, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize the Cross-Modal Attention module.
        
        Args:
            dance_dim: Dance feature dimension
            music_dim: Music feature dimension
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossModalAttention, self).__init__()
        
        # Projection layers
        self.dance_projection = nn.Linear(dance_dim, output_dim)
        self.music_projection = nn.Linear(music_dim, output_dim)
        
        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
                dance_features: torch.Tensor, 
                music_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Cross-Modal Attention module.
        
        Args:
            dance_features: Dance features of shape (batch_size, dance_seq_len, dance_dim)
            music_features: Music features of shape (batch_size, music_seq_len, music_dim)
            mask: Optional attention mask
            
        Returns:
            Aligned features of shape (batch_size, dance_seq_len, output_dim)
        """
        # Project features to common space
        dance_proj = self.dance_projection(dance_features)
        music_proj = self.music_projection(music_features)
        
        # Apply cross-attention (dance as query, music as key/value)
        attn_output, attn_weights = self.cross_attention(
            query=dance_proj,
            key=music_proj,
            value=music_proj,
            key_padding_mask=mask
        )
        
        # Concatenate original dance features with attended music features
        concat_features = torch.cat([dance_proj, attn_output], dim=2)
        
        # Apply output layer
        output = self.output_layer(concat_features)
        
        return output, attn_weights


class MultiScaleTransformer(nn.Module):
    """
    Multi-Scale Transformer for processing dance sequences at different temporal scales.
    """
    
    def __init__(self, 
                 input_dim: int = 512, 
                 hidden_dim: int = 1024, 
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 num_scales: int = 3):
        """
        Initialize the Multi-Scale Transformer module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            num_scales: Number of temporal scales
        """
        super(MultiScaleTransformer, self).__init__()
        
        # Transformer encoders for each scale
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=num_layers
            ) for _ in range(num_scales)
        ])
        
        # Temporal pooling layers for creating multi-scale inputs
        self.temporal_pools = nn.ModuleList([
            nn.Identity(), 
            TemporalPooling(pool_size=2),
            TemporalPooling(pool_size=4)
        ])
        
        # Cross-scale attention for information exchange between scales
        self.cross_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_scales - 1)
        ])
        
        # Fusion layer
        self.fusion_layer = nn.Linear(input_dim * num_scales, input_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_dim, dropout)
        
        # Number of scales
        self.num_scales = num_scales
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Multi-Scale Transformer.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Multi-scale features of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, feat_dim = x.size()
        
        # Generate multi-scale inputs
        multi_scale_inputs = [pool(x) for pool in self.temporal_pools]
        
        # Apply positional encoding to each scale
        multi_scale_inputs = [self.pos_encoding(x_scale) for x_scale in multi_scale_inputs]
        
        # Apply transformer to each scale
        multi_scale_outputs = [
            self.transformers[i](x_scale) 
            for i, x_scale in enumerate(multi_scale_inputs)
        ]
        
        # Cross-scale attention for information exchange
        for i in range(self.num_scales - 1):
            # Upsample coarser scale to match finer scale
            coarse_scale = multi_scale_outputs[i + 1]
            fine_scale = multi_scale_outputs[i]
            
            # Interpolate coarse scale to match fine scale length
            coarse_interp = F.interpolate(
                coarse_scale.transpose(1, 2),
                size=fine_scale.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
            # Apply cross-scale attention
            attn_output, _ = self.cross_scale_attention[i](
                query=fine_scale,
                key=coarse_interp,
                value=coarse_interp
            )
            
            # Update fine scale with cross-scale attention
            multi_scale_outputs[i] = fine_scale + attn_output
        
        # Upsample all scales to match original sequence length
        aligned_outputs = []
        for i, output in enumerate(multi_scale_outputs):
            if i == 0:
                aligned_outputs.append(output)
            else:
                aligned = F.interpolate(
                    output.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                aligned_outputs.append(aligned)
        
        # Concatenate aligned outputs
        concat_output = torch.cat(aligned_outputs, dim=2)
        
        # Apply fusion layer
        output = self.fusion_layer(concat_output)
        
        return output


class TemporalPooling(nn.Module):
    """
    Temporal pooling layer for creating multi-scale inputs.
    """
    
    def __init__(self, pool_size: int = 2):
        """
        Initialize the TemporalPooling layer.
        
        Args:
            pool_size: Pooling window size
        """
        super(TemporalPooling, self).__init__()
        self.pool_size = pool_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TemporalPooling layer.
        
        Args:
            x: Input features of shape (batch_size, seq_len, feat_dim)
            
        Returns:
            Pooled features of shape (batch_size, seq_len // pool_size, feat_dim)
        """
        batch_size, seq_len, feat_dim = x.size()
        
        # Handle case where sequence length is not divisible by pool_size
        if seq_len % self.pool_size != 0:
            # Pad the sequence
            pad_len = self.pool_size - (seq_len % self.pool_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len += pad_len
        
        # Reshape for pooling
        x = x.reshape(batch_size, seq_len // self.pool_size, self.pool_size, feat_dim)
        
        # Apply pooling (average)
        x = torch.mean(x, dim=2)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the PositionalEncoding module.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create position encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PositionalEncoding module.
        
        Args:
            x: Input features of shape (batch_size, seq_len, d_model)
            
        Returns:
            Position-encoded features of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DynamicGraphGenerator(nn.Module):
    """
    Dynamic Graph Generator for creating adaptive relationship graphs between dancers.
    """
    
    def __init__(self, 
                 input_dim: int = 512, 
                 hidden_dim: int = 256, 
                 max_dancers: int = 10,
                 threshold_init: float = 0.5):
        """
        Initialize the DynamicGraphGenerator module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            max_dancers: Maximum number of dancers
            threshold_init: Initial threshold for edge creation
        """
        super(DynamicGraphGenerator, self).__init__()
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
        
        # Maximum number of dancers
        self.max_dancers = max_dancers
        
    def forward(self, dancer_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DynamicGraphGenerator.
        
        Args:
            dancer_features: Dancer features of shape (batch_size, num_dancers, seq_len, input_dim)
            
        Returns:
            Adjacency matrix of shape (batch_size, num_dancers, num_dancers)
        """
        batch_size, num_dancers, seq_len, feat_dim = dancer_features.size()
        
        # Compute dancer representations by averaging over sequence length
        dancer_repr = torch.mean(dancer_features, dim=2)  # (batch_size, num_dancers, input_dim)
        
        # Initialize adjacency matrix
        adj_matrix = torch.zeros(batch_size, num_dancers, num_dancers, device=dancer_features.device)
        
        # Compute pairwise similarities
        for i in range(num_dancers):
            for j in range(i+1, num_dancers):  # Upper triangular only (undirected graph)
                dancer_i = dancer_repr[:, i, :]  # (batch_size, input_dim)
                dancer_j = dancer_repr[:, j, :]  # (batch_size, input_dim)
                
                # Concatenate features
                pair_features = torch.cat([dancer_i, dancer_j], dim=1)  # (batch_size, input_dim*2)
                
                # Compute similarity
                similarity = self.similarity_net(pair_features).squeeze(-1)  # (batch_size,)
                
                # Create edge if similarity exceeds threshold
                edge = (similarity > self.threshold).float()
                
                # Set adjacency matrix values (symmetric)
                adj_matrix[:, i, j] = edge
                adj_matrix[:, j, i] = edge
        
        # Add self-loops
        for i in range(num_dancers):
            adj_matrix[:, i, i] = 1.0
            
        return adj_matrix


class GroupChoreographyAnalyzer(nn.Module):
    """
    Group Choreography Analyzer for analyzing interactions between multiple dancers.
    """
    
    def __init__(self, 
                 input_dim: int = 512, 
                 hidden_dim: int = 256, 
                 max_dancers: int = 10):
        """
        Initialize the GroupChoreographyAnalyzer module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            max_dancers: Maximum number of dancers
        """
        super(GroupChoreographyAnalyzer, self).__init__()
        
        # Dynamic graph generator
        self.graph_generator = DynamicGraphGenerator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_dancers=max_dancers
        )
        
        # Graph convolutional layers
        self.graph_conv1 = GraphConv(input_dim, hidden_dim)
        self.graph_conv2 = GraphConv(hidden_dim, input_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, dancer_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GroupChoreographyAnalyzer.
        
        Args:
            dancer_features: Dancer features of shape (batch_size, num_dancers, seq_len, input_dim)
            
        Returns:
            Dictionary containing updated dancer features and adjacency matrix
        """
        batch_size, num_dancers, seq_len, feat_dim = dancer_features.size()
        
        # Generate dynamic graph
        adj_matrix = self.graph_generator(dancer_features)
        
        # Process each frame
        updated_features = []
        for t in range(seq_len):
            # Extract features for current frame
            frame_features = dancer_features[:, :, t, :]  # (batch_size, num_dancers, feat_dim)
            
            # Apply graph convolution
            graph_out1 = self.graph_conv1(frame_features, adj_matrix)
            graph_out2 = self.graph_conv2(graph_out1, adj_matrix)
            
            # Residual connection
            frame_updated = frame_features + graph_out2
            
            # Final projection
            frame_updated = self.output_projection(frame_updated)
            
            updated_features.append(frame_updated)
        
        # Stack along temporal dimension
        updated_dancer_features = torch.stack(updated_features, dim=2)  # (batch_size, num_dancers, seq_len, feat_dim)
        
        return {
            'dancer_features': updated_dancer_features,
            'adjacency_matrix': adj_matrix
        }


class GraphConv(nn.Module):
    """
    Graph Convolutional Layer.
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


class MusicFeatureExtractor(nn.Module):
    """
    Music Feature Extractor for extracting musical features from audio.
    """
    
    def __init__(self, 
                 input_dim: int = 128, 
                 hidden_dim: int = 256, 
                 output_dim: int = 512,
                 num_layers: int = 2):
        """
        Initialize the MusicFeatureExtractor module.
        
        Args:
            input_dim: Input feature dimension (MFCC + chroma features)
            hidden_dim: Hidden feature dimension
            output_dim: Output feature dimension
            num_layers: Number of LSTM layers
        """
        super(MusicFeatureExtractor, self).__init__()
        
        # Bidirectional LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output projection
        self.projection = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MusicFeatureExtractor.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Music features of shape (batch_size, seq_len, output_dim)
        """
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply projection
        output = self.projection(lstm_out)
        
        return output
    
    @staticmethod
    def extract_features_from_audio(audio_path: str, sr: int = 22050, hop_length: int = 512) -> np.ndarray:
        """
        Extract MFCC and chroma features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            sr: Sample rate
            hop_length: Hop length for feature extraction
            
        Returns:
            Array of shape (seq_len, input_dim) containing audio features
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        
        # Extract onset strength
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset = onset.reshape(1, -1)
        
        # Extract tempo and beat information
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_frames = np.zeros(mfcc.shape[1])
        beat_frames[beats] = 1
        beat_frames = beat_frames.reshape(1, -1)
        
        # Concatenate features
        features = np.vstack([mfcc, chroma, onset, beat_frames])
        
        # Transpose to get (seq_len, input_dim)
        features = features.T
        
        return features


class DanceFormer(nn.Module):
    """
    DanceFormer: Multimodal analysis network for dance and music integration.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the DanceFormer model.
        
        Args:
            config: Configuration dictionary
        """
        super(DanceFormer, self).__init__()
        
        self.config = config or {}
        
        # Get configuration parameters
        self.dance_dim = self.config.get('dance_feature_dim', 512)
        self.music_dim = self.config.get('music_feature_dim', 128)
        self.output_dim = self.config.get('output_dim', 512)
        self.hidden_dim = self.config.get('hidden_dim', 1024)
        self.num_heads = self.config.get('num_heads', 8)
        self.num_layers = self.config.get('transformer_layers', 4)
        self.num_scales = self.config.get('num_scales', 3)
        self.fusion_type = self.config.get('fusion_type', 'cross_attention')
        self.use_dynamic_graph = self.config.get('use_dynamic_graph', True)
        self.max_dancers = self.config.get('max_dancers', 10)
        
        # Cross-modal attention module
        self.cross_modal_attention = CrossModalAttention(
            dance_dim=self.dance_dim,
            music_dim=self.music_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads
        )
        
        # Multi-scale transformer
        self.multi_scale_transformer = MultiScaleTransformer(
            input_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_scales=self.num_scales
        )
        
        # Group choreography analyzer (optional)
        if self.use_dynamic_graph:
            self.group_analyzer = GroupChoreographyAnalyzer(
                input_dim=self.output_dim,
                hidden_dim=self.hidden_dim // 2,
                max_dancers=self.max_dancers
            )
        
        # Music feature extractor
        self.music_feature_extractor = MusicFeatureExtractor(
            input_dim=self.music_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=self.output_dim,
            num_layers=2
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.output_dim, self.output_dim)
        
    def forward(self, 
                dance_features: torch.Tensor, 
                music_features: torch.Tensor,
                dancer_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the DanceFormer.
        
        Args:
            dance_features: Dance features of shape (batch_size, seq_len, dance_dim)
                            or (batch_size, num_dancers, seq_len, dance_dim) for group choreography
            music_features: Music features of shape (batch_size, music_seq_len, music_dim)
            dancer_ids: Optional dancer IDs for group choreography
            
        Returns:
            Dictionary containing analysis results
        """
        # Check if input is for group choreography
        is_group = len(dance_features.shape) == 4
        
        # Process music features
        processed_music = self.music_feature_extractor(music_features)
        
        # Process individual dancers or single dancer
        if is_group:
            batch_size, num_dancers, seq_len, _ = dance_features.shape
            
            # Reshape for processing each dancer
            dance_flat = dance_features.reshape(batch_size * num_dancers, seq_len, -1)
            music_expanded = processed_music.unsqueeze(1).expand(-1, num_dancers, -1, -1)
            music_flat = music_expanded.reshape(batch_size * num_dancers, -1, self.output_dim)
            
            # Apply cross-modal attention
            cross_output, attn_weights = self.cross_modal_attention(dance_flat, music_flat)
            
            # Apply multi-scale transformer
            ms_output = self.multi_scale_transformer(cross_output)
            
            # Reshape back to group format
            ms_output = ms_output.reshape(batch_size, num_dancers, seq_len, -1)
            
            # Apply group choreography analysis if enabled
            if self.use_dynamic_graph:
                group_results = self.group_analyzer(ms_output)
                dancer_features = group_results['dancer_features']
                adjacency_matrix = group_results['adjacency_matrix']
            else:
                dancer_features = ms_output
                adjacency_matrix = None
                
            # Final projection
            output = self.output_projection(dancer_features)
            
            # Prepare results
            results = {
                'dancer_features': output,
                'attention_weights': attn_weights,
                'adjacency_matrix': adjacency_matrix
            }
            
        else:
            # Single dancer processing
            cross_output, attn_weights = self.cross_modal_attention(dance_features, processed_music)
            ms_output = self.multi_scale_transformer(cross_output)
            output = self.output_projection(ms_output)
            
            # Prepare results
            results = {
                'dancer_features': output,
                'attention_weights': attn_weights
            }
            
        return results
    
    def analyze(self, 
                pose_results: Dict, 
                music_path: str, 
                dancer_ids: Optional[List[int]] = None) -> Dict:
        """
        Analyze dance and music data.
        
        Args:
            pose_results: Results from pose estimation
            music_path: Path to the music file
            dancer_ids: Optional list of dancer IDs for group choreography
            
        Returns:
            Dictionary containing analysis results
        """
        import torch
        import numpy as np
        
        # Extract poses and metadata
        poses = pose_results['poses']
        fps = pose_results['fps']
        num_frames = pose_results['num_frames']
        
        # Convert poses to torch tensor
        poses_tensor = torch.tensor(poses, dtype=torch.float32)
        
        # Add batch dimension if not already present
        if len(poses_tensor.shape) == 3:  # (num_frames, num_joints, 3)
            poses_tensor = poses_tensor.unsqueeze(0)  # (1, num_frames, num_joints, 3)
        
        # Prepare dance features
        batch_size, seq_len, num_joints, _ = poses_tensor.shape
        dance_features = poses_tensor.reshape(batch_size, seq_len, -1)  # Flatten joints
        
        # Extract music features
        music_features_np = MusicFeatureExtractor.extract_features_from_audio(
            audio_path=music_path,
            sr=22050,
            hop_length=512
        )
        
        # Convert to torch tensor
        music_features = torch.tensor(music_features_np, dtype=torch.float32).unsqueeze(0)
        
        # Match music and dance sequence lengths
        music_seq_len = music_features.shape[1]
        dance_seq_len = dance_features.shape[1]
        
        # Adjust music length to match dance length (simple approach)
        if music_seq_len > dance_seq_len:
            # Downsample music
            ratio = dance_seq_len / music_seq_len
            indices = np.round(np.arange(0, music_seq_len) * ratio).astype(int)
            indices = np.clip(indices, 0, dance_seq_len - 1)
            music_features = music_features[:, indices, :]
        elif music_seq_len < dance_seq_len:
            # Upsample music
            music_features = F.interpolate(
                music_features.transpose(1, 2),
                size=dance_seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Group choreography processing if dancer_ids provided
        if dancer_ids is not None:
            # Reshape data for group processing
            num_dancers = len(dancer_ids)
            group_dance_features = []
            
            for dancer_id in dancer_ids:
                # Assumes pose_results contains separate poses for each dancer
                dancer_poses = pose_results['poses_by_dancer'][dancer_id]
                dancer_tensor = torch.tensor(dancer_poses, dtype=torch.float32).unsqueeze(0)
                group_dance_features.append(dancer_tensor)
            
            # Stack along dancer dimension
            group_dance_features = torch.cat(group_dance_features, dim=1)
            
            # Forward pass with group data
            with torch.no_grad():
                results = self.forward(
                    dance_features=group_dance_features,
                    music_features=music_features,
                    dancer_ids=torch.tensor(dancer_ids)
                )
                
        else:
            # Single dancer processing
            with torch.no_grad():
                results = self.forward(
                    dance_features=dance_features,
                    music_features=music_features
                )
        
        # Convert torch tensors to numpy for easier handling
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.detach().cpu().numpy()
        
        # Add metadata
        results.update({
            'fps': fps,
            'num_frames': num_frames,
            'music_path': music_path
        })
        
        return results
