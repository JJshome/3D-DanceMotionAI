"""
Music analysis module for dance-music correlation.

This module implements audio processing and feature extraction for
music analysis in the context of dance choreography.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import librosa
from scipy.signal import find_peaks


class AudioFeatureExtractor:
    """
    Extract meaningful features from audio signals for dance-music correlation.
    """
    def __init__(
        self, 
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 20,
        n_chroma: int = 12,
        use_hpss: bool = True,
        normalize: bool = True
    ):
        """
        Initialize AudioFeatureExtractor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            n_mels: Number of mel bands
            n_mfcc: Number of MFCCs to extract
            n_chroma: Number of chroma features
            use_hpss: Whether to use harmonic-percussive source separation
            normalize: Whether to normalize extracted features
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.use_hpss = use_hpss
        self.normalize = normalize
    
    def extract_features(
        self, 
        audio: Union[str, np.ndarray], 
        features: List[str] = ['rhythm', 'timbre', 'harmony'],
        fixed_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract audio features.
        
        Args:
            audio: Audio file path or audio array
            features: List of feature types to extract
            fixed_length: Optional fixed length for all features
            
        Returns:
            Dictionary of extracted features
        """
        # Load audio if path is provided
        if isinstance(audio, str):
            y, sr = librosa.load(audio, sr=self.sample_rate)
        else:
            y = audio
            sr = self.sample_rate
        
        # Apply harmonic-percussive source separation if requested
        if self.use_hpss:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
        else:
            y_harmonic, y_percussive = y, y
        
        # Initialize results dictionary
        results = {}
        
        # Extract rhythm features
        if 'rhythm' in features:
            rhythm_features = self._extract_rhythm_features(y_percussive)
            results.update(rhythm_features)
        
        # Extract timbre features
        if 'timbre' in features:
            timbre_features = self._extract_timbre_features(y)
            results.update(timbre_features)
        
        # Extract harmony features
        if 'harmony' in features:
            harmony_features = self._extract_harmony_features(y_harmonic)
            results.update(harmony_features)
        
        # Adjust feature lengths if needed
        if fixed_length is not None:
            results = self._adjust_feature_lengths(results, fixed_length)
        
        return results
    
    def _extract_rhythm_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract rhythm-related features from audio."""
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Compute tempo and beat frames
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Create beat activation signal
        beat_signal = np.zeros_like(onset_env)
        beat_signal[beats] = 1.0
        
        # Compute rhythm features
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Normalize if requested
        if self.normalize:
            onset_env = (onset_env - onset_env.mean()) / (onset_env.std() + 1e-8)
            tempogram = librosa.util.normalize(tempogram, axis=0)
        
        return {
            'onset_strength': onset_env,
            'beat_signal': beat_signal,
            'tempogram': tempogram,
            'tempo': tempo
        }
    
    def _extract_timbre_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract timbre-related features from audio."""
        # Compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(mel_spectrogram),
            n_mfcc=self.n_mfcc
        )
        
        # Compute delta and delta-delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Compute spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Normalize if requested
        if self.normalize:
            mel_spectrogram_db = librosa.util.normalize(mel_spectrogram_db, axis=1)
            mfcc = librosa.util.normalize(mfcc, axis=1)
            mfcc_delta = librosa.util.normalize(mfcc_delta, axis=1)
            mfcc_delta2 = librosa.util.normalize(mfcc_delta2, axis=1)
            spectral_contrast = librosa.util.normalize(spectral_contrast, axis=1)
        
        return {
            'mel_spectrogram': mel_spectrogram_db,
            'mfcc': mfcc,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'spectral_contrast': spectral_contrast
        }
    
    def _extract_harmony_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract harmony-related features from audio."""
        # Compute chromagram
        chromagram = librosa.feature.chroma_cqt(
            y=y, sr=self.sample_rate, hop_length=self.hop_length, n_chroma=self.n_chroma
        )
        
        # Compute key strength
        key_strength = librosa.feature.tonnetz(
            y=y, sr=self.sample_rate
        )
        
        # Normalize if requested
        if self.normalize:
            chromagram = librosa.util.normalize(chromagram, axis=1)
            key_strength = librosa.util.normalize(key_strength, axis=1)
        
        return {
            'chromagram': chromagram,
            'key_strength': key_strength
        }
    
    def _adjust_feature_lengths(
        self, 
        features: Dict[str, np.ndarray], 
        target_length: int
    ) -> Dict[str, np.ndarray]:
        """Adjust all features to the same length."""
        adjusted_features = {}
        
        for name, feature in features.items():
            # Skip non-time series features (e.g., scalar values like tempo)
            if not isinstance(feature, np.ndarray) or feature.ndim == 0:
                adjusted_features[name] = feature
                continue
            
            # Determine if feature is time-indexed
            if feature.ndim == 1 or (feature.ndim > 1 and feature.shape[1] > 1):
                # Time is the last dimension (for 1D) or second dimension (for 2D+)
                time_axis = 0 if feature.ndim == 1 else 1
                current_length = feature.shape[time_axis]
                
                # Resize feature to target length
                if current_length != target_length:
                    if feature.ndim == 1:
                        # For 1D features (e.g., onset_strength)
                        adjusted_feature = librosa.util.fix_length(
                            feature, size=target_length, axis=0
                        )
                    else:
                        # For 2D features (e.g., mel_spectrogram)
                        if time_axis == 1:  # Time is the second dimension
                            adjusted_feature = np.zeros(
                                (feature.shape[0], target_length), dtype=feature.dtype
                            )
                            # Copy data, zero-padding or truncating as needed
                            min_length = min(current_length, target_length)
                            adjusted_feature[:, :min_length] = feature[:, :min_length]
                        else:
                            adjusted_feature = feature  # No adjustment needed
                else:
                    adjusted_feature = feature
            else:
                adjusted_feature = feature
            
            adjusted_features[name] = adjusted_feature
        
        return adjusted_features


class MusicEncoder(nn.Module):
    """
    Encode music features for dance-music correlation.
    """
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize MusicEncoder.
        
        Args:
            input_dims: Dictionary mapping feature names to dimensions
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of layers in feature encoders
            dropout: Dropout probability
        """
        super(MusicEncoder, self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create feature encoders for each input type
        self.feature_encoders = nn.ModuleDict()
        
        for name, dim in input_dims.items():
            # Create a different encoder structure based on feature type
            if name in ['mel_spectrogram', 'tempogram', 'chromagram']:
                # 2D convolutional encoder for spectrograms
                self.feature_encoders[name] = self._create_conv_encoder(dim)
            elif name in ['mfcc', 'mfcc_delta', 'mfcc_delta2', 'spectral_contrast', 'key_strength']:
                # 1D convolutional encoder for frame-level features
                self.feature_encoders[name] = self._create_1d_encoder(dim)
            elif name in ['onset_strength', 'beat_signal']:
                # Simple MLP for 1D time series
                self.feature_encoders[name] = self._create_mlp_encoder(1)  # 1D time series
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Temporal modeling
        self.temporal_encoder = nn.GRU(
            input_size=output_dim,
            hidden_size=output_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(output_dim * 2, output_dim)  # *2 for bidirectional
    
    def _create_conv_encoder(self, input_dim: int) -> nn.Module:
        """Create a 2D convolutional encoder for spectrogram-like features."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # Pool frequency dimension but keep time
            
            nn.Flatten(start_dim=1, end_dim=2),  # Flatten channel and frequency dims
            nn.Linear(128, self.hidden_dim)
        )
    
    def _create_1d_encoder(self, input_dim: int) -> nn.Module:
        """Create a 1D convolutional encoder for frame-level features."""
        return nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
    
    def _create_mlp_encoder(self, input_dim: int) -> nn.Module:
        """Create a simple MLP encoder for 1D time series."""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode music features.
        
        Args:
            features: Dictionary of music features
            
        Returns:
            Encoded music features [batch_size, seq_len, output_dim]
        """
        batch_size = next(iter(features.values())).size(0)
        seq_len = max(feat.size(1) if feat.dim() > 2 else feat.size(-1) 
                     for feat in features.values() if feat.dim() > 1)
        
        # Process each feature type
        encoded_features = {}
        
        for name, encoder in self.feature_encoders.items():
            if name not in features:
                continue
            
            feat = features[name]
            
            if name in ['mel_spectrogram', 'tempogram', 'chromagram']:
                # Reshape for 2D convolution [batch, channel, freq, time]
                if feat.dim() == 3:  # [batch, freq, time]
                    feat = feat.unsqueeze(1)  # Add channel dimension
                
                # Apply encoder
                encoded = encoder(feat)  # [batch, hidden_dim, time]
                
                # If sequence length doesn't match, resize
                if encoded.size(-1) != seq_len:
                    encoded = F.interpolate(
                        encoded, size=seq_len, mode='linear', align_corners=False
                    )
            
            elif name in ['mfcc', 'mfcc_delta', 'mfcc_delta2', 'spectral_contrast', 'key_strength']:
                # Reshape for 1D convolution [batch, feat_dim, time]
                if feat.dim() == 3:  # [batch, feat_dim, time]
                    # Already in correct format
                    pass
                else:  # [batch, time, feat_dim]
                    feat = feat.transpose(1, 2)
                
                # Apply encoder
                encoded = encoder(feat)  # [batch, hidden_dim, time]
                
                # If sequence length doesn't match, resize
                if encoded.size(-1) != seq_len:
                    encoded = F.interpolate(
                        encoded, size=seq_len, mode='linear', align_corners=False
                    )
            
            elif name in ['onset_strength', 'beat_signal']:
                # Reshape for MLP [batch, time, 1]
                if feat.dim() == 2:  # [batch, time]
                    feat = feat.unsqueeze(-1)  # Add feature dimension
                
                # Apply encoder time-wise
                encoded = []
                for t in range(feat.size(1)):
                    encoded.append(encoder(feat[:, t]))
                encoded = torch.stack(encoded, dim=1)  # [batch, time, hidden_dim]
                
                # If sequence length doesn't match, resize
                if encoded.size(1) != seq_len:
                    encoded = F.interpolate(
                        encoded.transpose(1, 2), size=seq_len, mode='linear', align_corners=False
                    ).transpose(1, 2)
            
            else:
                # Skip unsupported feature types
                continue
            
            encoded_features[name] = encoded
        
        # Concatenate all encoded features
        if len(encoded_features) > 0:
            # Ensure all features have shape [batch, time, hidden_dim]
            for name, feat in encoded_features.items():
                if feat.dim() == 3 and feat.size(1) == self.hidden_dim:
                    # Shape is [batch, hidden_dim, time]
                    encoded_features[name] = feat.transpose(1, 2)
            
            # Concatenate along feature dimension
            concat_features = torch.cat(
                [feat for feat in encoded_features.values()], dim=-1
            )
            
            # Apply fusion
            fused_features = self.fusion(concat_features)
            
            # Apply temporal encoder
            temporal_features, _ = self.temporal_encoder(fused_features)
            
            # Apply output projection
            output_features = self.output_proj(temporal_features)
        else:
            # No features were encoded, return zero tensor
            output_features = torch.zeros(
                batch_size, seq_len, self.output_dim, device=next(self.parameters()).device
            )
        
        return output_features


class BeatAndTempoAnalyzer:
    """
    Analyze beats and tempo from audio for dance synchronization.
    """
    def __init__(
        self, 
        sample_rate: int = 22050,
        hop_length: int = 512,
        min_tempo: float = 60.0,
        max_tempo: float = 200.0
    ):
        """
        Initialize BeatAndTempoAnalyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            hop_length: Hop length for feature extraction
            min_tempo: Minimum tempo in BPM
            max_tempo: Maximum tempo in BPM
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_tempo = min_tempo
        self.max_tempo = max_tempo
    
    def analyze(
        self, 
        audio: Union[str, np.ndarray],
        return_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze beats and tempo from audio.
        
        Args:
            audio: Audio file path or audio array
            return_audio: Whether to include audio data in results
            
        Returns:
            Dictionary with beat and tempo information
        """
        # Load audio if path is provided
        if isinstance(audio, str):
            y, sr = librosa.load(audio, sr=self.sample_rate)
        else:
            y = audio
            sr = self.sample_rate
        
        # Extract percussive component for better beat detection
        y_percussive = librosa.effects.harmonic(y)
        
        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(
            y=y_percussive, sr=sr, hop_length=self.hop_length
        )
        
        # Dynamic tempo estimation using tempogram
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length,
            win_length=384  # About 8 seconds at 22050Hz with hop_length=512
        )
        
        # Estimate global tempo (static)
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length,
            trim=False, start_bpm=120.0,
            tightness=100,
            bpm=self.max_tempo
        )
        
        # Dynamic tempo estimation
        tempo_estimates = []
        window_size = 1000  # About 23 seconds at 22050Hz with hop_length=512
        hop_size = 500  # About 11.5 seconds hop
        
        for i in range(0, len(onset_env), hop_size):
            window_end = min(i + window_size, len(onset_env))
            if window_end - i < window_size // 2:  # Skip small windows at the end
                continue
            
            local_onset_env = onset_env[i:window_end]
            local_tempo, _ = librosa.beat.beat_track(
                onset_envelope=local_onset_env, sr=sr, hop_length=self.hop_length,
                trim=False, bpm=tempo  # Start with global tempo as prior
            )
            tempo_estimates.append((i, window_end, local_tempo))
        
        # Compute beat times using beat tracker
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length,
            trim=False, bpm=tempo  # Use global tempo estimate
        )
        
        # Convert frame indices to time
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # Compute beat intervals (time between consecutive beats)
        beat_intervals = np.diff(beat_times)
        
        # Detect rhythm changes
        rhythm_changes = []
        if len(beat_intervals) > 4:
            # Compute moving average of beat intervals
            window_size = 8  # Average over 8 beats
            moving_avg = np.convolve(beat_intervals, np.ones(window_size)/window_size, mode='valid')
            
            # Compute derivative of moving average
            moving_avg_diff = np.diff(moving_avg)
            
            # Find peaks in the absolute derivative (significant changes)
            peaks, _ = find_peaks(np.abs(moving_avg_diff), height=0.05, distance=16)
            
            # Convert peak indices to beat indices
            for peak in peaks:
                # Add window_size//2 to account for moving average offset
                beat_idx = peak + window_size//2
                if beat_idx < len(beat_times):
                    rhythm_changes.append({
                        'time': beat_times[beat_idx],
                        'type': 'acceleration' if moving_avg_diff[peak] < 0 else 'deceleration',
                        'magnitude': np.abs(moving_avg_diff[peak])
                    })
        
        # Compute beat intensity
        beat_intensity = onset_env[beats]
        
        # Find accented beats
        accented_beats = []
        if len(beat_intensity) > 0:
            # Normalize beat intensity
            normalized_intensity = (beat_intensity - beat_intensity.mean()) / (beat_intensity.std() + 1e-8)
            
            # Find peaks in normalized intensity
            threshold = 1.0  # Threshold for accent detection
            for i, intensity in enumerate(normalized_intensity):
                if intensity > threshold:
                    accented_beats.append({
                        'time': beat_times[i],
                        'intensity': float(intensity)
                    })
        
        # Create beat and bar structure
        beats_per_bar = 4  # Assuming 4/4 time signature (most common in pop music)
        bar_starts = beat_times[::beats_per_bar]
        
        # Prepare results
        results = {
            'tempo': float(tempo),
            'dynamic_tempo': tempo_estimates,
            'beat_times': beat_times,
            'beat_intervals': beat_intervals,
            'bar_starts': bar_starts,
            'rhythm_changes': rhythm_changes,
            'accented_beats': accented_beats,
            'beats_per_bar': beats_per_bar
        }
        
        # Include audio data if requested
        if return_audio:
            results['audio'] = y
            results['sample_rate'] = sr
        
        return results


class MusicAnalyzer:
    """
    Complete music analysis module for dance-music correlation.
    """
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        feature_output_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize MusicAnalyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            hop_length: Hop length for feature extraction
            feature_output_dim: Dimension of encoded features
            device: Device to use (cuda or cpu)
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.device = device
        
        # Initialize feature extractor
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        
        # Initialize beat and tempo analyzer
        self.beat_analyzer = BeatAndTempoAnalyzer(
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        
        # Define input feature dimensions
        self.input_dims = {
            'mel_spectrogram': 128,  # n_mels
            'mfcc': 20,  # n_mfcc
            'chromagram': 12,  # n_chroma
            'tempogram': 384,  # win_length
            'onset_strength': 1,
            'beat_signal': 1
        }
        
        # Initialize feature encoder
        self.feature_encoder = MusicEncoder(
            input_dims=self.input_dims,
            hidden_dim=256,
            output_dim=feature_output_dim
        )
        
        # Move models to device
        self.feature_encoder = self.feature_encoder.to(device)
    
    def analyze(
        self, 
        audio: Union[str, np.ndarray],
        return_features: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze music for dance correlation.
        
        Args:
            audio: Audio file path or audio array
            return_features: Whether to include raw features in results
            
        Returns:
            Dictionary with music analysis results
        """
        # Extract audio features
        features = self.feature_extractor.extract_features(audio)
        
        # Analyze beats and tempo
        beat_analysis = self.beat_analyzer.analyze(audio)
        
        # Convert features to tensors
        feature_tensors = {}
        for name, feat in features.items():
            # Skip non-array features
            if not isinstance(feat, np.ndarray):
                continue
            
            # Convert to tensor and add batch dimension
            if feat.ndim == 1:
                # 1D features
                tensor = torch.from_numpy(feat).float().unsqueeze(0)  # [1, time]
            elif feat.ndim == 2:
                # 2D features (freq, time)
                tensor = torch.from_numpy(feat).float().unsqueeze(0)  # [1, freq, time]
            else:
                # Unsupported dimension
                continue
            
            feature_tensors[name] = tensor.to(self.device)
        
        # Encode features
        with torch.no_grad():
            encoded_features = self.feature_encoder(feature_tensors)
        
        # Convert to numpy
        encoded_features = encoded_features.cpu().numpy()
        
        # Prepare results
        results = {
            'encoded_features': encoded_features,
            'beats': beat_analysis
        }
        
        # Include raw features if requested
        if return_features:
            results['raw_features'] = features
        
        return results
    
    def synchronize_music_to_video(
        self, 
        audio_features: Dict[str, Any],
        video_fps: float,
        video_length: int
    ) -> Dict[str, Any]:
        """
        Synchronize music analysis results with video frames.
        
        Args:
            audio_features: Music analysis results from analyze()
            video_fps: Video frames per second
            video_length: Number of video frames
            
        Returns:
            Dictionary with synchronized music features
        """
        # Get beat times
        beat_times = audio_features['beats']['beat_times']
        bar_starts = audio_features['beats']['bar_starts']
        
        # Convert times to frame indices
        beat_frames = np.floor(beat_times * video_fps).astype(int)
        bar_frames = np.floor(bar_starts * video_fps).astype(int)
        
        # Clip to video length
        beat_frames = beat_frames[beat_frames < video_length]
        bar_frames = bar_frames[bar_frames < video_length]
        
        # Create frame-level beat signal
        beat_signal = np.zeros(video_length)
        beat_signal[beat_frames] = 1.0
        
        # Create frame-level bar signal
        bar_signal = np.zeros(video_length)
        bar_signal[bar_frames] = 1.0
        
        # Align encoded features with video frames
        encoded_features = audio_features['encoded_features'][0]  # Remove batch dim
        
        # Compute frame indices for each feature
        feature_times = np.arange(encoded_features.shape[0]) * (self.hop_length / self.sample_rate)
        feature_frames = np.floor(feature_times * video_fps).astype(int)
        
        # Clip to video length and remove duplicates
        valid_indices = feature_frames < video_length
        feature_frames = feature_frames[valid_indices]
        unique_frames, unique_indices = np.unique(feature_frames, return_index=True)
        
        # Initialize frame-aligned features
        frame_features = np.zeros((video_length, encoded_features.shape[1]))
        
        # Fill valid frames
        valid_features = encoded_features[valid_indices][unique_indices]
        frame_features[unique_frames] = valid_features
        
        # Interpolate missing frames
        for i in range(1, video_length):
            if np.all(frame_features[i] == 0):
                frame_features[i] = frame_features[i-1]
        
        # Prepare synchronized results
        sync_results = {
            'beat_signal': beat_signal,
            'bar_signal': bar_signal,
            'frame_features': frame_features,
            'beat_frames': beat_frames,
            'bar_frames': bar_frames
        }
        
        return sync_results


def create_music_analyzer(config):
    """
    Create a MusicAnalyzer instance based on config.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Initialized MusicAnalyzer
    """
    analyzer = MusicAnalyzer(
        sample_rate=config.get('sample_rate', 22050),
        hop_length=config.get('hop_length', 512),
        feature_output_dim=config.get('music_feature_dim', 128),
        device=config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    )
    
    return analyzer