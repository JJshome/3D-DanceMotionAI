"""
Integration demo for the DanceMotionAI framework.

This script demonstrates the complete pipeline of the DanceMotionAI system,
including 3D pose estimation, hand tracking, music analysis, and 
choreography similarity detection.
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import librosa
import librosa.display

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import DanceMotionAI modules
from models.dancehrnet import create_model as create_dancehrnet
from models.hands_4d import Hands4D
from models.danceformer import create_model as create_danceformer
from models.dance_dtw import create_model as create_dance_dtw
from models.music_analysis import create_music_analyzer
from utils.visualization import (
    plot_3d_pose, 
    visualize_hand_keypoints, 
    plot_similarity_heatmap,
    create_pose_animation
)
from utils.metrics import compute_dance_metrics


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_models(config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Initialize all models based on configuration."""
    # Create DanceHRNet model
    dancehrnet = create_dancehrnet(config['pose_estimation'])
    
    # Create 4DHands model
    hands_4d = Hands4D(config['hands_4d'])
    
    # Create DanceFormer model
    danceformer = create_danceformer(config['multimodal_analysis'])
    
    # Create DanceDTW model
    dance_dtw = create_dance_dtw(config['similarity_analysis'])
    
    # Create Music Analyzer
    music_analyzer = create_music_analyzer(config['multimodal_analysis'])
    
    # Move models to device
    dancehrnet = dancehrnet.to(device)
    hands_4d = hands_4d.to(device)
    danceformer = danceformer.to(device)
    dance_dtw = dance_dtw.to(device)
    
    # Set models to evaluation mode
    dancehrnet.eval()
    hands_4d.eval()
    danceformer.eval()
    dance_dtw.eval()
    
    return {
        'dancehrnet': dancehrnet,
        'hands_4d': hands_4d,
        'danceformer': danceformer,
        'dance_dtw': dance_dtw,
        'music_analyzer': music_analyzer
    }


def load_video(video_path, max_frames=None):
    """Load video frames from file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit number of frames if specified
    if max_frames is not None and max_frames < total_frames:
        frames_to_read = max_frames
    else:
        frames_to_read = total_frames
    
    # Read frames
    frames = []
    for i in range(frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    # Release video file
    cap.release()
    
    # Check if any frames were read
    if not frames:
        raise RuntimeError(f"No frames could be read from video: {video_path}")
    
    # Convert to numpy array
    frames = np.array(frames)
    
    return frames, fps, (width, height)


def extract_audio(video_path, output_path=None, sample_rate=22050):
    """Extract audio from video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output path if not provided
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '.wav'
    
    # Extract audio using ffmpeg
    import subprocess
    cmd = [
        'ffmpeg', '-i', video_path, 
        '-vn', '-acodec', 'pcm_s16le', 
        '-ar', str(sample_rate), '-ac', '1',
        '-y', output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        return None
    
    # Load extracted audio
    y, sr = librosa.load(output_path, sr=sample_rate)
    
    return y, sr


def preprocess_frames(frames, target_size=(512, 512)):
    """Preprocess video frames for model input."""
    processed_frames = []
    
    for frame in frames:
        # Resize frame
        frame_resized = cv2.resize(frame, target_size)
        
        # Normalize pixel values
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Standardize using ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_normalized = (frame_normalized - mean) / std
        
        # Convert to channel-first format
        frame_normalized = frame_normalized.transpose(2, 0, 1)
        
        processed_frames.append(frame_normalized)
    
    # Stack frames
    processed_frames = np.array(processed_frames)
    
    return processed_frames


def process_video(frames, models, device, batch_size=4):
    """Process video frames to extract 3D poses."""
    num_frames = len(frames)
    dancehrnet = models['dancehrnet']
    
    # Process frames in batches
    all_poses_3d = []
    all_poses_2d = []
    all_left_hand_poses = []
    all_right_hand_poses = []
    
    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(start_idx + batch_size, num_frames)
        batch_frames = frames[start_idx:end_idx]
        
        # Convert to torch tensor
        batch_tensor = torch.from_numpy(batch_frames).float().to(device)
        
        # Process batch
        with torch.no_grad():
            outputs = dancehrnet(batch_tensor)
        
        # Extract poses
        poses_3d = outputs['poses_3d'].cpu().numpy()
        poses_2d = outputs['poses_2d'].cpu().numpy()
        left_hand_poses = outputs['left_hand_poses'].cpu().numpy()
        right_hand_poses = outputs['right_hand_poses'].cpu().numpy()
        
        all_poses_3d.append(poses_3d)
        all_poses_2d.append(poses_2d)
        all_left_hand_poses.append(left_hand_poses)
        all_right_hand_poses.append(right_hand_poses)
    
    # Concatenate batches
    all_poses_3d = np.concatenate(all_poses_3d, axis=0)
    all_poses_2d = np.concatenate(all_poses_2d, axis=0)
    all_left_hand_poses = np.concatenate(all_left_hand_poses, axis=0)
    all_right_hand_poses = np.concatenate(all_right_hand_poses, axis=0)
    
    return {
        'poses_3d': all_poses_3d,
        'poses_2d': all_poses_2d,
        'left_hand_poses': all_left_hand_poses,
        'right_hand_poses': all_right_hand_poses
    }


def analyze_music(audio, models, sr):
    """Analyze music for dance correlation."""
    music_analyzer = models['music_analyzer']
    
    # Analyze music
    music_analysis = music_analyzer.analyze(audio)
    
    return music_analysis


def correlate_dance_and_music(pose_data, music_analysis, fps, models, device, batch_size=4):
    """Correlate dance movements with music features."""
    danceformer = models['danceformer']
    
    # Get poses and music features
    poses_3d = pose_data['poses_3d']
    music_features = music_analysis['encoded_features']
    
    # Convert to torch tensors
    poses_tensor = torch.from_numpy(poses_3d).float().to(device)
    music_tensor = torch.from_numpy(music_features).float().to(device)
    
    # Process in batches
    all_fused_features = []
    
    for start_idx in range(0, len(poses_3d), batch_size):
        end_idx = min(start_idx + batch_size, len(poses_3d))
        batch_poses = poses_tensor[start_idx:end_idx]
        
        # Determine music range corresponding to pose batch
        # This is a simplification - a more sophisticated approach would use precise alignment
        start_time = start_idx / fps
        end_time = end_idx / fps
        start_frame = int(start_time * music_features.shape[1] / (len(poses_3d) / fps))
        end_frame = int(end_time * music_features.shape[1] / (len(poses_3d) / fps))
        batch_music = music_tensor[:, start_frame:end_frame]
        
        # Process batch
        with torch.no_grad():
            outputs = danceformer(batch_poses, batch_music)
        
        # Extract fused features
        fused_features = outputs['fused_features'].cpu().numpy()
        all_fused_features.append(fused_features)
    
    # Concatenate batches
    all_fused_features = np.concatenate(all_fused_features, axis=0)
    
    return {
        'fused_features': all_fused_features,
        'attention_weights': None  # Not using attention weights for simplicity
    }


def compare_choreographies(pose_data1, pose_data2, models, device):
    """Compare two choreographies for similarity."""
    dance_dtw = models['dance_dtw']
    
    # Get poses
    poses_3d_1 = pose_data1['poses_3d']
    poses_3d_2 = pose_data2['poses_3d']
    
    # Convert to torch tensors
    poses_tensor1 = torch.from_numpy(poses_3d_1).float().to(device)
    poses_tensor2 = torch.from_numpy(poses_3d_2).float().to(device)
    
    # Process with DTW
    with torch.no_grad():
        similarity_results = dance_dtw(poses_tensor1, poses_tensor2, compute_path=True)
    
    # Extract results
    similarity_score = similarity_results['similarity_score'].cpu().numpy()
    is_similar = similarity_results['is_similar'].cpu().numpy()
    
    return {
        'similarity_score': similarity_score,
        'is_similar': is_similar,
        'dtw_matrices': similarity_results['dtw_matrices'],
        'optimal_paths': similarity_results['optimal_paths']
    }


def visualize_results(frames, pose_data, music_analysis, correlation_results=None, similarity_results=None, config=None):
    """Visualize analysis results."""
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 3D pose
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    frame_idx = min(30, len(pose_data['poses_3d']) - 1)  # Use 30th frame or last frame
    plot_3d_pose(pose_data['poses_3d'][frame_idx], ax=ax1, title="3D Pose")
    
    # Plot hand pose
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    hand_connections = []
    for finger in range(5):  # 5 fingers
        base = finger * 4  # Each finger has 4 joints (including base)
        for j in range(3):  # Connect the 4 joints
            hand_connections.append((base + j, base + j + 1))
    
    # Connect finger bases to palm
    palm = 20  # Last joint is palm
    for finger in range(5):
        base = finger * 4
        hand_connections.append((base, palm))
    
    visualize_hand_keypoints(
        pose_data['left_hand_poses'][frame_idx][0],  # Assuming first hand is left
        hand_connections,
        ax=ax2,
        title="Left Hand Pose"
    )
    
    # Plot music features
    ax3 = fig.add_subplot(2, 3, 3)
    if 'raw_features' in music_analysis and 'mel_spectrogram' in music_analysis['raw_features']:
        mel_spec = music_analysis['raw_features']['mel_spectrogram']
        librosa.display.specshow(
            mel_spec, 
            y_axis='mel', 
            x_axis='time',
            sr=music_analysis.get('sample_rate', 22050),
            hop_length=512,
            ax=ax3
        )
        ax3.set_title('Mel Spectrogram')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_xlabel('Time (s)')
    else:
        ax3.text(0.5, 0.5, "Mel spectrogram not available", 
                 horizontalalignment='center', verticalalignment='center')
        ax3.set_title('Music Analysis')
    
    # Plot beat and tempo
    ax4 = fig.add_subplot(2, 3, 4)
    if 'beats' in music_analysis and 'beat_times' in music_analysis['beats']:
        beat_times = music_analysis['beats']['beat_times']
        tempo = music_analysis['beats']['tempo']
        
        # Plot beat times as vertical lines
        for beat_time in beat_times:
            ax4.axvline(x=beat_time, color='r', alpha=0.5, linestyle='-')
        
        # Add text for tempo
        ax4.text(0.5, 0.9, f"Tempo: {tempo:.1f} BPM", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax4.transAxes)
        
        ax4.set_title('Beat Analysis')
        ax4.set_ylabel('Beat Strength')
        ax4.set_xlabel('Time (s)')
        ax4.set_xlim([0, beat_times[-1] if len(beat_times) > 0 else 10])
        ax4.set_ylim([0, 1])
    else:
        ax4.text(0.5, 0.5, "Beat analysis not available", 
                 horizontalalignment='center', verticalalignment='center')
        ax4.set_title('Beat Analysis')
    
    # Plot dance-music correlation
    ax5 = fig.add_subplot(2, 3, 5)
    if correlation_results is not None and 'fused_features' in correlation_results:
        # Visualize the first component of fused features over time
        fused_features = correlation_results['fused_features']
        times = np.arange(len(fused_features)) / 30  # Assuming 30fps
        
        # PCA to reduce dimensions for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        reduced_features = pca.fit_transform(fused_features.reshape(fused_features.shape[0], -1))
        
        # Plot first 3 components
        ax5.plot(times, reduced_features[:, 0], 'r-', label='Component 1')
        ax5.plot(times, reduced_features[:, 1], 'g-', label='Component 2')
        ax5.plot(times, reduced_features[:, 2], 'b-', label='Component 3')
        
        ax5.set_title('Dance-Music Correlation')
        ax5.set_ylabel('Feature Value')
        ax5.set_xlabel('Time (s)')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "Correlation results not available", 
                 horizontalalignment='center', verticalalignment='center')
        ax5.set_title('Dance-Music Correlation')
    
    # Plot similarity results
    ax6 = fig.add_subplot(2, 3, 6)
    if similarity_results is not None and 'dtw_matrices' in similarity_results:
        # Plot DTW matrix
        dtw_matrix = similarity_results['dtw_matrices'][0][0].cpu().numpy()  # First scale, first batch
        
        # Plot similarity heatmap
        plot_similarity_heatmap(dtw_matrix, ax=ax6)
        
        # Add text for similarity score
        similarity_score = similarity_results['similarity_score'][0]  # First batch
        is_similar = similarity_results['is_similar'][0]  # First batch
        
        similarity_text = f"Similarity: {similarity_score:.2f}"
        if is_similar:
            similarity_text += " (Similar)"
        else:
            similarity_text += " (Different)"
        
        ax6.text(0.5, 0.05, similarity_text, 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax6.transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))
        
        ax6.set_title('Choreography Similarity')
    else:
        ax6.text(0.5, 0.5, "Similarity results not available", 
                 horizontalalignment='center', verticalalignment='center')
        ax6.set_title('Choreography Similarity')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def save_animation(frames, pose_data, music_analysis, output_path):
    """Save 3D pose animation with audio."""
    # Create pose animation
    animation = create_pose_animation(
        frames=frames,
        poses_3d=pose_data['poses_3d'],
        poses_2d=pose_data['poses_2d'],
        left_hand_poses=pose_data['left_hand_poses'],
        right_hand_poses=pose_data['right_hand_poses'],
        audio=music_analysis.get('audio'),
        sample_rate=music_analysis.get('sample_rate', 22050)
    )
    
    # Save animation
    animation.save(output_path, writer='ffmpeg', fps=30, dpi=100, bitrate=5000)
    
    return output_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='DanceMotionAI Integration Demo')
    parser.add_argument('--video1', type=str, required=True, help='Path to first video file')
    parser.add_argument('--video2', type=str, help='Path to second video file for comparison')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--max_frames', type=int, default=300, help='Maximum number of frames to process')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--save_animation', action='store_true', help='Save animation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    models = load_models(config, device)
    
    # Process first video
    print(f"Processing video: {args.video1}")
    frames1, fps1, size1 = load_video(args.video1, args.max_frames)
    processed_frames1 = preprocess_frames(frames1)
    
    print("Extracting 3D poses...")
    pose_data1 = process_video(processed_frames1, models, device)
    
    # Extract audio from first video
    print("Extracting audio...")
    audio_path1 = os.path.join(args.output_dir, 'audio1.wav')
    audio1, sr1 = extract_audio(args.video1, audio_path1)
    
    # Analyze music
    print("Analyzing music...")
    music_analysis1 = analyze_music(audio1, models, sr1)
    
    # Correlate dance and music
    print("Correlating dance and music...")
    correlation_results = correlate_dance_and_music(pose_data1, music_analysis1, fps1, models, device)
    
    # Process second video if provided
    similarity_results = None
    if args.video2:
        print(f"Processing comparison video: {args.video2}")
        frames2, fps2, size2 = load_video(args.video2, args.max_frames)
        processed_frames2 = preprocess_frames(frames2)
        
        print("Extracting 3D poses for comparison...")
        pose_data2 = process_video(processed_frames2, models, device)
        
        # Compare choreographies
        print("Comparing choreographies...")
        similarity_results = compare_choreographies(pose_data1, pose_data2, models, device)
        
        # Print similarity score
        print(f"Similarity score: {similarity_results['similarity_score'][0]:.4f}")
        print(f"Is similar: {bool(similarity_results['is_similar'][0])}")
    
    # Compute dance metrics
    print("Computing dance metrics...")
    metrics = compute_dance_metrics(pose_data1['poses_3d'])
    print("Dance Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, 'dance_metrics.txt')
    with open(metrics_path, 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")
    
    # Visualize results
    if args.visualize:
        print("Visualizing results...")
        fig = visualize_results(frames1, pose_data1, music_analysis1, correlation_results, similarity_results, config)
        
        # Save figure
        fig_path = os.path.join(args.output_dir, 'visualization.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {fig_path}")
        
        # Show figure
        plt.show()
    
    # Save animation
    if args.save_animation:
        print("Saving animation...")
        animation_path = os.path.join(args.output_dir, 'animation.mp4')
        save_animation(frames1, pose_data1, music_analysis1, animation_path)
        print(f"Animation saved to: {animation_path}")
    
    print("Demo completed!")


if __name__ == '__main__':
    main()
