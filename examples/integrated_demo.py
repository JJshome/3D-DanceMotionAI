"""
Integrated demo script for DanceMotionAI system.

This script demonstrates the complete pipeline of the DanceMotionAI system,
including video processing, 3D pose estimation, hand tracking, music analysis,
and choreography similarity detection.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import cv2
import librosa
from pathlib import Path

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import DanceMotionAI modules
from models.dancehrnet import create_model as create_dancehrnet
from models.hands_4d import Hands4D
from models.danceformer import create_model as create_danceformer
from models.dance_dtw import create_model as create_dance_dtw
from models.music_analysis import create_music_analyzer
from utils.visualization import create_pose_animation, create_comparison_visualization

def load_config(config_path):
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_video(video_path):
    """Load video file and return frames and FPS."""
    print(f"Loading video: {video_path}")
    
    # Check if the file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    # Release the video capture object
    cap.release()
    
    print(f"Loaded {len(frames)} frames")
    return frames, fps

def extract_audio(video_path, output_path):
    """Extract audio from video file and save it as a WAV file."""
    print(f"Extracting audio from: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use ffmpeg to extract audio
    import subprocess
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        '-y',  # Overwrite output file if it exists
        output_path
    ]
    
    # Execute the command
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extracted and saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"Command output: {e.stdout.decode()}")
        print(f"Command error: {e.stderr.decode()}")
        raise

def preprocess_frames(frames):
    """Preprocess frames for pose estimation."""
    print("Preprocessing frames...")
    
    # Initialize list for processed frames
    processed_frames = []
    
    # Process each frame
    for frame in frames:
        # Resize frame to a consistent size
        resized_frame = cv2.resize(frame, (512, 512))
        
        # Normalize pixel values to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Standardize using mean and std (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        standardized_frame = (normalized_frame - mean) / std
        
        # Convert to torch tensor and add batch dimension
        tensor_frame = torch.from_numpy(standardized_frame).permute(2, 0, 1).unsqueeze(0)
        
        # Add to processed frames list
        processed_frames.append(tensor_frame)
    
    print(f"Preprocessed {len(processed_frames)} frames")
    return processed_frames

def process_video(frames, dancehrnet, hands_4d, device):
    """Process video frames to extract 3D poses and hand movements."""
    print("Processing video for 3D pose estimation...")
    
    # Initialize lists for poses and hand positions
    poses_3d = []
    hand_poses = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    num_frames = len(frames)
    
    for i in range(0, num_frames, batch_size):
        print(f"Processing frames {i} to {min(i + batch_size, num_frames)}...")
        
        # Get batch of frames
        batch_frames = frames[i:min(i + batch_size, num_frames)]
        
        # Concatenate frames into a batch
        batch_tensor = torch.cat(batch_frames, dim=0).to(device)
        
        # Process with dancehrnet
        with torch.no_grad():
            batch_poses = dancehrnet(batch_tensor)
        
        # Process with hands_4d
        with torch.no_grad():
            batch_hands = hands_4d(batch_tensor, batch_poses)
        
        # Convert to numpy and append to lists
        batch_poses_np = batch_poses.cpu().numpy()
        batch_hands_np = batch_hands.cpu().numpy()
        
        poses_3d.extend([pose for pose in batch_poses_np])
        hand_poses.extend([hand for hand in batch_hands_np])
    
    print(f"Processed {len(poses_3d)} frames for pose estimation")
    return np.array(poses_3d), np.array(hand_poses)

def analyze_music(audio_path, music_analyzer):
    """Analyze music from audio file and extract features."""
    print(f"Analyzing music from: {audio_path}")
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract features
    features = music_analyzer.extract_features(y, sr)
    
    print("Music analysis complete")
    return features

def correlate_dance_and_music(poses_3d, music_features, danceformer, device):
    """Correlate dance movements with music features."""
    print("Correlating dance movements with music...")
    
    # Convert poses to tensor
    poses_tensor = torch.from_numpy(poses_3d).float().to(device)
    
    # Convert music features to tensor
    music_tensor = torch.from_numpy(music_features).float().to(device)
    
    # Process with danceformer
    with torch.no_grad():
        multimodal_features = danceformer(poses_tensor, music_tensor)
    
    # Convert to numpy
    multimodal_features_np = multimodal_features.cpu().numpy()
    
    print("Dance-music correlation complete")
    return multimodal_features_np

def compare_choreographies(features1, features2, dance_dtw):
    """Compare two choreographies and compute similarity."""
    print("Comparing choreographies...")
    
    # Convert features to tensor
    features1_tensor = torch.from_numpy(features1).float()
    features2_tensor = torch.from_numpy(features2).float()
    
    # Compute similarity using dance_dtw
    with torch.no_grad():
        similarity_scores, frame_similarities, key_differences = dance_dtw(features1_tensor, features2_tensor)
    
    # Convert to numpy
    similarity_scores_np = similarity_scores.cpu().numpy()
    frame_similarities_np = frame_similarities.cpu().numpy()
    
    # Process key differences
    key_diffs = []
    for diff in key_differences:
        key_diffs.append({
            'timestamp': diff['timestamp'].item(),
            'description': diff['description'],
            'suggestion': diff['suggestion']
        })
    
    print(f"Choreography comparison complete. Overall similarity: {similarity_scores_np[0]:.4f}")
    
    # Create similarity results dictionary
    similarity_results = {
        'overall_similarity': float(similarity_scores_np[0]),
        'frame_similarities': frame_similarities_np.tolist(),
        'key_differences': key_diffs
    }
    
    return similarity_results

def compute_dance_metrics(poses_3d, hand_poses):
    """Compute various dance performance metrics based on poses."""
    print("Computing dance performance metrics...")
    
    # Define metrics to compute
    metrics = {
        'movement_precision': 0.0,
        'timing_accuracy': 0.0,
        'hand_movement_quality': 0.0,
        'energy_level': 0.0,
        'movement_fluidity': 0.0,
        'balance_control': 0.0,
        'music_correlation_data': []
    }
    
    # Calculate movement precision
    # (In a real implementation, this would be based on comparison with reference movements
    # or analysis of joint angles consistency)
    # Here we use placeholder calculation
    joint_stability = compute_joint_stability(poses_3d)
    metrics['movement_precision'] = 60 + (joint_stability * 30)  # Scale to 0-100
    
    # Calculate timing accuracy (placeholder)
    metrics['timing_accuracy'] = 70 + np.random.normal(0, 5)
    
    # Calculate hand movement quality
    hand_quality = compute_hand_movement_quality(hand_poses)
    metrics['hand_movement_quality'] = 50 + (hand_quality * 40)  # Scale to 0-100
    
    # Calculate energy level
    energy = compute_energy_level(poses_3d)
    metrics['energy_level'] = 40 + (energy * 50)  # Scale to 0-100
    
    # Calculate movement fluidity
    fluidity = compute_movement_fluidity(poses_3d)
    metrics['movement_fluidity'] = 55 + (fluidity * 35)  # Scale to 0-100
    
    # Calculate balance control
    balance = compute_balance_control(poses_3d)
    metrics['balance_control'] = 65 + (balance * 25)  # Scale to 0-100
    
    # Generate mock music correlation data
    # (In a real implementation, this would come from actual music-movement correlation analysis)
    metrics['music_correlation_data'] = generate_mock_correlation_data(len(poses_3d))
    
    # Ensure all metrics are within 0-100 range
    for key in metrics:
        if key != 'music_correlation_data':
            metrics[key] = max(0, min(100, metrics[key]))
    
    print("Dance metrics computation complete")
    return metrics

def compute_joint_stability(poses_3d):
    """Compute stability of joints over time."""
    # Calculate joint velocity
    joint_velocities = np.diff(poses_3d, axis=0)
    
    # Calculate average joint velocity magnitude
    velocity_magnitudes = np.linalg.norm(joint_velocities, axis=2)
    avg_velocity = np.mean(velocity_magnitudes)
    
    # Map to 0-1 range (higher is more stable)
    # We use exponential decay to map high velocities to low stability
    stability = np.exp(-avg_velocity / 10)
    
    return float(stability)

def compute_hand_movement_quality(hand_poses):
    """Compute quality of hand movements."""
    # Calculate smoothness of hand trajectories
    hand_velocities = np.diff(hand_poses, axis=0)
    
    # Calculate jerk (derivative of acceleration)
    hand_acceleration = np.diff(hand_velocities, axis=0)
    hand_jerk = np.diff(hand_acceleration, axis=0)
    
    # Average magnitude of jerk (lower is smoother)
    jerk_magnitude = np.linalg.norm(hand_jerk, axis=2)
    avg_jerk = np.mean(jerk_magnitude)
    
    # Map to 0-1 range (higher is better quality)
    # We use exponential decay to map high jerk to low quality
    quality = np.exp(-avg_jerk / 5)
    
    return float(quality)

def compute_energy_level(poses_3d):
    """Compute energy level based on movement intensity."""
    # Calculate joint velocities
    joint_velocities = np.diff(poses_3d, axis=0)
    
    # Calculate average velocity magnitude
    velocity_magnitudes = np.linalg.norm(joint_velocities, axis=2)
    avg_velocity = np.mean(velocity_magnitudes)
    
    # Map to 0-1 range (higher velocity = higher energy)
    # Use sigmoid to map velocity to energy level
    energy = 2 / (1 + np.exp(-avg_velocity / 5)) - 1
    
    return float(energy)

def compute_movement_fluidity(poses_3d):
    """Compute fluidity of movements."""
    # Calculate joint velocities and accelerations
    joint_velocities = np.diff(poses_3d, axis=0)
    joint_accelerations = np.diff(joint_velocities, axis=0)
    
    # Calculate correlation between consecutive accelerations
    # High correlation means consistent motion (more fluid)
    correlations = []
    for i in range(len(joint_accelerations) - 1):
        corr = np.corrcoef(
            joint_accelerations[i].flatten(), 
            joint_accelerations[i+1].flatten()
        )[0, 1]
        
        if not np.isnan(corr):
            correlations.append(corr)
    
    # Average correlation (higher is more fluid)
    avg_correlation = np.mean(correlations) if correlations else 0.5
    
    # Map to 0-1 range
    fluidity = (avg_correlation + 1) / 2  # Transform from [-1, 1] to [0, 1]
    
    return float(fluidity)

def compute_balance_control(poses_3d):
    """Compute balance control based on stability of center of mass."""
    # Assuming poses_3d has shape [frames, joints, 3]
    
    # Define torso joints (simplified)
    torso_joints = [1, 2, 3, 4]  # Example indices for spine and hip joints
    
    # Extract torso positions
    torso_positions = poses_3d[:, torso_joints, :]
    
    # Calculate approximate center of mass (COM) as average of torso joints
    com = np.mean(torso_positions, axis=1)
    
    # Calculate COM movement in horizontal plane (x, z)
    com_movement = np.diff(com[:, [0, 2]], axis=0)
    com_velocity = np.linalg.norm(com_movement, axis=1)
    
    # Lower horizontal velocity indicates better balance
    avg_com_velocity = np.mean(com_velocity)
    
    # Map to 0-1 range (higher is better balance)
    # We use exponential decay to map high velocity to low balance
    balance = np.exp(-avg_com_velocity / 3)
    
    return float(balance)

def generate_mock_correlation_data(num_frames):
    """Generate mock music-movement correlation data."""
    # Generate smooth random data using sine waves with noise
    x = np.linspace(0, 10, num_frames)
    # Base sine wave
    base = 0.6 + 0.2 * np.sin(x) + 0.1 * np.sin(2.5 * x)
    # Add noise
    noise = np.random.normal(0, 0.05, num_frames)
    # Combine and clip to 0-1 range
    correlation = np.clip(base + noise, 0, 1)
    
    return correlation.tolist()

def main():
    parser = argparse.ArgumentParser(description='DanceMotionAI Integrated Demo')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--reference', type=str, default=None, help='Path to reference video file for comparison')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create model instances
    print("Loading models...")
    dancehrnet = create_dancehrnet(config['pose_estimation']).to(device)
    hands_4d = Hands4D(config['hands_4d']).to(device)
    danceformer = create_danceformer(config['multimodal_analysis']).to(device)
    dance_dtw = create_dance_dtw(config['similarity_analysis']).to(device)
    music_analyzer = create_music_analyzer(config['music_analysis'])
    
    # Set models to evaluation mode
    dancehrnet.eval()
    hands_4d.eval()
    danceformer.eval()
    dance_dtw.eval()
    
    print("Models loaded successfully")
    
    # Process input video
    print("\n--- Processing input video ---")
    frames, fps = load_video(args.video)
    
    # Extract audio from input video
    audio_path = os.path.join(args.output_dir, 'audio.wav')
    extract_audio(args.video, audio_path)
    
    # Preprocess frames
    processed_frames = preprocess_frames(frames)
    
    # Extract 3D poses and hand movements
    poses_3d, hand_poses = process_video(processed_frames, dancehrnet, hands_4d, device)
    
    # Analyze music
    music_features = analyze_music(audio_path, music_analyzer)
    
    # Correlate dance and music
    multimodal_features = correlate_dance_and_music(poses_3d, music_features, danceformer, device)
    
    # Process reference video if provided
    reference_frames = None
    reference_poses_3d = None
    reference_hand_poses = None
    reference_multimodal_features = None
    similarity_results = None
    
    if args.reference:
        print("\n--- Processing reference video ---")
        reference_frames, reference_fps = load_video(args.reference)
        
        # Extract audio from reference video
        reference_audio_path = os.path.join(args.output_dir, 'reference_audio.wav')
        extract_audio(args.reference, reference_audio_path)
        
        # Preprocess frames
        processed_reference_frames = preprocess_frames(reference_frames)
        
        # Extract 3D poses and hand movements
        reference_poses_3d, reference_hand_poses = process_video(
            processed_reference_frames, dancehrnet, hands_4d, device
        )
        
        # Analyze music
        reference_music_features = analyze_music(reference_audio_path, music_analyzer)
        
        # Correlate dance and music
        reference_multimodal_features = correlate_dance_and_music(
            reference_poses_3d, reference_music_features, danceformer, device
        )
        
        # Compare choreographies
        similarity_results = compare_choreographies(
            multimodal_features, reference_multimodal_features, dance_dtw
        )
    
    # Compute dance metrics
    metrics = compute_dance_metrics(poses_3d, hand_poses)
    
    # Create visualizations
    print("\n--- Creating visualizations ---")
    visualization_path = os.path.join(args.output_dir, 'visualization.mp4')
    create_pose_animation(frames, poses_3d, visualization_path, fps=fps)
    
    if reference_frames is not None and reference_poses_3d is not None:
        comparison_path = os.path.join(args.output_dir, 'comparison.mp4')
        create_comparison_visualization(
            frames, poses_3d, 
            reference_frames, reference_poses_3d,
            comparison_path, fps=fps
        )
    
    # Save results
    import json
    results = {
        'metrics': metrics,
    }
    
    if similarity_results:
        results['similarity'] = similarity_results
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Visualization saved to: {visualization_path}")
    
    if args.reference:
        print(f"Comparison visualization saved to: {comparison_path}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
