"""
4DHands Module Demo

This script demonstrates the usage of the 4DHands module for precise hand movement
tracking and analysis in dance motions.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hands_4d import Hands4D
from utils.visualization import visualize_hand_keypoints


def load_config(config_path='configs/default_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_synthetic_hand_data(batch_size=1, seq_length=60, num_hands=2, num_keypoints=21, input_dim=3):
    """
    Generate synthetic hand motion data for demonstration.
    
    Args:
        batch_size: Number of sequences in batch
        seq_length: Number of frames in sequence
        num_hands: Number of hands (typically 2 for left and right)
        num_keypoints: Number of keypoints per hand
        input_dim: Dimension of each keypoint (3 for x,y,z)
    
    Returns:
        Tensor of shape [batch_size, seq_length, num_hands, num_keypoints, input_dim]
    """
    # Create base hand pose (realistic hand shape in rest position)
    base_pose = np.zeros((num_keypoints, input_dim))
    
    # Wrist at origin
    base_pose[0] = [0, 0, 0]
    
    # Finger bases (MCP joints)
    finger_bases = [
        [0.0, 0.04, 0.0],    # Thumb
        [0.03, 0.08, 0.0],   # Index
        [0.01, 0.09, 0.0],   # Middle
        [-0.01, 0.085, 0.0], # Ring
        [-0.03, 0.08, 0.0]   # Pinky
    ]
    
    # Set finger bases
    for i, pos in enumerate(finger_bases):
        base_pose[1 + i*4] = pos
        
        # Set finger joints (PIP, DIP, TIP)
        for j in range(1, 4):
            # Each joint is further along the finger direction, with slight curvature
            base_pose[1 + i*4 + j] = [
                pos[0] + j * 0.015 * (0.5 if i == 0 else 0.0),  # Thumb has different direction
                pos[1] + j * 0.02,
                pos[2] - j * 0.005  # Slight curvature inward
            ]
    
    # Create animated sequence by adding motion
    hand_data = np.zeros((batch_size, seq_length, num_hands, num_keypoints, input_dim))
    
    for b in range(batch_size):
        for t in range(seq_length):
            for h in range(num_hands):
                # Base pose with time-varying deformation
                phase = t / seq_length * 2 * np.pi
                scale_factor = 1.0 if h == 0 else -1.0  # Mirror for left/right hand
                
                # Add wave motion to fingers
                for i in range(5):  # For each finger
                    finger_phase = phase + i * np.pi / 5  # Phase shift for each finger
                    finger_amplitude = 0.02 * np.sin(finger_phase)
                    
                    # Apply to finger joints
                    for j in range(4):  # For each joint in the finger
                        idx = 1 + i*4 + j
                        joint_amplitude = finger_amplitude * (j + 1) / 4  # Amplitude increases toward fingertip
                        
                        # Copy base pose and add animation
                        hand_data[b, t, h, idx] = base_pose[idx] + scale_factor * np.array([
                            0.0,
                            joint_amplitude * np.sin(finger_phase),
                            joint_amplitude * np.cos(finger_phase)
                        ])
                
                # Set wrist position (different for each hand)
                hand_data[b, t, h, 0] = [
                    scale_factor * 0.1,  # Hands are horizontally separated
                    0.05 * np.sin(phase + h * np.pi),  # Vertical motion
                    0.05 * np.cos(phase + h * np.pi)   # Depth motion
                ]
    
    return torch.tensor(hand_data, dtype=torch.float32)


def main():
    """Main function to demonstrate 4DHands module."""
    # Load configuration
    config = load_config()
    
    # Create 4DHands model
    hands_4d_config = config.get('hands_4d', {})
    hands_4d_config.setdefault('input_dim', 3)
    hands_4d_config.setdefault('hidden_dim', 128)
    hands_4d_config.setdefault('num_heads', 4)
    hands_4d_config.setdefault('num_classes', 10)  # For gesture classification
    
    model = Hands4D(hands_4d_config)
    
    # Generate synthetic data
    batch_size = 2
    seq_length = 60
    num_hands = 2
    num_keypoints = 21  # 1 wrist + 5 fingers * 4 joints
    input_dim = 3  # x, y, z
    
    hand_data = generate_synthetic_hand_data(
        batch_size=batch_size,
        seq_length=seq_length,
        num_hands=num_hands,
        num_keypoints=num_keypoints,
        input_dim=input_dim
    )
    
    print(f"Generated hand data shape: {hand_data.shape}")
    
    # Forward pass through model
    with torch.no_grad():
        hand_features, logits = model(hand_data)
    
    print(f"Hand features shape: {hand_features.shape}")
    if logits is not None:
        print(f"Logits shape: {logits.shape}")
    
    # Visualize hand keypoints
    print("Visualizing hand keypoints...")
    
    # Define hand connections for visualization
    hand_connections = [
        # Thumb connections
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger connections
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger connections
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger connections
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky connections
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (1, 5), (5, 9), (9, 13), (13, 17)
    ]
    
    # Convert single sample to numpy for visualization
    sample_hand_data = hand_data[0, 0].numpy()  # First batch, first frame
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 6))
    
    # Plot left hand
    ax1 = fig.add_subplot(121, projection='3d')
    left_hand = sample_hand_data[0]  # Left hand
    visualize_hand_keypoints(left_hand, hand_connections, ax1, title="Left Hand")
    
    # Plot right hand
    ax2 = fig.add_subplot(122, projection='3d')
    right_hand = sample_hand_data[1]  # Right hand
    visualize_hand_keypoints(right_hand, hand_connections, ax2, title="Right Hand")
    
    plt.tight_layout()
    plt.savefig("examples/hand_visualization.png")
    print(f"Visualization saved to examples/hand_visualization.png")
    
    # Animate hand motion (optional, requires ffmpeg)
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation
        
        print("Creating animation...")
        
        # Create new figure for animation
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Function to update the plot in each animation frame
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            # Get hand data for current frame
            frame_data = hand_data[0, frame].numpy()
            
            # Plot hands
            visualize_hand_keypoints(frame_data[0], hand_connections, ax1, title=f"Left Hand (Frame {frame})")
            visualize_hand_keypoints(frame_data[1], hand_connections, ax2, title=f"Right Hand (Frame {frame})")
            
            # Set consistent axes limits
            ax1.set_xlim([-0.2, 0.2])
            ax1.set_ylim([-0.2, 0.2])
            ax1.set_zlim([-0.2, 0.2])
            ax2.set_xlim([-0.2, 0.2])
            ax2.set_ylim([-0.2, 0.2])
            ax2.set_zlim([-0.2, 0.2])
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=min(30, seq_length), interval=100)
        
        # Save animation
        ani.save("examples/hand_animation.mp4", writer="ffmpeg", fps=10)
        print(f"Animation saved to examples/hand_animation.mp4")
    except (ImportError, ValueError) as e:
        print(f"Animation could not be created: {e}")
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
