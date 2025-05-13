"""
Utility functions for the DanceMotionAI system.

This module provides helper functions for various tasks including:
- Data preprocessing and normalization
- 3D visualization
- Report generation
- File handling
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import datetime


def normalize_pose(pose: np.ndarray, center_joint: int = 0) -> np.ndarray:
    """
    Normalize pose by centering and scaling.
    
    Args:
        pose: Pose data [num_joints, 3] or [seq_len, num_joints, 3]
        center_joint: Index of joint to use as center (default: 0 - hip)
        
    Returns:
        Normalized pose with same shape as input
    """
    if pose.ndim == 2:
        # Single pose [num_joints, 3]
        centered_pose = pose - pose[center_joint:center_joint+1]
        
        # Scale by average distance from center
        scale = np.mean(np.sqrt(np.sum(centered_pose**2, axis=1)))
        if scale > 0:
            normalized_pose = centered_pose / scale
        else:
            normalized_pose = centered_pose
            
        return normalized_pose
    
    elif pose.ndim == 3:
        # Sequence of poses [seq_len, num_joints, 3]
        normalized_seq = np.zeros_like(pose)
        for i in range(pose.shape[0]):
            normalized_seq[i] = normalize_pose(pose[i], center_joint)
            
        return normalized_seq
    
    else:
        raise ValueError(f"Invalid pose shape: {pose.shape}")


def smooth_pose_sequence(sequence: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply temporal smoothing to pose sequence using moving average.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        window_size: Size of smoothing window
        
    Returns:
        Smoothed pose sequence with same shape as input
    """
    if sequence.ndim != 3:
        raise ValueError(f"Expected 3D sequence, got shape {sequence.shape}")
    
    seq_len, num_joints, dims = sequence.shape
    
    if seq_len <= window_size:
        return sequence
    
    # Reshape for easier processing
    flat_seq = sequence.reshape(seq_len, -1)
    
    # Apply moving average
    smoothed = np.zeros_like(flat_seq)
    half_window = window_size // 2
    
    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        smoothed[i] = np.mean(flat_seq[start:end], axis=0)
    
    # Reshape back
    return smoothed.reshape(seq_len, num_joints, dims)


def compute_joint_angles(pose: np.ndarray) -> np.ndarray:
    """
    Compute joint angles for a pose.
    
    Args:
        pose: 3D pose data [num_joints, 3]
        
    Returns:
        Joint angles in radians [num_angles]
    """
    # Define pairs of joints to compute angles for
    # This is a simplified example - in practice, would define specific triplets
    # of joints for anatomically meaningful angles
    joint_triplets = [
        # Right arm
        (2, 3, 4),    # Shoulder, elbow, wrist
        # Left arm
        (5, 6, 7),    # Shoulder, elbow, wrist
        # Right leg
        (9, 10, 11),  # Hip, knee, ankle
        # Left leg
        (12, 13, 14), # Hip, knee, ankle
        # Spine
        (0, 1, 8),    # Pelvis, spine, neck
    ]
    
    angles = []
    
    for a, b, c in joint_triplets:
        # Get vectors
        v1 = pose[a] - pose[b]
        v2 = pose[c] - pose[b]
        
        # Normalize
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Compute angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    return np.array(angles)


def compute_velocities(sequence: np.ndarray) -> np.ndarray:
    """
    Compute velocities for a pose sequence using finite differences.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Velocities [seq_len-1, num_joints, 3]
    """
    return np.diff(sequence, axis=0)


def compute_accelerations(sequence: np.ndarray) -> np.ndarray:
    """
    Compute accelerations for a pose sequence using finite differences.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Accelerations [seq_len-2, num_joints, 3]
    """
    velocities = compute_velocities(sequence)
    return np.diff(velocities, axis=0)


def load_pose_sequence(file_path: str) -> np.ndarray:
    """
    Load pose sequence from file.
    
    Args:
        file_path: Path to pose sequence file
        
    Returns:
        Pose sequence [seq_len, num_joints, 3]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file format from extension
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        # NumPy binary file
        return np.load(file_path)
    
    elif ext == '.json':
        # JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if data is in expected format
        if isinstance(data, list) and all(isinstance(frame, list) for frame in data):
            return np.array(data)
        else:
            raise ValueError(f"Invalid JSON format in {file_path}")
    
    elif ext in ['.txt', '.csv']:
        # Text file - assume each line is a flattened pose
        poses = []
        with open(file_path, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split(',')]
                poses.append(values)
        
        # Try to infer dimensions
        poses = np.array(poses)
        if poses.shape[1] % 3 == 0:
            num_joints = poses.shape[1] // 3
            return poses.reshape(-1, num_joints, 3)
        else:
            raise ValueError(f"Cannot reshape pose data with shape {poses.shape}")
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_pose_sequence(sequence: np.ndarray, file_path: str) -> None:
    """
    Save pose sequence to file.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        file_path: Path to output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Determine file format from extension
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        # NumPy binary file
        np.save(file_path, sequence)
    
    elif ext == '.json':
        # JSON file
        with open(file_path, 'w') as f:
            json.dump(sequence.tolist(), f)
    
    elif ext in ['.txt', '.csv']:
        # Text file - flatten poses
        with open(file_path, 'w') as f:
            for pose in sequence:
                flat_pose = pose.flatten()
                line = ','.join(str(x) for x in flat_pose)
                f.write(line + '\n')
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def visualize_pose_3d(
    pose: np.ndarray,
    connections: Optional[List[Tuple[int, int]]] = None,
    ax: Optional[plt.Axes] = None,
    color: str = 'b',
    marker: str = 'o',
    markersize: int = 5,
    linewidth: int = 2,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Visualize a 3D pose.
    
    Args:
        pose: 3D pose data [num_joints, 3]
        connections: List of joint index pairs to connect
        ax: Matplotlib 3D axes (created if None)
        color: Color for pose
        marker: Marker style for joints
        markersize: Size of markers
        linewidth: Width of connection lines
        title: Plot title
        
    Returns:
        Matplotlib 3D axes
    """
    if pose.ndim != 2 or pose.shape[1] != 3:
        raise ValueError(f"Expected pose with shape [num_joints, 3], got {pose.shape}")
    
    # Default COCO connections if none provided
    if connections is None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # Right arm
            (0, 5), (5, 6), (6, 7),             # Left arm
            (0, 8), (8, 9), (9, 10),            # Right leg
            (0, 11), (11, 12), (12, 13),        # Left leg
            (0, 14), (14, 15), (15, 16),        # Spine and head
        ]
    
    # Create 3D axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color, marker=marker, s=markersize)
    
    # Plot connections
    for i, j in connections:
        if i < pose.shape[0] and j < pose.shape[0]:
            ax.plot(
                [pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]],
                [pose[i, 2], pose[j, 2]],
                c=color, linewidth=linewidth
            )
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set title if provided
    if title:
        ax.set_title(title)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax


def create_pose_animation(
    sequence: np.ndarray,
    connections: Optional[List[Tuple[int, int]]] = None,
    output_path: Optional[str] = None,
    fps: int = 30,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None
) -> Union[animation.FuncAnimation, None]:
    """
    Create animation of a pose sequence.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        connections: List of joint index pairs to connect
        output_path: Path to save animation (if None, returns animation object)
        fps: Frames per second
        figsize: Figure size
        title: Animation title
        
    Returns:
        Animation object if output_path is None, otherwise None
    """
    if sequence.ndim != 3 or sequence.shape[2] != 3:
        raise ValueError(f"Expected sequence with shape [seq_len, num_joints, 3], got {sequence.shape}")
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine axis limits for consistent view
    x_min, y_min, z_min = np.min(sequence, axis=(0, 1))
    x_max, y_max, z_max = np.max(sequence, axis=(0, 1))
    
    # Add some padding
    padding = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_min -= padding
    y_min -= padding
    z_min -= padding
    x_max += padding
    y_max += padding
    z_max += padding
    
    # Initialize with first frame
    scatter = ax.scatter([], [], [], c='b', marker='o', s=10)
    lines = []
    
    # Default COCO connections if none provided
    if connections is None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # Right arm
            (0, 5), (5, 6), (6, 7),             # Left arm
            (0, 8), (8, 9), (9, 10),            # Right leg
            (0, 11), (11, 12), (12, 13),        # Left leg
            (0, 14), (14, 15), (15, 16),        # Spine and head
        ]
    
    for _ in connections:
        line, = ax.plot([], [], [], 'b-', linewidth=2)
        lines.append(line)
    
    # Set consistent view
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    
    # Animation update function
    def update(frame):
        pose = sequence[frame]
        
        # Update joints
        scatter._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        
        # Update connections
        for i, (joint1, joint2) in enumerate(connections):
            if joint1 < pose.shape[0] and joint2 < pose.shape[0]:
                lines[i].set_data([pose[joint1, 0], pose[joint2, 0]], [pose[joint1, 1], pose[joint2, 1]])
                lines[i].set_3d_properties([pose[joint1, 2], pose[joint2, 2]])
        
        return [scatter] + lines
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(sequence), interval=1000/fps, blit=True
    )
    
    # Save if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine format from extension
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.gif':
            anim.save(output_path, writer='pillow', fps=fps)
        elif ext == '.mp4':
            anim.save(output_path, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'])
        else:
            raise ValueError(f"Unsupported output format: {ext}")
        
        plt.close(fig)
        return None
    else:
        return anim


def create_similarity_heatmap(
    distance_matrix: np.ndarray,
    optimal_path: Optional[List[Tuple[int, int]]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "DTW Similarity Heatmap",
    cmap: str = 'viridis_r'
) -> Optional[plt.Figure]:
    """
    Create similarity heatmap visualization.
    
    Args:
        distance_matrix: Distance matrix [seq_len1, seq_len2]
        optimal_path: Optional optimal alignment path to overlay
        output_path: Path to save visualization (if None, returns figure)
        figsize: Figure size
        title: Plot title
        cmap: Colormap name
        
    Returns:
        Matplotlib figure if output_path is None, otherwise None
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distance matrix as heatmap
    im = ax.imshow(distance_matrix, cmap=cmap, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance')
    
    # Plot optimal path if provided
    if optimal_path:
        path_x = [p[0] for p in optimal_path]
        path_y = [p[1] for p in optimal_path]
        ax.plot(path_y, path_x, 'r-', linewidth=2)
    
    # Add labels and title
    ax.set_xlabel('Sequence 2 Frame')
    ax.set_ylabel('Sequence 1 Frame')
    ax.set_title(title)
    
    # Save if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig


def plot_alignment_path(
    path: List[Tuple[int, int]],
    seq1_len: int,
    seq2_len: int,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "DTW Alignment Path"
) -> Optional[plt.Figure]:
    """
    Plot the optimal alignment path from DTW.
    
    Args:
        path: Optimal alignment path as list of (idx1, idx2) tuples
        seq1_len: Length of first sequence
        seq2_len: Length of second sequence
        output_path: Path to save visualization (if None, returns figure)
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure if output_path is None, otherwise None
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot diagonal line for reference
    ax.plot([0, min(seq1_len, seq2_len)], [0, min(seq1_len, seq2_len)], 'k--', alpha=0.5)
    
    # Plot alignment path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    
    # Color path by distance from diagonal
    diagonal_dist = [abs(x - y) for x, y in zip(path_x, path_y)]
    max_dist = max(diagonal_dist) if diagonal_dist else 1
    norm_dist = [d / max_dist for d in diagonal_dist]
    
    # Plot path as connected colored segments
    points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create line collection
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='coolwarm', norm=plt.Normalize(0, 1))
    lc.set_array(np.array(norm_dist))
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    
    # Add colorbar
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('Distance from Diagonal')
    
    # Set limits
    ax.set_xlim(0, seq1_len)
    ax.set_ylim(0, seq2_len)
    
    # Add labels and title
    ax.set_xlabel('Sequence 1 Frame')
    ax.set_ylabel('Sequence 2 Frame')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig


def plot_joint_trajectories(
    sequence: np.ndarray,
    joint_indices: List[int],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Joint Trajectories"
) -> Optional[plt.Figure]:
    """
    Plot trajectories of selected joints.
    
    Args:
        sequence: Pose sequence [seq_len, num_joints, 3]
        joint_indices: Indices of joints to plot
        output_path: Path to save visualization (if None, returns figure)
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure if output_path is None, otherwise None
    """
    if sequence.ndim != 3 or sequence.shape[2] != 3:
        raise ValueError(f"Expected sequence with shape [seq_len, num_joints, 3], got {sequence.shape}")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create 3D axis for 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Create 2D axes for separate x, y, z plots
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # Colors for different joints
    colors = plt.cm.tab10(np.linspace(0, 1, len(joint_indices)))
    
    # Plot 3D trajectories
    for i, joint_idx in enumerate(joint_indices):
        if joint_idx < sequence.shape[1]:
            joint_sequence = sequence[:, joint_idx]
            ax1.plot(joint_sequence[:, 0], joint_sequence[:, 1], joint_sequence[:, 2], 
                     c=colors[i], label=f"Joint {joint_idx}")
            
            # Plot trajectories for each dimension
            frames = np.arange(len(sequence))
            ax2.plot(frames, joint_sequence[:, 0], c=colors[i], label=f"Joint {joint_idx}")
            ax3.plot(frames, joint_sequence[:, 1], c=colors[i], label=f"Joint {joint_idx}")
            ax4.plot(frames, joint_sequence[:, 2], c=colors[i], label=f"Joint {joint_idx}")
    
    # Set labels and titles
    ax1.set_title("3D Trajectories")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    ax2.set_title("X Coordinate")
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('X')
    ax2.legend()
    
    ax3.set_title("Y Coordinate")
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Y')
    ax3.legend()
    
    ax4.set_title("Z Coordinate")
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Z')
    ax4.legend()
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig


def generate_report_html(
    results: Dict[str, Any],
    visualization_paths: Dict[str, str],
    output_path: str
) -> None:
    """
    Generate HTML report from similarity analysis results.
    
    Args:
        results: Similarity analysis results
        visualization_paths: Dictionary of visualization file paths
        output_path: Path to save HTML report
    """
    # Create HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dance Similarity Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .summary {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .score {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }}
            .category {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }}
            .very-similar {{
                background-color: #2ecc71;
                color: white;
            }}
            .similar {{
                background-color: #3498db;
                color: white;
            }}
            .partially-similar {{
                background-color: #f39c12;
                color: white;
            }}
            .not-similar {{
                background-color: #e74c3c;
                color: white;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
            }}
            .visualization {{
                margin-bottom: 30px;
            }}
            img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dance Similarity Analysis Report</h1>
            <p><strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Overall Similarity Score:</strong> <span class="score">{results['similarity_score']:.2f}</span></p>
                <p><strong>Category:</strong> <span class="category {results['similarity_category'].lower().replace(' ', '-')}">{results['similarity_category']}</span></p>
                <p><strong>Method Used:</strong> {results['method']}</p>
            </div>
            
            <h2>Similar Segments</h2>
    """
    
    # Add similar segments table
    if 'similar_segments' in results and results['similar_segments']:
        html += """
            <table>
                <thead>
                    <tr>
                        <th>Sequence 1</th>
                        <th>Sequence 2</th>
                        <th>Segment Length</th>
                        <th>Similarity</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for segment in results['similar_segments']:
            html += f"""
                    <tr>
                        <td>Frames {segment['sequence1_start']} - {segment['sequence1_end']}</td>
                        <td>Frames {segment['sequence2_start']} - {segment['sequence2_end']}</td>
                        <td>{segment['segment_length1']} frames</td>
                        <td>{segment['segment_similarity']:.2f}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        """
    else:
        html += "<p>No significant similar segments found.</p>"
    
    # Add weights table
    html += """
            <h2>Distance Weights</h2>
            <table>
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for component, weight in results['weights_used'].items():
        html += f"""
                <tr>
                    <td>{component.replace('_', ' ').title()}</td>
                    <td>{weight:.2f}</td>
                </tr>
        """
    
    html += """
                </tbody>
            </table>
    """
    
    # Add visualizations
    html += "<h2>Visualizations</h2>"
    
    for name, path in visualization_paths.items():
        if os.path.exists(path):
            # Get relative path from output directory
            rel_path = os.path.relpath(path, os.path.dirname(output_path))
            
            html += f"""
            <div class="visualization">
                <h3>{name.replace('_', ' ').title()}</h3>
                <img src="{rel_path}" alt="{name}">
            </div>
            """
    
    # Close HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html)


def compare_choreographies(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare the results of multiple choreography similarity analyses.
    
    Args:
        results1: First similarity analysis results
        results2: Second similarity analysis results
        output_path: Path to save comparison plot
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "similarity_diff": abs(results1["similarity_score"] - results2["similarity_score"]),
        "category_match": results1["similarity_category"] == results2["similarity_category"],
        "num_similar_segments": (
            len(results1.get("similar_segments", [])), 
            len(results2.get("similar_segments", []))
        ),
    }
    
    # Create comparison visualization if output path provided
    if output_path:
        # Create bar chart comparing scores
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = [results1["similarity_score"], results2["similarity_score"]]
        labels = ["Analysis 1", "Analysis 2"]
        
        bars = ax.bar(labels, scores, color=['#3498db', '#2ecc71'])
        
        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom'
            )
        
        # Add threshold lines
        ax.axhline(y=0.85, color='#2ecc71', linestyle='--', alpha=0.7, label="Very Similar")
        ax.axhline(y=0.7, color='#3498db', linestyle='--', alpha=0.7, label="Similar")
        ax.axhline(y=0.55, color='#f39c12', linestyle='--', alpha=0.7, label="Partially Similar")
        
        # Set labels and title
        ax.set_xlabel('Analysis')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Comparison of Similarity Analyses')
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return comparison


def extract_features(pose_sequence: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract various features from pose sequence for analysis.
    
    Args:
        pose_sequence: Pose sequence [seq_len, num_joints, 3]
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # Basic sequence info
    features['sequence_length'] = pose_sequence.shape[0]
    features['num_joints'] = pose_sequence.shape[1]
    
    # Position features
    features['mean_position'] = np.mean(pose_sequence, axis=0)  # [num_joints, 3]
    features['std_position'] = np.std(pose_sequence, axis=0)    # [num_joints, 3]
    
    # Movement features
    if pose_sequence.shape[0] > 1:
        velocities = compute_velocities(pose_sequence)
        features['mean_velocity'] = np.mean(velocities, axis=0)        # [num_joints, 3]
        features['max_velocity'] = np.max(np.abs(velocities), axis=0)  # [num_joints, 3]
    
    # Joint angle features
    angles = np.array([compute_joint_angles(pose) for pose in pose_sequence])  # [seq_len, num_angles]
    features['mean_angles'] = np.mean(angles, axis=0)  # [num_angles]
    features['std_angles'] = np.std(angles, axis=0)    # [num_angles]
    
    # Movement complexity (approximated by frequency analysis)
    if pose_sequence.shape[0] > 10:
        # Compute FFT for each joint coordinate
        flattened = pose_sequence.reshape(pose_sequence.shape[0], -1)  # [seq_len, num_joints*3]
        fft_result = np.abs(np.fft.rfft(flattened, axis=0))  # [seq_len//2+1, num_joints*3]
        
        # Measure complexity by energy in higher frequencies
        total_energy = np.sum(fft_result**2, axis=0)
        high_freq_energy = np.sum(fft_result[fft_result.shape[0]//4:, :]**2, axis=0)
        features['complexity'] = high_freq_energy / (total_energy + 1e-10)  # [num_joints*3]
    
    return features


def analyze_choreography_style(
    pose_sequence: np.ndarray,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the style characteristics of a choreography sequence.
    
    Args:
        pose_sequence: Pose sequence [seq_len, num_joints, 3]
        output_path: Path to save style analysis visualization
        
    Returns:
        Dictionary with style analysis metrics
    """
    features = extract_features(pose_sequence)
    
    # Compute basic statistics
    stats = {
        'duration_frames': features['sequence_length'],
        'duration_seconds': features['sequence_length'] / 30.0,  # Assuming 30fps
    }
    
    # Compute motion intensity
    if 'max_velocity' in features:
        motion_intensity = np.mean(features['max_velocity'])
        stats['motion_intensity'] = float(motion_intensity)
    
    # Compute spatial extent
    spatial_extent = np.ptp(pose_sequence, axis=0).mean()  # Peak-to-peak range
    stats['spatial_extent'] = float(spatial_extent)
    
    # Compute rhythmic regularity (if sequence is long enough)
    if pose_sequence.shape[0] > 60:  # At least 2 seconds at 30fps
        # Compute auto-correlation for key joints (e.g., hands, feet)
        key_joints = [3, 4, 10, 13]  # Example joint indices for hands and feet
        joint_positions = pose_sequence[:, key_joints, :].reshape(pose_sequence.shape[0], -1)
        
        # Compute normalized auto-correlation (up to half sequence length)
        max_lag = min(pose_sequence.shape[0] // 2, 150)  # Up to 5 seconds at 30fps
        auto_corr = np.zeros(max_lag)
        
        for lag in range(1, max_lag):
            correlation = np.corrcoef(joint_positions[:-lag], joint_positions[lag:])[0, 1]
            auto_corr[lag] = correlation
        
        # Find peaks in auto-correlation (potential rhythmic patterns)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(auto_corr, height=0.2, distance=10)
        
        if len(peaks) > 0:
            # Average distance between peaks indicates rhythmic period
            avg_period = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
            stats['rhythm_period_frames'] = float(avg_period)
            stats['rhythm_period_seconds'] = float(avg_period / 30.0)  # Assuming 30fps
            
            # Strength of rhythmic pattern
            stats['rhythm_strength'] = float(np.mean(auto_corr[peaks]))
    
    # Create visualization if output path provided
    if output_path:
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot motion intensity over time
        ax1 = fig.add_subplot(221)
        if pose_sequence.shape[0] > 1:
            velocities = np.linalg.norm(compute_velocities(pose_sequence), axis=2)  # [seq_len-1, num_joints]
            mean_velocity = np.mean(velocities, axis=1)  # [seq_len-1]
            frames = np.arange(len(mean_velocity))
            ax1.plot(frames, mean_velocity)
            ax1.set_title("Motion Intensity")
            ax1.set_xlabel("Frame")
            ax1.set_ylabel("Average Joint Velocity")
        
        # Plot spatial distribution
        ax2 = fig.add_subplot(222, projection='3d')
        # Flatten all frames to show overall spatial distribution
        all_positions = pose_sequence.reshape(-1, 3)
        ax2.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], alpha=0.1, s=1)
        ax2.set_title("Spatial Distribution")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        
        # Plot rhythmic pattern (auto-correlation)
        ax3 = fig.add_subplot(223)
        if 'rhythm_period_frames' in stats:
            lags = np.arange(len(auto_corr))
            ax3.plot(lags, auto_corr)
            ax3.set_title("Rhythmic Pattern (Auto-correlation)")
            ax3.set_xlabel("Lag (frames)")
            ax3.set_ylabel("Correlation")
            # Plot detected peaks
            ax3.plot(peaks, auto_corr[peaks], 'ro')
        
        # Plot joint coordination
        ax4 = fig.add_subplot(224)
        if pose_sequence.shape[0] > 1:
            # Compute correlation between key joints
            key_joints = [3, 4, 10, 13]  # Example joint indices for hands and feet
            joint_names = ["Right Hand", "Left Hand", "Right Foot", "Left Foot"]
            
            # Create correlation matrix
            corr_matrix = np.zeros((len(key_joints), len(key_joints)))
            
            for i, joint1 in enumerate(key_joints):
                for j, joint2 in enumerate(key_joints):
                    # Correlation between joint velocities
                    vel1 = np.linalg.norm(velocities[:, joint1])
                    vel2 = np.linalg.norm(velocities[:, joint2])
                    corr_matrix[i, j] = np.corrcoef(vel1, vel2)[0, 1]
            
            # Plot correlation matrix
            im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_title("Joint Coordination")
            plt.colorbar(im, ax=ax4)
            ax4.set_xticks(np.arange(len(joint_names)))
            ax4.set_yticks(np.arange(len(joint_names)))
            ax4.set_xticklabels(joint_names)
            ax4.set_yticklabels(joint_names)
            plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Set overall title
        fig.suptitle("Choreography Style Analysis", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return stats


def extract_keyframes(
    pose_sequence: np.ndarray,
    num_keyframes: int = 10,
    method: str = 'kmeans'
) -> List[int]:
    """
    Extract representative keyframes from pose sequence.
    
    Args:
        pose_sequence: Pose sequence [seq_len, num_joints, 3]
        num_keyframes: Number of keyframes to extract
        method: Method for keyframe extraction ('kmeans', 'uniform', or 'velocity')
        
    Returns:
        List of keyframe indices
    """
    if pose_sequence.shape[0] <= num_keyframes:
        return list(range(pose_sequence.shape[0]))
    
    if method == 'uniform':
        # Uniform sampling
        return np.linspace(0, pose_sequence.shape[0] - 1, num_keyframes, dtype=int).tolist()
    
    elif method == 'velocity':
        # Velocity-based keyframes
        if pose_sequence.shape[0] <= 1:
            return [0]
        
        # Compute velocities
        velocities = np.linalg.norm(compute_velocities(pose_sequence), axis=(1, 2))  # [seq_len-1]
        
        # Find peaks (local maxima)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(velocities)
        
        # Also consider start and end frames
        peaks = np.concatenate([[0], peaks, [pose_sequence.shape[0] - 1]])
        
        # If too many peaks, select the highest ones
        if len(peaks) > num_keyframes:
            peak_heights = velocities[peaks[1:-1]]  # Exclude start and end frames
            highest_indices = np.argsort(peak_heights)[::-1][:num_keyframes-2]  # -2 for start and end
            selected_peaks = peaks[1:-1][highest_indices]
            keyframes = np.concatenate([[0], selected_peaks, [pose_sequence.shape[0] - 1]])
            return sorted(keyframes.tolist())
        
        # If too few peaks, add uniformly sampled frames
        elif len(peaks) < num_keyframes:
            remaining = num_keyframes - len(peaks)
            mask = np.ones(pose_sequence.shape[0], dtype=bool)
            mask[peaks] = False
            remaining_indices = np.where(mask)[0]
            
            # Select uniformly from remaining indices
            selected = np.linspace(0, len(remaining_indices) - 1, remaining, dtype=int)
            additional = remaining_indices[selected]
            
            keyframes = np.concatenate([peaks, additional])
            return sorted(keyframes.tolist())
        
        return sorted(peaks.tolist())
    
    elif method == 'kmeans':
        # K-means clustering of poses
        from sklearn.cluster import KMeans
        
        # Reshape poses for clustering
        flattened = pose_sequence.reshape(pose_sequence.shape[0], -1)  # [seq_len, num_joints*3]
        
        # Apply K-means
        kmeans = KMeans(n_clusters=num_keyframes, random_state=0).fit(flattened)
        
        # Find closest pose to each cluster center
        keyframes = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(flattened - center, axis=1)
            closest = np.argmin(distances)
            keyframes.append(closest)
        
        return sorted(keyframes)
    
    else:
        raise ValueError(f"Unsupported keyframe extraction method: {method}")
