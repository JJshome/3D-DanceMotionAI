"""
Visualization module for the DanceMotionAI system.

This module provides functions for visualizing 3D dance poses, 
comparison results, and similarity metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json


# Human skeleton structure (25 joint COCO format connection pairs)
SKELETON_CONNECTIONS = [
    (0, 1),   # nose to left eye
    (0, 2),   # nose to right eye
    (1, 3),   # left eye to left ear
    (2, 4),   # right eye to right ear
    (0, 5),   # nose to left shoulder
    (0, 6),   # nose to right shoulder
    (5, 7),   # left shoulder to left elbow
    (7, 9),   # left elbow to left wrist
    (6, 8),   # right shoulder to right elbow
    (8, 10),  # right elbow to right wrist
    (5, 11),  # left shoulder to left hip
    (6, 12),  # right shoulder to right hip
    (11, 13), # left hip to left knee
    (13, 15), # left knee to left ankle
    (12, 14), # right hip to right knee
    (14, 16), # right knee to right ankle
    (5, 6),   # left shoulder to right shoulder
    (11, 12)  # left hip to right hip
]

# Joint colors for different body parts
JOINT_COLORS = {
    'head': 'blue',            # head joints (0-4)
    'torso': 'green',          # torso joints (5-6, 11-12)
    'left_arm': 'red',         # left arm joints (5, 7, 9)
    'right_arm': 'magenta',    # right arm joints (6, 8, 10)
    'left_leg': 'cyan',        # left leg joints (11, 13, 15)
    'right_leg': 'yellow'      # right leg joints (12, 14, 16)
}

# Joint group definitions
JOINT_GROUPS = {
    'head': [0, 1, 2, 3, 4],
    'torso': [5, 6, 11, 12],
    'left_arm': [5, 7, 9],
    'right_arm': [6, 8, 10],
    'left_leg': [11, 13, 15],
    'right_leg': [12, 14, 16]
}


def plot_3d_pose(pose: np.ndarray, ax: Optional[plt.Axes] = None, 
                 title: Optional[str] = None, 
                 color_by_part: bool = True,
                 alpha: float = 1.0,
                 view_angles: Optional[Tuple[float, float]] = None) -> plt.Axes:
    """
    Plot a single 3D pose.
    
    Args:
        pose: 3D pose data with shape [num_joints, 3]
        ax: Optional matplotlib 3D axis to plot on
        title: Optional title for the plot
        color_by_part: Whether to color joints by body part
        alpha: Opacity of the plot elements
        view_angles: Tuple of (elevation, azimuth) viewing angles
        
    Returns:
        The matplotlib Axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = pose[:, 0]
    y = pose[:, 1]
    z = pose[:, 2]
    
    # Plot joints
    if color_by_part:
        for part_name, joint_indices in JOINT_GROUPS.items():
            color = JOINT_COLORS[part_name]
            valid_indices = [i for i in joint_indices if i < len(pose)]
            if valid_indices:
                ax.scatter(x[valid_indices], y[valid_indices], z[valid_indices], 
                           color=color, alpha=alpha, s=50)
    else:
        ax.scatter(x, y, z, color='blue', alpha=alpha, s=50)
    
    # Plot skeleton connections
    for connection in SKELETON_CONNECTIONS:
        if max(connection) < len(pose):
            i, j = connection
            if color_by_part:
                # Determine color based on the starting joint
                for part_name, joint_indices in JOINT_GROUPS.items():
                    if i in joint_indices:
                        color = JOINT_COLORS[part_name]
                        break
                else:
                    color = 'blue'
            else:
                color = 'blue'
            
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
                    color=color, alpha=alpha, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.max([np.max(x) - np.min(x), 
                        np.max(y) - np.min(y), 
                        np.max(z) - np.min(z)])
    mid_x = (np.max(x) + np.min(x)) * 0.5
    mid_y = (np.max(y) + np.min(y)) * 0.5
    mid_z = (np.max(z) + np.min(z)) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    # Set view angle if provided
    if view_angles:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    return ax


def plot_pose_sequence(pose_sequence: np.ndarray, 
                       interval: int = 50, 
                       save_path: Optional[str] = None,
                       view_angles: Optional[Tuple[float, float]] = None) -> None:
    """
    Create an animation of a 3D pose sequence.
    
    Args:
        pose_sequence: 3D pose sequence data with shape [num_frames, num_joints, 3]
        interval: Time interval between frames in milliseconds
        save_path: Optional path to save the animation as a video file (.mp4)
        view_angles: Tuple of (elevation, azimuth) viewing angles
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find global min and max for consistent scaling
    all_x = pose_sequence[:, :, 0].flatten()
    all_y = pose_sequence[:, :, 1].flatten()
    all_z = pose_sequence[:, :, 2].flatten()
    
    max_range = np.max([np.max(all_x) - np.min(all_x), 
                        np.max(all_y) - np.min(all_y), 
                        np.max(all_z) - np.min(all_z)])
    mid_x = (np.max(all_x) + np.min(all_x)) * 0.5
    mid_y = (np.max(all_y) + np.min(all_y)) * 0.5
    mid_z = (np.max(all_z) + np.min(all_z)) * 0.5
    
    # Set axis limits for consistent view
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    # Set view angle if provided
    if view_angles:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Plot the first frame
    scat = ax.scatter([], [], [], s=50)
    lines = [ax.plot([], [], [], linewidth=2)[0] for _ in SKELETON_CONNECTIONS]
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame: 0/{len(pose_sequence)-1}')
    
    def update(frame):
        """Update function for animation"""
        ax.clear()
        plot_3d_pose(pose_sequence[frame], ax=ax)
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        if view_angles:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
        ax.set_title(f'Frame: {frame}/{len(pose_sequence)-1}')
        return []
    
    anim = FuncAnimation(fig, update, frames=len(pose_sequence), 
                         interval=interval, blit=False)
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=1000/interval)
    
    plt.tight_layout()
    plt.show()


def plot_pose_comparison(pose1: np.ndarray, pose2: np.ndarray,
                         title1: str = 'Reference', title2: str = 'Comparison',
                         joint_errors: Optional[np.ndarray] = None,
                         view_angles: Optional[Tuple[float, float]] = None) -> None:
    """
    Plot two 3D poses side by side for comparison.
    
    Args:
        pose1: First 3D pose data with shape [num_joints, 3]
        pose2: Second 3D pose data with shape [num_joints, 3]
        title1: Title for the first pose plot
        title2: Title for the second pose plot
        joint_errors: Optional array of error values for each joint
        view_angles: Tuple of (elevation, azimuth) viewing angles
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Plot first pose
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_pose(pose1, ax=ax1, title=title1, view_angles=view_angles)
    
    # Plot second pose
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_pose(pose2, ax=ax2, title=title2, view_angles=view_angles)
    
    # If joint errors are provided, adjust the color intensity based on error
    if joint_errors is not None:
        # Normalize errors to [0, 1] range for colormapping
        if len(joint_errors) > 0:
            normalized_errors = (joint_errors - np.min(joint_errors)) / (np.max(joint_errors) - np.min(joint_errors))
            
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap='plasma', 
                                       norm=plt.Normalize(vmin=np.min(joint_errors), 
                                                          vmax=np.max(joint_errors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='vertical', pad=0.05)
            cbar.set_label('Joint Error (mm)')
            
            # Plot points with error-based colors in the second plot
            scatter_points = ax2.scatter(pose2[:, 0], pose2[:, 1], pose2[:, 2], 
                                         c=joint_errors, cmap='plasma', s=100)
    
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmap(similarity_matrix: np.ndarray, 
                            xlabel: str = 'Reference Frames', 
                            ylabel: str = 'Comparison Frames',
                            title: str = 'Pose Similarity Matrix',
                            cmap: str = 'viridis',
                            figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot a heatmap of similarity scores between two dance sequences.
    
    Args:
        similarity_matrix: 2D array of similarity scores
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        title: Title for the plot
        cmap: Colormap name
        figsize: Figure size in inches (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(similarity_matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar(img, ax=ax, label='Similarity Score')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add tick marks
    x_ticks = np.linspace(0, similarity_matrix.shape[1]-1, min(10, similarity_matrix.shape[1]))
    y_ticks = np.linspace(0, similarity_matrix.shape[0]-1, min(10, similarity_matrix.shape[0]))
    ax.set_xticks(x_ticks.astype(int))
    ax.set_yticks(y_ticks.astype(int))
    
    plt.tight_layout()
    plt.show()


def plot_joint_trajectories(pose_sequence: np.ndarray, 
                            joint_indices: List[int],
                            joint_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (15, 15)) -> None:
    """
    Plot trajectories of selected joints over time.
    
    Args:
        pose_sequence: 3D pose sequence data with shape [num_frames, num_joints, 3]
        joint_indices: List of joint indices to plot
        joint_names: Optional list of joint names for the legend
        figsize: Figure size in inches (width, height)
    """
    if joint_names is None:
        joint_names = [f"Joint {i}" for i in joint_indices]
    
    fig = plt.figure(figsize=figsize)
    
    # Plot X coordinates
    ax1 = fig.add_subplot(311)
    for i, joint_idx in enumerate(joint_indices):
        ax1.plot(pose_sequence[:, joint_idx, 0], label=joint_names[i])
    ax1.set_title('X Coordinate Over Time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Position')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Y coordinates
    ax2 = fig.add_subplot(312)
    for i, joint_idx in enumerate(joint_indices):
        ax2.plot(pose_sequence[:, joint_idx, 1], label=joint_names[i])
    ax2.set_title('Y Coordinate Over Time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y Position')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Z coordinates
    ax3 = fig.add_subplot(313)
    for i, joint_idx in enumerate(joint_indices):
        ax3.plot(pose_sequence[:, joint_idx, 2], label=joint_names[i])
    ax3.set_title('Z Coordinate Over Time')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Z Position')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_dtw_alignment_path(dtw_path: List[Tuple[int, int]], 
                            similarity_matrix: np.ndarray,
                            title: str = 'DTW Alignment Path',
                            figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot the DTW alignment path overlaid on a similarity matrix.
    
    Args:
        dtw_path: List of (i, j) index pairs representing the alignment path
        similarity_matrix: 2D array of similarity scores
        title: Title for the plot
        figsize: Figure size in inches (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot similarity matrix as heatmap
    img = ax.imshow(similarity_matrix, interpolation='nearest', cmap='viridis', alpha=0.7)
    plt.colorbar(img, ax=ax, label='Similarity Score')
    
    # Extract path coordinates
    path_i, path_j = zip(*dtw_path)
    
    # Plot the alignment path
    ax.plot(path_j, path_i, 'r-', linewidth=2, label='DTW Path')
    ax.plot(path_j, path_i, 'ro', markersize=4)
    
    ax.set_xlabel('Reference Frames')
    ax.set_ylabel('Comparison Frames')
    ax.set_title(title)
    ax.legend()
    
    # Add tick marks
    x_ticks = np.linspace(0, similarity_matrix.shape[1]-1, min(10, similarity_matrix.shape[1]))
    y_ticks = np.linspace(0, similarity_matrix.shape[0]-1, min(10, similarity_matrix.shape[0]))
    ax.set_xticks(x_ticks.astype(int))
    ax.set_yticks(y_ticks.astype(int))
    
    plt.tight_layout()
    plt.show()


def plot_radar_metrics(metrics: Dict[str, float], 
                      title: str = 'Dance Performance Metrics',
                      figsize: Tuple[int, int] = (10, 10)) -> None:
    """
    Plot a radar chart of dance performance metrics.
    
    Args:
        metrics: Dictionary of metric names and their values (0-1 range)
        title: Title for the plot
        figsize: Figure size in inches (width, height)
    """
    # Number of metrics
    num_metrics = len(metrics)
    
    # Compute angles for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Extract names and values
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    metric_values += metric_values[:1]  # Close the polygon
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Draw the chart
    ax.plot(angles, metric_values, 'o-', linewidth=2)
    ax.fill(angles, metric_values, alpha=0.25)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    
    # Set y-ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add title
    plt.title(title)
    
    plt.tight_layout()
    plt.show()


def save_visualization_to_html(pose_sequence: np.ndarray, output_path: str) -> None:
    """
    Save a 3D pose sequence visualization as an interactive HTML file.
    
    Args:
        pose_sequence: 3D pose sequence data with shape [num_frames, num_joints, 3]
        output_path: Path to save the HTML file
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly is required for HTML visualization. Install with: pip install plotly")
        return
    
    # Create figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # Create frame data for animation
    frames = []
    
    for frame_idx in range(len(pose_sequence)):
        frame_data = pose_sequence[frame_idx]
        
        # Prepare data for this frame
        x, y, z = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
        
        # Create lines for skeleton
        lines_x, lines_y, lines_z = [], [], []
        
        for connection in SKELETON_CONNECTIONS:
            if max(connection) < len(frame_data):
                i, j = connection
                lines_x.extend([x[i], x[j], None])
                lines_y.extend([y[i], y[j], None])
                lines_z.extend([z[i], z[j], None])
        
        # Add frame data
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(size=8, color='blue'),
                    name='Joints'
                ),
                go.Scatter3d(
                    x=lines_x, y=lines_y, z=lines_z,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Skeleton'
                )
            ],
            name=f'frame_{frame_idx}'
        )
        frames.append(frame)
    
    # Initial data (first frame)
    first_frame = pose_sequence[0]
    x, y, z = first_frame[:, 0], first_frame[:, 1], first_frame[:, 2]
    
    # Initial skeleton lines
    lines_x, lines_y, lines_z = [], [], []
    for connection in SKELETON_CONNECTIONS:
        if max(connection) < len(first_frame):
            i, j = connection
            lines_x.extend([x[i], x[j], None])
            lines_y.extend([y[i], y[j], None])
            lines_z.extend([z[i], z[j], None])
    
    # Add traces for initial view
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Joints'
        )
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color='red', width=2),
            name='Skeleton'
        )
    )
    
    # Update figure layout
    fig.update_layout(
        title='3D Pose Animation',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                dict(
                    label='Play',
                    method='animate',
                    args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                )
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'y': 0,
            'xanchor': 'right',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 50, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [f'frame_{i}'],
                        {'frame': {'duration': 50, 'redraw': True}, 'mode': 'immediate'}
                    ],
                    'label': str(i),
                    'method': 'animate'
                } for i in range(len(pose_sequence))
            ]
        }]
    )
    
    # Add frames to the figure
    fig.frames = frames
    
    # Save as HTML
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")


def export_3d_model(pose: np.ndarray, output_path: str) -> None:
    """
    Export a 3D pose as a 3D model file (OBJ format).
    
    Args:
        pose: 3D pose data with shape [num_joints, 3]
        output_path: Path to save the OBJ file
    """
    # Basic sphere mesh to represent joints
    def create_sphere(radius, center, resolution=10):
        vertices = []
        faces = []
        
        # Create vertices
        for i in range(resolution + 1):
            lat = np.pi * i / resolution
            for j in range(resolution * 2):
                lon = 2 * np.pi * j / (resolution * 2)
                x = center[0] + radius * np.sin(lat) * np.cos(lon)
                y = center[1] + radius * np.sin(lat) * np.sin(lon)
                z = center[2] + radius * np.cos(lat)
                vertices.append((x, y, z))
        
        # Create faces
        for i in range(resolution):
            for j in range(resolution * 2):
                p1 = i * (resolution * 2) + j
                p2 = p1 + 1
                if j == resolution * 2 - 1:
                    p2 = i * (resolution * 2)
                p3 = p1 + (resolution * 2)
                p4 = p2 + (resolution * 2)
                if p4 >= len(vertices):
                    p4 = p4 - len(vertices)
                if p3 >= len(vertices):
                    p3 = p3 - len(vertices)
                
                faces.append((p1 + 1, p2 + 1, p4 + 1))
                faces.append((p1 + 1, p4 + 1, p3 + 1))
        
        return vertices, faces
    
    # Create OBJ file
    with open(output_path, 'w') as f:
        f.write("# 3D Pose as OBJ file\n")
        
        # Joint spheres
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for joint_idx in range(len(pose)):
            center = pose[joint_idx]
            radius = 0.05  # Adjust radius as needed
            
            vertices, faces = create_sphere(radius, center)
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Update faces with correct vertex indices
            for face in faces:
                all_faces.append((face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset))
            
            vertex_offset += len(vertices)
            all_vertices.extend(vertices)
        
        # Write skeleton lines as cylinders (simplified as edges here)
        for connection in SKELETON_CONNECTIONS:
            if max(connection) < len(pose):
                i, j = connection
                f.write(f"l {i+1} {j+1}\n")
        
        # Write faces
        for face in all_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"3D model saved to {output_path}")
