# DanceMotionAI

![GitHub](https://img.shields.io/github/license/JJshome/3D-DanceMotionAI)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)

High-Precision 3D Pose Estimation-based AI Dance Choreography Analysis and Evaluation System

## Overview

DanceMotionAI is a state-of-the-art system for analyzing and evaluating dance choreographies using advanced AI techniques. The system leverages multiple cutting-edge technologies:

- 3D pose estimation with adaptive graph transformer networks
- Precise hand movement tracking with 4DHands technology
- Cross-modal analysis of dance and music
- Dynamic Time Warping (DTW) for choreography similarity analysis
- Plagiarism detection in dance choreographies

This system can be used for dance education, choreography creation, copyright protection, and academic research on dance motion.

## Features

- **High-precision 3D pose estimation**: Accurately track dancers' movements in 3D space
- **4DHands hand tracking**: Detailed tracking and analysis of hand movements for expressive gestures
- **Detailed motion analysis**: Analyze velocity, acceleration, and jerk of dance movements
- **Beat alignment**: Evaluate synchronization between dance movements and music beats
- **Dance quality metrics**: Comprehensive evaluation of dance performances
- **Choreography similarity**: Compare dance sequences to detect similarities and differences
- **Plagiarism detection**: Identify potential copyright infringement in choreographies
- **Visualization tools**: Advanced 3D visualization of poses, hand movements, and dance metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/JJshome/3D-DanceMotionAI.git
cd 3D-DanceMotionAI

# Install dependencies
pip install -e .
```

## Dependencies

- numpy
- matplotlib
- scipy
- torch
- torchvision
- opencv-python
- fastdtw
- plotly (for interactive visualizations)
- PyWavelets
- scikit-learn

## Usage

### Basic Example

```python
import numpy as np
from utils import plot_3d_pose, compute_dance_metrics

# Load your dance sequence data
reference_sequence = np.load('reference_dance.npy')
comparison_sequence = np.load('comparison_dance.npy')

# Visualize a pose from the reference sequence
plot_3d_pose(reference_sequence[0])

# Compute dance quality metrics
metrics = compute_dance_metrics(reference_sequence, comparison_sequence)
print(metrics)
```

### 4DHands Module Example

```python
import numpy as np
import torch
from models.hands_4d import Hands4D
from utils.visualization import visualize_hand_keypoints
import yaml
import matplotlib.pyplot as plt

# Load configuration
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize 4DHands model
hands_4d_config = config['hands_4d']
model = Hands4D(hands_4d_config)

# Sample hand data: [batch_size, sequence_length, num_hands, num_keypoints, 3]
hand_data = torch.randn(1, 30, 2, 21, 3)  

# Process through model
hand_features, gesture_logits = model(hand_data)

# Visualize the first frame of hand keypoints
sample_hand = hand_data[0, 0, 0].numpy()  # First batch, first frame, left hand

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

# Visualize 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
visualize_hand_keypoints(sample_hand, hand_connections, ax=ax, title="Hand Pose")
plt.show()
```

### Demo Script

Run the included demo scripts to see the system's capabilities:

```bash
# General demo
python examples/demo.py

# 4DHands module demo
python examples/hands_4d_demo.py
```

These will generate synthetic dance and hand movement sequences, analyze them, and produce visualizations demonstrating the system's features.

## Core Components

### DanceHRNet

The core 3D pose estimation engine, featuring:
- Adaptive Graph Transformer (AGT) blocks for capturing global and local dependencies
- Global-Local Adaptive Graph Convolutional Network (GLA-GCN) for handling occlusions
- Keypoint Information-based Temporal Rotation Optimization (KITRO) for improved depth accuracy

### 4DHands Module

A specialized module for precise hand tracking and gesture analysis, consisting of:
- **RAT (Relation-aware Two-Hand Tokenization)**: Models complex relationships between fingers and joints
- **SIR (Spatio-temporal Interaction Reasoning)**: Analyzes interactions between hands and their temporal dynamics
- Support for gesture recognition and hand choreography analysis
- Optimized for dance performances with expressive hand movements

### DanceFormer

Cross-modal analysis system that correlates music and dance, including:
- Cross-modal attention mechanism for music-dance alignment
- Multi-scale transformer for hierarchical choreography analysis
- Dynamic graph generation for modeling dancer interactions

### 3D-DanceDTW

Advanced similarity analysis system:
- Multi-scale Dynamic Time Warping for temporal alignment
- Wavelet-based decomposition for different time scales
- Adaptive weighting system for dance style-specific comparisons

## Module Structure

- `models/` - Core model implementations
  - `dancehrnet.py` - 3D pose estimation model
  - `hands_4d.py` - Hand tracking and analysis module
  - `danceformer.py` - Music-dance correlation model
- `utils/` - Utility functions
  - `visualization.py` - Functions for visualizing 3D poses and metrics
  - `metrics.py` - Functions for computing various dance metrics
- `examples/` - Example scripts demonstrating system usage
  - `demo.py` - Comprehensive demonstration of features
  - `hands_4d_demo.py` - Demo for the 4DHands module
- `configs/` - Configuration files

## Citation

If you use this software in your research, please cite:

```
@software{dancemotionai2025,
  author = {JJshome},
  title = {DanceMotionAI: High-Precision 3D Pose Estimation-based AI Dance Choreography Analysis and Evaluation System},
  url = {https://github.com/JJshome/3D-DanceMotionAI},
  year = {2025},
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or feedback about DanceMotionAI, please create an issue or contact the repository owner.
