# DanceMotionAI

![GitHub](https://img.shields.io/github/license/JJshome/3D-DanceMotionAI)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)

High-Precision 3D Pose Estimation-based AI Dance Choreography Analysis and Evaluation System

## Overview

DanceMotionAI is a state-of-the-art system for analyzing and evaluating dance choreographies using advanced AI techniques. The system leverages multiple cutting-edge technologies:

- 3D pose estimation with adaptive graph transformer networks
- Cross-modal analysis of dance and music
- Dynamic Time Warping (DTW) for choreography similarity analysis
- Plagiarism detection in dance choreographies

This system can be used for dance education, choreography creation, copyright protection, and academic research on dance motion.

## Features

- **High-precision 3D pose estimation**: Accurately track dancers' movements in 3D space
- **Detailed motion analysis**: Analyze velocity, acceleration, and jerk of dance movements
- **Beat alignment**: Evaluate synchronization between dance movements and music beats
- **Dance quality metrics**: Comprehensive evaluation of dance performances
- **Choreography similarity**: Compare dance sequences to detect similarities and differences
- **Plagiarism detection**: Identify potential copyright infringement in choreographies
- **Visualization tools**: Advanced 3D visualization of poses and dance metrics

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
- fastdtw
- plotly (for interactive visualizations)

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

### Demo Script

Run the included demo script to see the system's capabilities:

```bash
python examples/demo.py
```

This will generate synthetic dance sequences, analyze them, and produce visualizations demonstrating the system's features.

## Module Structure

- `utils/` - Utility functions
  - `visualization.py` - Functions for visualizing 3D poses and metrics
  - `metrics.py` - Functions for computing various dance metrics
- `examples/` - Example scripts demonstrating system usage
  - `demo.py` - Comprehensive demonstration of features

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
