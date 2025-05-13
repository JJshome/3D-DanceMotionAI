# DanceMotionAI: 3D Dance Choreography Analysis and Evaluation System

![DanceMotionAI Logo](https://via.placeholder.com/800x200?text=DanceMotionAI)

## Overview

DanceMotionAI is an advanced AI-powered dance movement analysis system with high-precision 3D pose estimation, choreography comparison, and detailed feedback. The system uses cutting-edge deep learning techniques to analyze dance performances, evaluate technique, and detect choreography similarities.

### Key Features

- **High-Precision 3D Pose Estimation**: Using our proprietary DanceHRNet with adaptive graph transformer network technology for accurate joint tracking
- **Hand Movement Analysis**: Detailed hand gesture tracking with our 4DHands module for comprehensive analysis of even the smallest movements
- **Music-Movement Correlation**: Advanced analysis of how dance movements synchronize with music using our DanceFormer multimodal analysis system
- **Choreography Comparison**: Compare dance performances to reference videos with our 3D-DanceDTW algorithm for accurate similarity analysis
- **Performance Metrics**: Comprehensive statistics on movement quality, timing accuracy, synchronization, and more
- **3D Visualization**: Interactive 3D visualizations of dance movements for detailed analysis

## Project Structure

```
3D-DanceMotionAI/
├── configs/                   # Configuration files
│   └── default_config.yaml    # Default configuration settings
├── data/                      # Data handling modules
│   ├── data_loader.py         # Data loading utilities
│   ├── data_augmentation.py   # Data augmentation techniques
│   └── dataset.py             # Dataset classes
├── examples/                  # Example scripts
│   └── integrated_demo.py     # Demo script showing the complete pipeline
├── interfaces/                # User interfaces
│   ├── web_interface.py       # Web-based interface
│   ├── mobile_interface.py    # Mobile app interface
│   ├── ar_vr_interface.py     # AR/VR interface
│   └── templates/             # HTML templates for web interface
│       ├── index.html         # Main page template
│       ├── processing.html    # Processing page template
│       └── results.html       # Results page template
├── models/                    # Model implementations
│   ├── dancehrnet.py          # DanceHRNet model for 3D pose estimation
│   ├── hands_4d.py            # 4DHands model for hand movement analysis
│   ├── danceformer.py         # DanceFormer model for multimodal analysis
│   ├── dance_dtw.py           # 3D-DanceDTW model for similarity analysis
│   └── music_analysis.py      # Music analysis module
├── utils/                     # Utility functions
│   ├── visualization.py       # Visualization utilities
│   └── metrics.py             # Performance metrics calculation
├── weights/                   # Pre-trained model weights (not included in repo)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.3+ (for GPU acceleration)
- FFmpeg (for audio extraction)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/JJshome/3D-DanceMotionAI.git
   cd 3D-DanceMotionAI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained model weights:
   ```bash
   python scripts/download_weights.py
   ```

## Usage

### Command-line Interface

You can analyze dance videos using the integrated demo script:

```bash
python examples/integrated_demo.py --video path/to/video.mp4 --output_dir output
```

For choreography comparison:

```bash
python examples/integrated_demo.py --video path/to/video.mp4 --reference path/to/reference.mp4 --output_dir output
```

### Web Interface

1. Start the web server:
   ```bash
   python interfaces/web_interface.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload your dance video and follow the on-screen instructions

### API

DanceMotionAI can also be used programmatically:

```python
from models.dancehrnet import create_model as create_dancehrnet
from models.hands_4d import Hands4D
from models.danceformer import create_model as create_danceformer
from models.dance_dtw import create_model as create_dance_dtw
from examples.integrated_demo import process_video

# Load configuration and models
config = load_config('configs/default_config.yaml')
dancehrnet = create_dancehrnet(config['pose_estimation'])
hands_4d = Hands4D(config['hands_4d'])

# Process a video
frames, fps = load_video('path/to/video.mp4')
processed_frames = preprocess_frames(frames)
poses_3d, hand_poses = process_video(processed_frames, dancehrnet, hands_4d, device)

# Analyze performance
metrics = compute_dance_metrics(poses_3d, hand_poses)
```

## Configuration

You can customize the behavior of DanceMotionAI by modifying the configuration file at `configs/default_config.yaml`. Some key parameters include:

- `pose_estimation.confidence_threshold`: Minimum confidence for keypoints
- `multimodal_analysis.temporal_window`: Number of frames to consider for temporal analysis
- `similarity_analysis.similarity_threshold`: Threshold for similarity detection
- `visualization.*`: Various visualization settings

## System Components

### DanceHRNet

Our 3D pose estimation model combines graph convolutional networks with transformer architecture to achieve high precision in pose tracking. Key innovations include:

- **Adaptive Graph Transformer (AGT) Blocks**: Combine the global context modeling of transformers with the local structural reasoning of graph neural networks
- **Global-Local Adaptive Graph Convolutional Network (GLA-GCN)**: Specialized for handling occlusions and complex movements
- **Keypoint Information-based Rotation Optimization (KITRO)**: Improves depth estimation accuracy

### 4DHands

The hand movement analysis module provides detailed tracking of hand gestures with:

- **Relation-aware Two-Hand Tokenization (RAT)**: Models relationships between finger joints
- **Spatio-temporal Interaction Reasoning (SIR)**: Analyzes how both hands interact over time

### DanceFormer

Our multimodal analysis system correlates dance movements with music features using:

- **Cross-modal Attention Mechanism**: Aligns specific music elements (beats, melody) with dance movements
- **Multi-scale Transformer**: Analyzes movement patterns at different time scales (frame, phrase, sequence)
- **Dynamic Graph Generation**: Captures dancer interactions and temporal relationships

### 3D-DanceDTW

The choreography comparison module uses an enhanced Dynamic Time Warping algorithm specifically designed for 3D dance movement data:

- **3D Pose-based Distance Metric**: Considers joint positions, velocities, accelerations, and angles
- **Multi-scale DTW**: Analyzes similarity at different temporal scales using wavelet transform
- **Style-adaptive Weighting**: Adapts comparison metrics based on dance style

## Applications

DanceMotionAI can be used in various scenarios including:

- **Dance Education**: Provide detailed feedback and guidance to dance students
- **Choreography Creation**: Assist choreographers in developing and refining new dance routines
- **Performance Evaluation**: Objectively assess dance performances in competitions
- **Copyright Protection**: Identify choreography similarities for intellectual property protection
- **Virtual Reality**: Create realistic avatar movements for VR dance applications

## Contributing

We welcome contributions to improve DanceMotionAI! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DanceMotionAI in your research, please cite our paper:

```
@article{jang2025dancemotionai,
  title={DanceMotionAI: High-Precision 3D Pose Estimation-based AI Dance Choreography Analysis and Evaluation System},
  author={Jang, Jiwhan and Kim, Hoyeon},
  journal={Proceedings of the International Conference on Computer Vision},
  year={2025}
}
```

## Contact

- Jiwhan Jang - jjshome@example.com
- Project Link: [https://github.com/JJshome/3D-DanceMotionAI](https://github.com/JJshome/3D-DanceMotionAI)

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
- [MediaPipe](https://mediapipe.dev/)
- [Librosa](https://librosa.org/)
