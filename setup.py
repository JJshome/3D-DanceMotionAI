from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dance-motion-ai",
    version="0.1.0",
    author="JJshome",
    author_email="author@example.com",
    description="3D Pose Estimation-based AI Dance Choreography Analysis System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JJshome/3D-DanceMotionAI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "scipy>=1.8.0",
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "opencv-python>=4.6.0",
        "fastdtw>=0.3.4",
        "plotly>=5.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.1.0",
        "PyWavelets>=1.3.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
        ],
        "web": [
            "flask>=2.2.0",
            "werkzeug>=2.2.0",
            "requests>=2.28.0"
        ],
        "ml": [
            "tensorboard>=2.10.0"
        ]
    },
    entry_points={
        'console_scripts': [
            'dance-motion-ai=dancemotionai.cli:main',
        ],
    },
)
