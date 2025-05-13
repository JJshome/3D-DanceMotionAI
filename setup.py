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
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "fastdtw>=0.3.4",
        "plotly>=5.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
        ],
    },
)
