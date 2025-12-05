"""
Setup script for AI training sub-project.
"""

from setuptools import setup, find_packages

setup(
    name="brain-scan-training",
    version="1.0.0",
    description="AI training pipeline for healthy brain scan analysis",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "nibabel>=5.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.8",
)

