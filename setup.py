"""
Setup configuration for PyPI distribution

Install:
    pip install neurallayers

Import:
    from neurallayers import UnifiedBrainLogicNetwork
    from neurallayers.brain import BrainNetwork
    from neurallayers.consciousness import ConsciousnessEmergence
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="neurallayers",
    version="1.0.0",
    author="biblicalandr0id",
    author_email="",  # Add your email
    description="Unified Neural-Logical Network Dynamics Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biblicalandr0id/NeuralLayers",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "python-dateutil>=2.8.2",
        "typing-extensions>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
        "viz": [
            "streamlit>=1.20.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
        "monitoring": [
            "tensorboard>=2.12.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurallayers-train=train:main",
            "neurallayers-export=export_model:main",
            "neurallayers-demo=demo_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "neurallayers": ["config.yaml"],
    },
    keywords=[
        "neural networks",
        "logical reasoning",
        "consciousness",
        "brain simulation",
        "hybrid AI",
        "neural-symbolic",
        "deep learning",
        "neuroscience",
    ],
    project_urls={
        "Documentation": "https://github.com/biblicalandr0id/NeuralLayers#readme",
        "Source": "https://github.com/biblicalandr0id/NeuralLayers",
        "Tracker": "https://github.com/biblicalandr0id/NeuralLayers/issues",
    },
)
