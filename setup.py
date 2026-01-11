"""
FlowShield-UDRL: Safe Command-Conditioned RL via Flow Matching

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="flowshield-udrl",
    version="0.1.0",
    author="FlowShield Team",
    author_email="flowshield@research.ai",
    description="Safe Command-Conditioned Reinforcement Learning via Flow Matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/flowshield-udrl",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/flowshield-udrl/issues",
        "Documentation": "https://flowshield-udrl.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "wandb>=0.15.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "torchdyn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.23.0",
            "ipywidgets>=8.0.0",
        ],
        "d4rl": [
            "d4rl>=1.1",
            "mujoco>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flowshield-train=scripts.train:main",
            "flowshield-eval=scripts.evaluate:main",
            "flowshield-collect=scripts.collect_data:main",
        ],
    },
)
