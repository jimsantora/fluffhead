# setup.py
from setuptools import setup, find_packages

setup(
    name="fluffhead",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "datasets",
        "pyyaml",
        "tqdm",
        "pytest",
        "pytest-mock",
        "pillow",  # for image processing
        "numpy",
        "wandb",  # for experiment tracking
        "psutil",  # for system resource monitoring
    ],
)
