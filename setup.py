"""
Setup script for license_plate_detection package.
"""

from setuptools import setup, find_packages

setup(
    name="license_plate_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.5.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.60.0",
    ],
    author="PROJ-H419 Team",
    author_email="example@email.com",
    description="A package for license plate detection using CNN",
    keywords="license plate, detection, computer vision, CNN",
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "license-plate-detect=license_plate_detection.cli:main",
        ],
    },
)
