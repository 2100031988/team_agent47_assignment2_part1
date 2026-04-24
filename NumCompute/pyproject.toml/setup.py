from setuptools import setup, find_packages

setup(
    name="numcompute",
    version="0.1.0",
    description="A lightweight NumPy-based ML framework",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    python_requires=">=3.8",
)