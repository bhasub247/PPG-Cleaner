from setuptools import setup, find_packages

setup(
    name="ppg_cleaner",
    version="0.1.0",
    description="A library for cleaning and preprocessing PPG and ABP signals.",
    author="Bharath Subramanian",
    author_email="bsubramanian@ucdavis.edu",
    url="https://github.com/bhasub247/PPG-Cleaner.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "sklearn"
    ],
    python_requires=">=3.7",
)