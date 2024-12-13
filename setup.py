from setuptools import setup, find_packages

setup(
    name="ppg_cleaner",
    version="0.1.0",
    description="A library for cleaning and preprocessing PPG and ABP signals.",
    author="Bharath Subramanian",
    author_email="bsubramanian@ucdavis.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
    python_requires=">=3.7",
)