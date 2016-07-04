from setuptools import setup, find_packages    

setup(
    name='tensormodels', 
    version='0.0.1',
    packages=find_packages(),
    author="Ashesh Jain",
    install_requires=[
        "numpy >= 1.10.1",
        "tensorflow >= 0.9.0",
        "setuptools >= 1.1.6",
    ],
)
