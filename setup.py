from setuptools import setup, find_packages

setup(
    name="glitchpy",
    version="0.0.1",
    install_requires=[
        'imageio',
        'matplotlib',
        'numpy',
        'scikit-image',
    ],
    packages=find_packages(),
)

