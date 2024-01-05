from setuptools import setup, find_packages

setup(
    name="glitchpy",
    version="0.0.2",
    install_requires=[
        'imageio',
        'matplotlib',
        'numpy',
        'scikit-image',
    ],
    packages=find_packages(),
)

