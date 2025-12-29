"""Setup configuration for VQVAEAudioGenerator package."""

from setuptools import setup, find_packages

setup(
    name='vqvae-audio-generator',
    version='0.1.0',
    description='VQVAE Audio Generator',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'tensorflow[and-cuda]>=2.15.0,<3.0.0',
        'librosa>=0.10.0',
        'numpy>=1.24.0,<2.0.0',
        'matplotlib>=3.7.0',
        'tinytag>=1.8.0',
        'pyaudio>=0.2.13',
    ],
)

