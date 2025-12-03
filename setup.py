from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ear-vae",
    version="0.1.0",
    author="Kangdi Wang, Zhiyue Wu, Dinghao Zhou, Rui Lin, Junyu Dai, Tao Jiang",
    description="High Fidelity Music Reconstruction Model - Perceptually Driven VAE for 44.1kHz Audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/polson/EAR_VAE",
    packages=["ear_vae", "ear_vae.model", "ear_vae.tools"],
    package_dir={"ear_vae": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "numpy>=1.21.0",
        "descript-audio-codec",
        "alias-free-torch",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "eval": [
            "auraloss",
        ],
    },
    package_data={
        "ear_vae": ["config/*.json", "pretrained_weight/*.pyt"],
    },
    include_package_data=True,
)
