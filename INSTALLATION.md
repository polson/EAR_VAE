# EAR-VAE Installation Guide

## Quick Install from GitHub

The easiest way to use the Encoder and Decoder modules is to install directly from GitHub:

```bash
pip install git+https://github.com/polson/EAR_VAE.git
```

After installation, download the pretrained weights from [Hugging Face](https://huggingface.co/earlab/EAR_VAE).

## Usage After Installation

Once installed via pip, you can import and use the modules in your Python code:

```python
from ear_vae import Encoder, Decoder

# Initialize with paths to checkpoint and config
encoder = Encoder(
    checkpoint_path='path/to/ear_vae_44k.pyt',
    config_path='path/to/model_config.json'
)

decoder = Decoder(
    checkpoint_path='path/to/ear_vae_44k.pyt',
    config_path='path/to/model_config.json'
)

# Use them
latents = encoder(audio)
reconstructed = decoder(latents)
```

## Development Installation

For development or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/polson/EAR_VAE.git
cd EAR_VAE

# Install in editable mode
pip install -e .
```

## Dependencies

The package will automatically install:
- `torch>=1.13.0`
- `torchaudio>=0.13.0`
- `numpy>=1.21.0`
- `descript-audio-codec`
- `alias-free-torch`

Note: FFmpeg is also required for audio loading. Install it with:
```bash
conda install -c conda-forge 'ffmpeg<7'
```

## Verifying Installation

Test that the package is installed correctly:

```python
import ear_vae
print(ear_vae.__version__)  # Should print: 0.1.0

from ear_vae import Encoder, Decoder, EAR_VAE
print("✓ All imports successful!")
```

## Package Structure

After installation, the package provides:

- `ear_vae.Encoder` - Standalone encoder module
- `ear_vae.Decoder` - Standalone decoder module
- `ear_vae.EAR_VAE` - Full VAE model
- `ear_vae.OobleckEncoder` - Encoder architecture
- `ear_vae.OobleckDecoder` - Decoder architecture
- `ear_vae.ContinuousTransformer` - Transformer component

## Troubleshooting

### ModuleNotFoundError: No module named 'ear_vae'

Make sure you've installed the package first:
```bash
pip install git+https://github.com/polson/EAR_VAE.git
```

### ModuleNotFoundError: No module named 'torch'

Install PyTorch first:
```bash
pip install torch torchaudio
```

### Permission errors during installation

Use `--user` flag:
```bash
pip install --user git+https://github.com/polson/EAR_VAE.git
```

Or install in a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install git+https://github.com/polson/EAR_VAE.git
```
