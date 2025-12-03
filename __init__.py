"""
εar-VAE: High Fidelity Music Reconstruction Model

A perceptually-driven VAE for high-quality 44.1kHz audio reconstruction.

Example usage:
    from ear_vae import Encoder, Decoder

    # Initialize encoder and decoder
    encoder = Encoder(
        checkpoint_path='./pretrained_weight/ear_vae_44k.pyt',
        config_path='./config/model_config.json'
    )
    decoder = Decoder(
        checkpoint_path='./pretrained_weight/ear_vae_44k.pyt',
        config_path='./config/model_config.json'
    )

    # Encode audio to latents
    latents = encoder(audio)

    # Decode latents back to audio
    reconstructed = decoder(latents)

For more information, see: https://github.com/polson/EAR_VAE
"""

__version__ = "0.1.0"

from .model import (
    EAR_VAE,
    Encoder,
    Decoder,
    OobleckEncoder,
    OobleckDecoder,
    ContinuousTransformer,
)

__all__ = [
    "EAR_VAE",
    "Encoder",
    "Decoder",
    "OobleckEncoder",
    "OobleckDecoder",
    "ContinuousTransformer",
]
