import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor, nn, no_grad
from .autoencoders import OobleckDecoder, OobleckEncoder

from .transformer import ContinuousTransformer
LRELU_SLOPE = 0.1
padding_mode = "zeros"
sample_eps = 1e-6

def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale)
    var = stdev * stdev + sample_eps
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    
    return latents, kl


class EAR_VAE(nn.Module):

    def __init__(self, model_config: dict = None):
        super().__init__()

        if model_config is None:
            model_config = {
                "encoder": {
                    "config": {
                        "in_channels": 2,
                        "channels": 128,
                        "c_mults": [1, 2, 4, 8, 16],
                        "strides": [2, 4, 4, 4, 8],
                        "latent_dim": 128,
                        "use_snake": True
                    }
                },
                "decoder": {
                    "config": {
                        "out_channels": 2,
                        "channels": 128,
                        "c_mults": [1, 2, 4, 8, 16],
                        "strides": [2, 4, 4, 4, 8],
                        "latent_dim": 64,
                        "use_nearest_upsample": False,
                        "use_snake": True,
                        "final_tanh": False,
                    },
                },
                "latent_dim": 64,
                "downsampling_ratio": 1024,
                "io_channels": 2,
            }
        else:
            model_config = model_config

        if model_config.get("transformer") is not None:
            self.transformers = ContinuousTransformer(
                dim=model_config["decoder"]["config"]["latent_dim"],
                depth=model_config["transformer"]["depth"],
                **model_config["transformer"].get("config", {}),
            )
        else:
            self.transformers = None

        self.encoder = OobleckEncoder(**model_config["encoder"]["config"])
        self.decoder = OobleckDecoder(**model_config["decoder"]["config"])

    def forward(self, audio) -> Tensor:
        """
        audio: Input audio tensor [B,C,T]
        """
        status = self.encoder(audio)
        mean, scale = status.chunk(2, dim=1)
        z, kl = vae_sample(mean, scale)
        
        if self.transformers is not None:
            z = z.permute(0, 2, 1)
            z = self.transformers(z)
            z = z.permute(0, 2, 1)

        x = self.decoder(z)
        return x, kl

    def encode(self, audio, use_sample=True):
        x = self.encoder(audio)
        mean, scale = x.chunk(2, dim=1)
        if use_sample:
            z, _ = vae_sample(mean, scale)
        else:
            z = mean
        return z

    def decode(self, z):
        
        if self.transformers is not None:
            z = z.permute(0, 2, 1)
            z = self.transformers(z)
            z = z.permute(0, 2, 1)
            
        x = self.decoder(z)
        return x

    @no_grad()
    def inference(self, audio):
        z = self.encode(audio)
        recon_audio = self.decode(z)
        return recon_audio


class Encoder(nn.Module):
    """
    Standalone Encoder module that loads from a pretrained EAR_VAE checkpoint.

    Args:
        checkpoint_path: Path to the pretrained model checkpoint (.pyt file)
        config_path: Path to the model configuration JSON file
        use_sample: If True, sample from the latent distribution. If False, use mean only.

    Example:
        encoder = Encoder(
            checkpoint_path='./pretrained_weight/ear_vae_44k.pyt',
            config_path='./config/model_config.json'
        )
        latents = encoder(audio)  # audio: [B, C, T] -> latents: [B, latent_dim, T//downsampling_ratio]
    """

    def __init__(self, checkpoint_path: str, config_path: str, use_sample: bool = True):
        super().__init__()

        import json

        # Load config
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        # Store config and settings
        self.use_sample = use_sample
        self.model_config = model_config

        # Initialize encoder
        self.encoder = OobleckEncoder(**model_config["encoder"]["config"])

        # Load checkpoint and extract encoder weights
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        encoder_state_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }
        self.encoder.load_state_dict(encoder_state_dict)

    def forward(self, audio: Tensor) -> Tensor:
        """
        Encode audio to latent representation.

        Args:
            audio: Input audio tensor [B, C, T]

        Returns:
            latents: Latent representation [B, latent_dim, T//downsampling_ratio]
        """
        x = self.encoder(audio)
        mean, scale = x.chunk(2, dim=1)

        if self.use_sample:
            z, _ = vae_sample(mean, scale)
        else:
            z = mean

        return z

    @no_grad()
    def encode(self, audio: Tensor) -> Tensor:
        """Convenience method for inference."""
        return self.forward(audio)


class Decoder(nn.Module):
    """
    Standalone Decoder module that loads from a pretrained EAR_VAE checkpoint.
    Includes the optional transformer if present in the model configuration.

    Args:
        checkpoint_path: Path to the pretrained model checkpoint (.pyt file)
        config_path: Path to the model configuration JSON file

    Example:
        decoder = Decoder(
            checkpoint_path='./pretrained_weight/ear_vae_44k.pyt',
            config_path='./config/model_config.json'
        )
        audio = decoder(latents)  # latents: [B, latent_dim, T] -> audio: [B, C, T*downsampling_ratio]
    """

    def __init__(self, checkpoint_path: str, config_path: str):
        super().__init__()

        import json

        # Load config
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        self.model_config = model_config

        # Initialize transformer if present in config
        if model_config.get("transformer") is not None:
            self.transformers = ContinuousTransformer(
                dim=model_config["decoder"]["config"]["latent_dim"],
                depth=model_config["transformer"]["depth"],
                **model_config["transformer"].get("config", {}),
            )
        else:
            self.transformers = None

        # Initialize decoder
        self.decoder = OobleckDecoder(**model_config["decoder"]["config"])

        # Load checkpoint and extract decoder (and transformer) weights
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        decoder_state_dict = {
            k.replace('decoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('decoder.')
        }
        self.decoder.load_state_dict(decoder_state_dict)

        if self.transformers is not None:
            transformer_state_dict = {
                k.replace('transformers.', ''): v
                for k, v in state_dict.items()
                if k.startswith('transformers.')
            }
            self.transformers.load_state_dict(transformer_state_dict)

    def forward(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to audio.

        Args:
            z: Latent representation [B, latent_dim, T]

        Returns:
            audio: Reconstructed audio [B, C, T*downsampling_ratio]
        """
        if self.transformers is not None:
            z = z.permute(0, 2, 1)
            z = self.transformers(z)
            z = z.permute(0, 2, 1)

        x = self.decoder(z)
        return x

    @no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Convenience method for inference."""
        return self.forward(z)
