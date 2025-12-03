"""
Example usage of the standalone Encoder and Decoder modules.

This script demonstrates how to use the Encoder and Decoder modules
separately for encoding audio to latents and decoding latents back to audio.
"""

import torch
import torchaudio
from model.ear_vae import Encoder, Decoder


def example_full_reconstruction():
    """Example: Encode and decode audio using separate modules."""

    # Paths to your pretrained checkpoint and config
    checkpoint_path = './pretrained_weight/ear_vae_44k.pyt'
    config_path = './config/model_config.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize encoder and decoder
    print("Loading Encoder...")
    encoder = Encoder(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        use_sample=True  # Set to False to use mean only (deterministic)
    ).to(device).eval()

    print("Loading Decoder...")
    decoder = Decoder(
        checkpoint_path=checkpoint_path,
        config_path=config_path
    ).to(device).eval()

    # Load audio
    audio_path = './data/example.wav'  # Replace with your audio file
    audio, sr = torchaudio.load(audio_path)

    # Preprocess audio
    if sr != 44100:
        audio = torchaudio.transforms.Resample(sr, 44100)(audio)

    # Convert to stereo if mono
    if audio.shape[0] == 1:
        audio = torch.cat([audio, audio], dim=0)

    # Add batch dimension and move to device
    audio = audio.unsqueeze(0).to(device)  # [1, 2, T]

    print(f"Input audio shape: {audio.shape}")

    # Encode audio to latents
    with torch.no_grad():
        latents = encoder(audio)

    print(f"Latent shape: {latents.shape}")

    # Decode latents back to audio
    with torch.no_grad():
        reconstructed_audio = decoder(latents)

    print(f"Reconstructed audio shape: {reconstructed_audio.shape}")

    # Save reconstructed audio
    output_path = './results/reconstructed.wav'
    torchaudio.save(
        output_path,
        reconstructed_audio.squeeze(0).cpu(),
        sample_rate=44100
    )
    print(f"Saved reconstructed audio to {output_path}")


def example_encode_only():
    """Example: Only encode audio to latents for downstream tasks."""

    checkpoint_path = './pretrained_weight/ear_vae_44k.pyt'
    config_path = './config/model_config.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize only the encoder
    encoder = Encoder(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        use_sample=False  # Use mean only for consistent latents
    ).to(device).eval()

    # Load and preprocess audio
    audio_path = './data/example.wav'
    audio, sr = torchaudio.load(audio_path)

    if sr != 44100:
        audio = torchaudio.transforms.Resample(sr, 44100)(audio)
    if audio.shape[0] == 1:
        audio = torch.cat([audio, audio], dim=0)

    audio = audio.unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        latents = encoder(audio)

    print(f"Encoded latents shape: {latents.shape}")

    # Save latents for later use
    torch.save(latents.cpu(), './results/latents.pt')
    print("Saved latents to ./results/latents.pt")

    return latents


def example_decode_only():
    """Example: Only decode latents to audio."""

    checkpoint_path = './pretrained_weight/ear_vae_44k.pyt'
    config_path = './config/model_config.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize only the decoder
    decoder = Decoder(
        checkpoint_path=checkpoint_path,
        config_path=config_path
    ).to(device).eval()

    # Load previously saved latents
    latents = torch.load('./results/latents.pt').to(device)

    # Decode
    with torch.no_grad():
        audio = decoder(latents)

    print(f"Decoded audio shape: {audio.shape}")

    # Save audio
    torchaudio.save(
        './results/decoded.wav',
        audio.squeeze(0).cpu(),
        sample_rate=44100
    )
    print("Saved decoded audio to ./results/decoded.wav")


def example_latent_manipulation():
    """Example: Encode, manipulate latents, then decode."""

    checkpoint_path = './pretrained_weight/ear_vae_44k.pyt'
    config_path = './config/model_config.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = Encoder(checkpoint_path, config_path).to(device).eval()
    decoder = Decoder(checkpoint_path, config_path).to(device).eval()

    # Load audio
    audio, sr = torchaudio.load('./data/example.wav')
    if sr != 44100:
        audio = torchaudio.transforms.Resample(sr, 44100)(audio)
    if audio.shape[0] == 1:
        audio = torch.cat([audio, audio], dim=0)
    audio = audio.unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode
        latents = encoder(audio)

        # Manipulate latents (example: scale by 0.8)
        modified_latents = latents * 0.8

        # Decode original
        original_recon = decoder(latents)

        # Decode modified
        modified_recon = decoder(modified_latents)

    # Save both versions
    torchaudio.save('./results/original_recon.wav', original_recon.squeeze(0).cpu(), 44100)
    torchaudio.save('./results/modified_recon.wav', modified_recon.squeeze(0).cpu(), 44100)

    print("Saved original and modified reconstructions")


def example_batch_processing():
    """Example: Process multiple audio files in a batch."""

    checkpoint_path = './pretrained_weight/ear_vae_44k.pyt'
    config_path = './config/model_config.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = Encoder(checkpoint_path, config_path).to(device).eval()
    decoder = Decoder(checkpoint_path, config_path).to(device).eval()

    # Load multiple audio files (assuming same length for batching)
    audio_paths = ['./data/audio1.wav', './data/audio2.wav']
    audio_batch = []

    for path in audio_paths:
        audio, sr = torchaudio.load(path)
        if sr != 44100:
            audio = torchaudio.transforms.Resample(sr, 44100)(audio)
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)
        audio_batch.append(audio)

    # Stack into batch
    audio_batch = torch.stack(audio_batch).to(device)  # [B, 2, T]

    with torch.no_grad():
        # Encode batch
        latents = encoder(audio_batch)
        print(f"Batch latents shape: {latents.shape}")

        # Decode batch
        reconstructed = decoder(latents)
        print(f"Batch reconstructed shape: {reconstructed.shape}")


if __name__ == '__main__':
    print("=== Example 1: Full Reconstruction ===")
    example_full_reconstruction()

    print("\n=== Example 2: Encode Only ===")
    example_encode_only()

    print("\n=== Example 3: Decode Only ===")
    example_decode_only()

    print("\n=== Example 4: Latent Manipulation ===")
    example_latent_manipulation()

    print("\n=== Example 5: Batch Processing ===")
    # example_batch_processing()  # Uncomment if you have multiple audio files
