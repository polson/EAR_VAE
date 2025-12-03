#!/usr/bin/env python3
"""
Test script to verify package imports work correctly.
"""

import sys
import os

# Add parent directory to path to simulate installed package
sys.path.insert(0, os.path.dirname(__file__))

try:
    print("Testing imports...")
    from ear_vae import Encoder, Decoder, EAR_VAE
    print("✓ Successfully imported: Encoder, Decoder, EAR_VAE")

    from ear_vae import OobleckEncoder, OobleckDecoder
    print("✓ Successfully imported: OobleckEncoder, OobleckDecoder")

    from ear_vae import ContinuousTransformer
    print("✓ Successfully imported: ContinuousTransformer")

    print("\n✓ All imports successful!")
    print("\nAvailable classes:")
    print(f"  - Encoder: {Encoder}")
    print(f"  - Decoder: {Decoder}")
    print(f"  - EAR_VAE: {EAR_VAE}")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
