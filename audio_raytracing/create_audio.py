#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile
import os

def generate_tone(freq=440, duration=1.0, sample_rate=44100):
    """Generate a sine wave at the given frequency"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * freq * t)
    
    # Apply envelope to avoid clicks
    envelope = np.ones_like(tone)
    attack = int(0.05 * sample_rate)
    release = int(0.05 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    return tone * envelope

def create_test_audio():
    """Create a test audio file with a clear, recognizable sound"""
    sample_rate = 44100
    
    # Generate a simple piano-like tone with harmonics
    duration = 1.0
    fundamental = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a note with harmonics (fundamental + overtones)
    audio = np.sin(2 * np.pi * fundamental * t) * 0.7  # fundamental
    audio += np.sin(2 * np.pi * fundamental * 2 * t) * 0.2  # first overtone
    audio += np.sin(2 * np.pi * fundamental * 3 * t) * 0.1  # second overtone
    
    # Apply envelope for natural decay
    envelope = np.exp(-t * 3)  # Exponential decay
    audio = audio * envelope
    
    # Normalize to use full dynamic range
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save to file
    os.makedirs('audio_raytracing/assets', exist_ok=True)  # Make sure directory exists
    wavfile.write('audio_raytracing/assets/audio.wav', sample_rate, audio_int16)
    print(f"Created audio file: audio_raytracing/assets/audio.wav")

if __name__ == "__main__":
    create_test_audio() 