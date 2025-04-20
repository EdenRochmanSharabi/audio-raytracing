#!/usr/bin/env python3
from pydub import AudioSegment
import os

def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Convert MP3 file to WAV format
    
    Args:
        mp3_path: Path to the MP3 file
        wav_path: Path where WAV file should be saved
    """
    try:
        # Load MP3 file
        print(f"Loading {mp3_path}...")
        sound = AudioSegment.from_mp3(mp3_path)
        
        # Export as WAV
        print(f"Converting to WAV format...")
        sound.export(wav_path, format="wav")
        
        print(f"Conversion complete. WAV file saved at {wav_path}")
        return True
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

if __name__ == "__main__":
    # Directory paths
    assets_dir = "assets"
    
    # Ensure the assets directory exists
    os.makedirs(assets_dir, exist_ok=True)
    
    # Input and output file paths
    mp3_path = os.path.join(assets_dir, "audio.mp3")
    wav_path = os.path.join(assets_dir, "audio.wav")
    
    # Check if MP3 file exists
    if not os.path.exists(mp3_path):
        print(f"Error: {mp3_path} does not exist.")
        print("Please place your audio.mp3 file in the assets directory.")
        exit(1)
    
    # Convert MP3 to WAV
    if convert_mp3_to_wav(mp3_path, wav_path):
        # Update the audio file path in main.py
        main_file = "main.py"
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Replace MP3 with WAV in the source definition
        updated_content = content.replace(
            'audio_file="assets/audio.mp3"',
            'audio_file="assets/audio.wav"'
        )
        
        with open(main_file, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated {main_file} to use audio.wav") 