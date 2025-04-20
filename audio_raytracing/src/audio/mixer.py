import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from scipy.io import wavfile
from pydub import AudioSegment
import pygame
import os
import math


class SpatialAudioMixer:
    """
    Implements a spatial audio mixer that converts ray tracing data to audio output.
    
    This class receives the ray-traced sound paths and creates a spatial audio
    representation based on the timing, intensity, and direction of sound rays.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the spatial audio mixer.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, channels=2)
        
        # Audio cache for loaded sound files
        self.audio_cache = {}
        
        # Output buffer for final mixed audio
        self.left_channel = np.zeros(0)
        self.right_channel = np.zeros(0)
        
        # Mixer state
        self.is_playing = False
        self.max_buffer_size = 5 * sample_rate  # 5 seconds buffer
        
        # Spatialization parameters
        self.head_width = 0.2  # meters
        self.speed_of_sound = 343.0  # m/s
        self.distance_factor = 0.5  # Scaling factor for distance calculations
        self.intensity_factor = 10000.0  # Global volume control (increased dramatically from 100.0)
        
        # Pan law (determines how volume is distributed between channels)
        self.pan_law = "constant_power"  # "linear", "constant_power", or "square_root"
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio data from a file and cache it.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        if file_path in self.audio_cache:
            return self.audio_cache[file_path]
        
        try:
            # For WAV files, use scipy for better performance
            if file_path.lower().endswith('.wav'):
                sample_rate, data = wavfile.read(file_path)
                
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = data.mean(axis=1).astype(data.dtype)
                
                # Normalize to float in range [-1, 1]
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128) / 128.0
                
                self.audio_cache[file_path] = (data, sample_rate)
                return data, sample_rate
            
            # For other formats, use pydub
            else:
                audio = AudioSegment.from_file(file_path)
                sample_rate = audio.frame_rate
                
                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples()).astype(np.float32)
                
                # Convert to mono if stereo
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                
                # Normalize
                samples = samples / 32768.0  # Assuming 16-bit audio
                
                self.audio_cache[file_path] = (samples, sample_rate)
                return samples, sample_rate
                
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            # Return empty audio and original sample rate as fallback
            return np.zeros(0), self.sample_rate
    
    def clear_cache(self) -> None:
        """Clear the audio cache to free memory."""
        self.audio_cache = {}
    
    def calculate_spatial_delay(self, angle: float, distance: float) -> Tuple[float, float]:
        """
        Calculate the delay between left and right ears based on sound direction.
        
        Args:
            angle: Angle from listener to sound source in radians
            distance: Distance from listener to sound source in meters
            
        Returns:
            Tuple of (left_delay, right_delay) in seconds
        """
        # Base delay from distance
        base_delay = distance / self.speed_of_sound
        
        # Calculate additional delay from interaural time difference (ITD)
        # Maximum delay when sound comes directly from one side
        max_itd = self.head_width / self.speed_of_sound
        
        # Scale by angle (0 when in front/back, maximum when from side)
        # Simplification: itd = max_itd * sin(angle)
        scaled_itd = max_itd * math.sin(angle)
        
        # Apply to left/right channels
        # Sound from the right (positive angle) delays left ear
        if angle > 0:
            left_delay = base_delay + abs(scaled_itd)
            right_delay = base_delay
        # Sound from the left (negative angle) delays right ear
        else:
            left_delay = base_delay
            right_delay = base_delay + abs(scaled_itd)
        
        return left_delay, right_delay
    
    def calculate_panning(self, angle: float) -> Tuple[float, float]:
        """
        Calculate stereo panning based on sound direction.
        
        Args:
            angle: Angle from listener to sound source in radians
            
        Returns:
            Tuple of (left_gain, right_gain) from 0.0 to 1.0
        """
        # Normalize angle to range [-pi/2, pi/2]
        norm_angle = max(-math.pi/2, min(math.pi/2, angle))
        
        # Convert to a pan value from -1 (full left) to 1 (full right)
        pan = norm_angle / (math.pi/2)
        
        # Apply pan law
        if self.pan_law == "linear":
            # Linear panning (simple but can reduce overall volume at center)
            left_gain = max(0, 1 - pan)
            right_gain = max(0, 1 + pan)
        
        elif self.pan_law == "square_root":
            # Square root law (compromise between constant power and linear)
            left_gain = math.sqrt(max(0, 1 - pan))
            right_gain = math.sqrt(max(0, 1 + pan))
        
        else:  # "constant_power" (default)
            # Constant power law (maintains consistent perceived volume)
            # Using the formula: L = cos(θ), R = sin(θ) where θ goes from 0 to π/2
            pan_angle = (pan + 1) * math.pi / 4  # Convert -1..1 to 0..π/2
            left_gain = math.cos(pan_angle)
            right_gain = math.sin(pan_angle)
        
        return left_gain, right_gain
    
    def calculate_intensity(self, energy: float, distance: float) -> float:
        """
        Calculate sound intensity based on energy and distance.
        
        Args:
            energy: Energy of the sound (0.0 to 1.0)
            distance: Distance from source to listener
            
        Returns:
            Intensity factor (uncapped)
        """
        # Apply a gentler distance falloff for better gameplay experience
        # Use a minimum distance to avoid extreme volumes up close
        min_distance = 0.1
        distance = max(min_distance, distance)
        
        # Calculate intensity - using linear falloff instead of squared
        intensity = energy * (1.0 / distance)
        
        # Apply global intensity factor
        intensity *= self.intensity_factor
        
        # No maximum cap on intensity to allow louder sounds at distance
        return max(0.0, intensity)
    
    def create_spatial_audio(self, sound_paths: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial audio output based on sound paths from ray tracing.
        
        Args:
            sound_paths: List of sound path data from ray tracing
            
        Returns:
            Tuple of (left_channel, right_channel) audio arrays
        """
        # Print diagnostic info
        print(f"Creating spatial audio from {len(sound_paths)} sound paths")
        
        # Initialize empty output channels
        max_length = 0
        sound_segments = []
        
        # Process each sound path
        for path in sound_paths:
            try:
                # Extract path data
                source_id = path.get("source_id", 0)
                audio_file = path.get("audio_file")
                frequency = path.get("frequency", 440)  # Default to 440Hz if not specified
                intensity = path.get("intensity", 1.0)
                angle = path.get("angle", 0.0)  # Angle in radians
                distance = path.get("distance", 1.0)  # Distance in meters
                delay = path.get("delay", 0.0)  # Delay in seconds
                energy = path.get("energy", 1.0)
                
                # Sanity check: must have reasonable values
                if intensity <= 0 or energy <= 0:
                    continue
                
                # Get the audio data
                audio_data = None
                sample_rate = self.sample_rate
                
                if audio_file and os.path.exists(audio_file):
                    # Load audio from file
                    audio_data, sample_rate = self.load_audio(audio_file)
                
                if audio_data is None or len(audio_data) == 0:
                    # Generate a tone if no audio file or loading failed
                    duration = 0.5  # 500ms tone
                    t = np.linspace(0, duration, int(self.sample_rate * duration), False)
                    audio_data = np.sin(2 * np.pi * frequency * t)
                    
                    # Apply envelope to avoid clicks
                    attack = int(0.01 * self.sample_rate)
                    release = int(0.01 * self.sample_rate)
                    
                    if len(audio_data) > attack + release:
                        envelope = np.ones_like(audio_data)
                        envelope[:attack] = np.linspace(0, 1, attack)
                        envelope[-release:] = np.linspace(1, 0, release)
                        audio_data = audio_data * envelope
                
                # Resample if needed
                if sample_rate != self.sample_rate:
                    # Simple resampling
                    duration = len(audio_data) / sample_rate
                    new_length = int(duration * self.sample_rate)
                    indices = np.linspace(0, len(audio_data) - 1, new_length)
                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
                
                # Calculate spatial parameters
                final_intensity = self.calculate_intensity(energy, distance)
                left_gain, right_gain = self.calculate_panning(angle)
                left_delay_sec, right_delay_sec = self.calculate_spatial_delay(angle, distance)
                
                # Add base delay from propagation
                left_delay_sec += delay
                right_delay_sec += delay
                
                # Convert delays to samples
                left_delay = int(left_delay_sec * self.sample_rate)
                right_delay = int(right_delay_sec * self.sample_rate)
                
                # Create stereo audio segment
                left = np.zeros(left_delay + len(audio_data))
                right = np.zeros(right_delay + len(audio_data))
                
                # Apply audio to channels with appropriate delays and gains
                left[left_delay:left_delay + len(audio_data)] = audio_data * left_gain * final_intensity
                right[right_delay:right_delay + len(audio_data)] = audio_data * right_gain * final_intensity
                
                # Ensure both channels are the same length
                max_len = max(len(left), len(right))
                if len(left) < max_len:
                    left = np.pad(left, (0, max_len - len(left)))
                if len(right) < max_len:
                    right = np.pad(right, (0, max_len - len(right)))
                
                # Add to segments
                sound_segments.append((left, right))
                max_length = max(max_length, max_len)
                
            except Exception as e:
                print(f"Error processing sound path: {e}")
                continue
        
        # If no valid segments, return empty arrays
        if not sound_segments or max_length == 0:
            print("Warning: No valid sound segments produced")
            return np.zeros(0), np.zeros(0)
        
        try:
            # Ensure all segments are the same length
            for i in range(len(sound_segments)):
                left, right = sound_segments[i]
                if len(left) < max_length:
                    sound_segments[i] = (
                        np.pad(left, (0, max_length - len(left))),
                        np.pad(right, (0, max_length - len(right)))
                    )
            
            # Mix all segments together
            left_channel = np.zeros(max_length)
            right_channel = np.zeros(max_length)
            
            for left, right in sound_segments:
                left_channel += left
                right_channel += right
            
            # Normalize if needed to prevent clipping
            max_amplitude = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
            if max_amplitude > 1.0:
                left_channel = left_channel / max_amplitude
                right_channel = right_channel / max_amplitude
                
            # Set class properties
            self.left_channel = left_channel
            self.right_channel = right_channel
            
            print(f"Successfully created spatial audio: {len(left_channel)} samples")
            return left_channel, right_channel
            
        except Exception as e:
            print(f"Error mixing audio: {e}")
            return np.zeros(0), np.zeros(0)
    
    def play_audio(self, left_channel: np.ndarray, right_channel: np.ndarray) -> None:
        """
        Play the processed audio through pygame mixer.
        
        Args:
            left_channel: Left channel audio data
            right_channel: Right channel audio data
        """
        # Combine channels into stereo
        stereo_audio = np.column_stack((left_channel, right_channel))
        
        # Convert to int16 format for pygame
        stereo_audio = (stereo_audio * 32767).astype(np.int16)
        
        # Create a pygame Sound object
        sound = pygame.mixer.Sound(stereo_audio)
        
        # Play the sound
        sound.play()
        self.is_playing = True
    
    def process_sound_paths(self, sound_paths: List[Dict[str, Any]]) -> None:
        """
        Process sound paths from ray tracing and play the resulting audio.
        
        Args:
            sound_paths: List of sound path data from ray tracing
        """
        # Print diagnostic info about the paths
        print(f"Sound paths: {len(sound_paths)}")
        if sound_paths:
            print(f"First sound path: {sound_paths[0]}")
        
        # Always process audio even if we've had sound paths before
        # This ensures continuous audio as the player moves
        
        # Create spatial audio
        left_channel, right_channel = self.create_spatial_audio(sound_paths)
        
        # Store for later access
        self.left_channel = left_channel
        self.right_channel = right_channel
        
        # Play audio if we have valid channels
        if len(left_channel) > 0 and len(right_channel) > 0:
            # Always play the new audio
            self.play_audio(left_channel, right_channel)
        elif not sound_paths:
            # If no sound paths and no audio is playing, ensure silence
            self.stop_audio()
    
    def stop_audio(self) -> None:
        """Stop all currently playing audio."""
        pygame.mixer.stop()
        self.is_playing = False
    
    def set_parameters(self, head_width: Optional[float] = None, 
                      speed_of_sound: Optional[float] = None,
                      distance_factor: Optional[float] = None,
                      intensity_factor: Optional[float] = None,
                      pan_law: Optional[str] = None) -> None:
        """
        Set the parameters for audio spatialization.
        
        Args:
            head_width: Width of head in meters for ITD calculation
            speed_of_sound: Speed of sound in meters/second
            distance_factor: Scaling factor for distance calculations
            intensity_factor: Global volume control
            pan_law: "linear", "constant_power", or "square_root"
        """
        if head_width is not None:
            self.head_width = max(0.1, head_width)
        
        if speed_of_sound is not None:
            self.speed_of_sound = max(1.0, speed_of_sound)
        
        if distance_factor is not None:
            self.distance_factor = max(0.1, distance_factor)
        
        if intensity_factor is not None:
            self.intensity_factor = max(0.0, min(5.0, intensity_factor))
        
        if pan_law is not None and pan_law in ["linear", "constant_power", "square_root"]:
            self.pan_law = pan_law
    
    def save_output(self, file_path: str) -> bool:
        """
        Save the most recently generated audio to a file.
        
        Args:
            file_path: Path to save the audio file
            
        Returns:
            True if save was successful, False otherwise
        """
        if len(self.left_channel) == 0 or len(self.right_channel) == 0:
            print("No audio to save")
            return False
        
        try:
            # Combine channels into stereo
            stereo_audio = np.column_stack((self.left_channel, self.right_channel))
            
            # Convert to int16 format
            stereo_audio = (stereo_audio * 32767).astype(np.int16)
            
            # Save to file
            wavfile.write(file_path, self.sample_rate, stereo_audio)
            return True
            
        except Exception as e:
            print(f"Error saving audio to {file_path}: {e}")
            return False 