import pygame
import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Callable


class AudioPlayer:
    """
    Handles audio playback for the ray tracing simulation.
    
    This class manages the playback of processed audio through pygame's
    audio system, providing control over volume, looping, and playback state.
    """
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2, buffer_size: int = 1024):
        """
        Initialize the audio player.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            buffer_size: Size of the audio buffer in samples
        """
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, channels=channels, buffer=buffer_size)
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        
        # Playback properties
        self.volume = 1.0
        self.is_playing = False
        self.is_looping = False
        
        # Active sounds
        self.active_sounds = {}
        self.sound_id_counter = 0
        
        # Callbacks
        self.on_playback_complete = None
    
    def play_sound(self, audio_data: np.ndarray, volume: float = 1.0, 
                  loop: bool = False) -> int:
        """
        Play a sound from numpy audio data.
        
        Args:
            audio_data: Audio data as a numpy array (float32, -1.0 to 1.0)
            volume: Volume level (0.0 to 1.0)
            loop: Whether to loop the sound
            
        Returns:
            Sound ID that can be used to stop the sound later
        """
        # Ensure audio data is 2D for stereo (channels in second dimension)
        if len(audio_data.shape) == 1:
            # Convert mono to stereo by duplicating the channel
            if self.channels == 2:
                audio_data = np.column_stack((audio_data, audio_data))
        elif audio_data.shape[1] == 1 and self.channels == 2:
            # Convert mono to stereo by duplicating the channel
            audio_data = np.column_stack((audio_data, audio_data))
        elif audio_data.shape[1] == 2 and self.channels == 1:
            # Convert stereo to mono by averaging channels
            audio_data = np.mean(audio_data, axis=1)
        
        # Scale to int16 range
        audio_int16 = (audio_data * 32767 * volume * self.volume).astype(np.int16)
        
        # Create pygame Sound object
        sound = pygame.mixer.Sound(audio_int16)
        
        # Generate unique ID for this sound
        sound_id = self.sound_id_counter
        self.sound_id_counter += 1
        
        # Store in active sounds
        self.active_sounds[sound_id] = {
            "sound": sound,
            "is_looping": loop,
            "volume": volume,
            "channel": None
        }
        
        # Play the sound
        channel = sound.play(-1 if loop else 0)
        self.active_sounds[sound_id]["channel"] = channel
        
        self.is_playing = True
        return sound_id
    
    def stop_sound(self, sound_id: int) -> bool:
        """
        Stop a specific sound.
        
        Args:
            sound_id: ID of the sound to stop
            
        Returns:
            True if sound was stopped, False if not found
        """
        if sound_id in self.active_sounds:
            sound_info = self.active_sounds[sound_id]
            sound = sound_info["sound"]
            sound.stop()
            del self.active_sounds[sound_id]
            return True
        return False
    
    def stop_all_sounds(self) -> None:
        """Stop all currently playing sounds."""
        pygame.mixer.stop()
        self.active_sounds.clear()
        self.is_playing = False
    
    def pause_all_sounds(self) -> None:
        """Pause all currently playing sounds."""
        pygame.mixer.pause()
        self.is_playing = False
    
    def resume_all_sounds(self) -> None:
        """Resume all paused sounds."""
        pygame.mixer.unpause()
        if len(self.active_sounds) > 0:
            self.is_playing = True
    
    def set_volume(self, volume: float) -> None:
        """
        Set the master volume for all sounds.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))
        
        # Apply to all active sounds
        for sound_id, sound_info in self.active_sounds.items():
            sound = sound_info["sound"]
            sound.set_volume(sound_info["volume"] * self.volume)
    
    def set_sound_volume(self, sound_id: int, volume: float) -> bool:
        """
        Set the volume for a specific sound.
        
        Args:
            sound_id: ID of the sound
            volume: Volume level (0.0 to 1.0)
            
        Returns:
            True if volume was set, False if sound not found
        """
        if sound_id in self.active_sounds:
            volume = max(0.0, min(1.0, volume))
            sound_info = self.active_sounds[sound_id]
            sound = sound_info["sound"]
            sound_info["volume"] = volume
            sound.set_volume(volume * self.volume)
            return True
        return False
    
    def update(self) -> None:
        """
        Update the player state (check for completed sounds, etc.).
        Should be called regularly from the main loop.
        """
        # Check for completed sounds
        completed_sounds = []
        
        for sound_id, sound_info in self.active_sounds.items():
            channel = sound_info["channel"]
            
            # Check if channel is done playing and not looping
            if channel is not None and not channel.get_busy() and not sound_info["is_looping"]:
                completed_sounds.append(sound_id)
        
        # Remove completed sounds
        for sound_id in completed_sounds:
            del self.active_sounds[sound_id]
            
            # Call completion callback if set
            if self.on_playback_complete:
                self.on_playback_complete(sound_id)
        
        # Update playing state
        self.is_playing = len(self.active_sounds) > 0 and pygame.mixer.get_busy()
    
    def get_busy(self) -> bool:
        """
        Check if any sounds are currently playing.
        
        Returns:
            True if sounds are playing, False otherwise
        """
        return pygame.mixer.get_busy()
    
    def set_playback_complete_callback(self, callback: Callable[[int], None]) -> None:
        """
        Set a callback function to be called when a sound completes playback.
        
        Args:
            callback: Function that takes a sound_id parameter
        """
        self.on_playback_complete = callback
    
    def play_audio_file(self, file_path: str, volume: float = 1.0, 
                       loop: bool = False) -> int:
        """
        Play audio from a file.
        
        Args:
            file_path: Path to audio file
            volume: Volume level (0.0 to 1.0)
            loop: Whether to loop the sound
            
        Returns:
            Sound ID or -1 if file couldn't be loaded
        """
        try:
            # Create sound from file
            sound = pygame.mixer.Sound(file_path)
            
            # Generate unique ID
            sound_id = self.sound_id_counter
            self.sound_id_counter += 1
            
            # Store in active sounds
            self.active_sounds[sound_id] = {
                "sound": sound,
                "is_looping": loop,
                "volume": volume,
                "channel": None
            }
            
            # Set volume
            sound.set_volume(volume * self.volume)
            
            # Play the sound
            channel = sound.play(-1 if loop else 0)
            self.active_sounds[sound_id]["channel"] = channel
            
            self.is_playing = True
            return sound_id
            
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return -1
    
    def fade_out(self, sound_id: int, fade_ms: int = 1000) -> bool:
        """
        Fade out a specific sound.
        
        Args:
            sound_id: ID of the sound to fade
            fade_ms: Fade out duration in milliseconds
            
        Returns:
            True if fade started, False if sound not found
        """
        if sound_id in self.active_sounds:
            sound_info = self.active_sounds[sound_id]
            channel = sound_info["channel"]
            
            if channel:
                channel.fadeout(fade_ms)
                return True
        return False
    
    def fade_out_all(self, fade_ms: int = 1000) -> None:
        """
        Fade out all currently playing sounds.
        
        Args:
            fade_ms: Fade out duration in milliseconds
        """
        pygame.mixer.fadeout(fade_ms)
    
    def set_position(self, sound_id: int, position_sec: float) -> bool:
        """
        Set playback position for a sound.
        
        Args:
            sound_id: ID of the sound
            position_sec: Position in seconds
            
        Returns:
            True if position was set, False if not supported or sound not found
        """
        # Note: pygame.mixer.Sound doesn't support seeking directly
        # This is a placeholder for future implementation
        return False 