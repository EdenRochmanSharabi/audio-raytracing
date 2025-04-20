import math
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
import os


class SoundSource:
    """
    Represents a sound source in the audio ray tracing environment.
    
    A sound source emits audio that propagates through the environment
    as rays. It has a position and properties related to the sound it produces.
    """
    
    def __init__(self, position: Tuple[float, float], audio_file: str = None, 
                 volume: float = 1.0, name: str = "Source"):
        """
        Initialize a sound source.
        
        Args:
            position: (x, y) coordinates in the environment
            audio_file: Path to audio file to play (if None, a tone is generated)
            volume: Base volume of the sound (0.0 to 1.0)
            name: Name identifier for the source
        """
        self.position = position
        self.audio_file = audio_file
        self.volume = volume
        self.name = name
        
        # Sound properties
        self.frequency = 440  # Default frequency in Hz (A4) if generating a tone
        self.active = True  # Whether the source is currently emitting sound
        
        # Directionality properties (omnidirectional by default)
        self.directional = False
        self.direction = 0.0  # Direction in degrees
        self.direction_rad = 0.0
        self.beam_width = 180.0  # Width of directional beam in degrees
        self.beam_width_rad = math.radians(self.beam_width)
        
        # Audio data (loaded lazily)
        self._audio_data = None
        self._sample_rate = None
    
    def load_audio(self) -> bool:
        """
        Load audio data from the specified file.
        
        Returns:
            True if audio was loaded successfully, False otherwise
        """
        if not self.audio_file or not os.path.exists(self.audio_file):
            return False
            
        try:
            # Try to use pydub for audio loading
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(self.audio_file)
            self._sample_rate = audio.frame_rate
            
            # Convert to numpy array (normalized to -1.0 to 1.0)
            samples = np.array(audio.get_array_of_samples())
            
            # Convert to mono if stereo
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            
            # Normalize
            self._audio_data = samples / 32768.0  # Assuming 16-bit audio
            return True
            
        except ImportError:
            # Fallback to scipy if pydub is not available
            try:
                from scipy.io import wavfile
                
                sample_rate, data = wavfile.read(self.audio_file)
                self._sample_rate = sample_rate
                
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                
                # Normalize
                if data.dtype == np.int16:
                    self._audio_data = data / 32768.0
                elif data.dtype == np.int32:
                    self._audio_data = data / 2147483648.0
                elif data.dtype == np.uint8:
                    self._audio_data = (data.astype(np.float32) - 128) / 128.0
                else:
                    self._audio_data = data  # Assume already normalized
                
                return True
                
            except Exception as e:
                print(f"Error loading audio file: {e}")
                return False
        
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def generate_tone(self, duration: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a simple sine wave tone at the source's frequency.
        
        Args:
            duration: Duration of the tone in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array of audio samples
        """
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * self.frequency * t)
        
        # Apply a simple envelope to avoid clicks
        attack = int(0.01 * sample_rate)
        release = int(0.01 * sample_rate)
        
        envelope = np.ones_like(tone)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        
        return tone * envelope
    
    def get_audio_data(self) -> Tuple[np.ndarray, int]:
        """
        Get the audio data for this source.
        
        If an audio file is specified and loaded, returns that data.
        Otherwise, generates a tone at the specified frequency.
        
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        if self._audio_data is None and self.audio_file:
            self.load_audio()
        
        if self._audio_data is not None:
            return self._audio_data, self._sample_rate
        else:
            # Generate a tone if no audio file or loading failed
            sample_rate = 44100
            tone = self.generate_tone(1.0, sample_rate)
            return tone, sample_rate
    
    def set_directional(self, directional: bool, direction: float = 0.0, beam_width: float = 90.0) -> None:
        """
        Configure the directionality of the sound source.
        
        Args:
            directional: Whether the source is directional (True) or omnidirectional (False)
            direction: Direction angle in degrees
            beam_width: Width of the directional beam in degrees
        """
        self.directional = directional
        self.direction = direction
        self.direction_rad = math.radians(direction)
        self.beam_width = beam_width
        self.beam_width_rad = math.radians(beam_width)
    
    def set_position(self, position: Tuple[float, float]) -> None:
        """
        Move the sound source to a new position.
        
        Args:
            position: New (x, y) coordinates
        """
        self.position = position
    
    def directional_factor(self, angle: float) -> float:
        """
        Calculate the directional intensity factor for a given angle.
        
        For omnidirectional sources, this is always 1.0.
        For directional sources, this follows a cosine pattern with maximum
        intensity in the direction of the beam and decreasing to 0 at the edges.
        
        Args:
            angle: Angle in radians
            
        Returns:
            Factor from 0.0 to 1.0
        """
        if not self.directional:
            return 1.0
        
        # Calculate angle difference
        angle_diff = abs(angle - self.direction_rad)
        
        # Normalize to [0, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        angle_diff = abs(angle_diff)
        
        # Check if outside beam width
        half_beam_width = self.beam_width_rad / 2
        if angle_diff > half_beam_width:
            return 0.0
        
        # Calculate cosine falloff within beam
        # Maps [0, beam_width_rad/2] to [1.0, 0.0]
        # Using cos²(x * π/2 / half_width) for a smoother falloff
        normalized_diff = angle_diff / half_beam_width
        factor = math.cos(normalized_diff * math.pi / 2)
        factor = factor * factor  # Square for a more realistic falloff
        
        return factor
    
    def get_ray_directions(self, ray_count: int) -> List[Tuple[float, float]]:
        """
        Get ray direction vectors for ray casting.
        
        For omnidirectional sources, rays are evenly distributed in all directions.
        For directional sources, rays are concentrated within the beam width.
        
        Args:
            ray_count: Number of rays to generate
            
        Returns:
            List of (dx, dy) normalized direction vectors
        """
        ray_directions = []
        
        if self.directional:
            # For directional sources, distribute rays within beam_width
            half_width = self.beam_width_rad / 2
            start_angle = self.direction_rad - half_width
            end_angle = self.direction_rad + half_width
            
            for i in range(ray_count):
                # Distribute angles, with more concentration toward center
                # using a cosine distribution
                t = i / (ray_count - 1) if ray_count > 1 else 0.5
                bias = 0.5 * (1 - math.cos(t * math.pi))
                angle = start_angle + bias * (end_angle - start_angle)
                
                dx = math.cos(angle)
                dy = math.sin(angle)
                ray_directions.append((dx, dy))
        else:
            # For omnidirectional sources, distribute rays evenly in all directions
            angle_step = 2 * math.pi / ray_count
            
            for i in range(ray_count):
                angle = i * angle_step
                dx = math.cos(angle)
                dy = math.sin(angle)
                ray_directions.append((dx, dy))
        
        return ray_directions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert sound source to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the sound source
        """
        return {
            "position": self.position,
            "audio_file": self.audio_file,
            "volume": self.volume,
            "name": self.name,
            "frequency": self.frequency,
            "active": self.active,
            "directional": self.directional,
            "direction": self.direction,
            "beam_width": self.beam_width
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SoundSource':
        """
        Create a sound source from dictionary data.
        
        Args:
            data: Dictionary with sound source properties
            
        Returns:
            SoundSource instance
        """
        source = cls(
            position=tuple(data["position"]),
            audio_file=data.get("audio_file"),
            volume=data.get("volume", 1.0),
            name=data.get("name", "Source")
        )
        
        # Set optional properties if present
        if "frequency" in data:
            source.frequency = data["frequency"]
        if "active" in data:
            source.active = data["active"]
            
        # Set directionality if applicable
        if data.get("directional", False):
            source.set_directional(
                directional=True,
                direction=data.get("direction", 0.0),
                beam_width=data.get("beam_width", 90.0)
            )
            
        return source 