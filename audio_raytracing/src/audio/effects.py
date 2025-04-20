import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math


class AudioEffects:
    """
    Implements audio effects processing for the ray tracing simulation.
    
    This class provides various audio effects such as delay, echo, and
    reverberation that can be applied to audio data based on the physical
    properties of the environment and sound path.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the audio effects processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Effect parameters with default values
        self.effect_params = {
            "delay": {
                "enabled": True,
                "max_delay_time": 1.0,  # Maximum delay time in seconds
                "min_volume": 0.1       # Minimum volume for delayed sounds
            },
            "echo": {
                "enabled": True,
                "feedback": 0.5,        # Echo feedback amount (0.0 to 1.0)
                "delay_time": 0.3,      # Echo delay time in seconds
                "decay": 0.7            # Echo decay factor (0.0 to 1.0)
            },
            "reverb": {
                "enabled": False,
                "room_size": 0.5,       # Virtual room size (0.0 to 1.0)
                "damping": 0.5,         # High frequency damping (0.0 to 1.0)
                "wet_level": 0.33,      # Wet (processed) signal level
                "dry_level": 0.4,       # Dry (original) signal level
                "width": 1.0            # Stereo width (0.0 to 1.0)
            }
        }
    
    def apply_delay(self, audio_data: np.ndarray, delay_time: float, 
                   feedback: float = 0.0) -> np.ndarray:
        """
        Apply a simple delay effect to audio data.
        
        Args:
            audio_data: Input audio samples
            delay_time: Delay time in seconds
            feedback: Amount of feedback (0.0 to 1.0)
            
        Returns:
            Processed audio data
        """
        if not self.effect_params["delay"]["enabled"] or delay_time <= 0:
            return audio_data
        
        # Limit delay time
        max_delay = self.effect_params["delay"]["max_delay_time"]
        delay_time = min(delay_time, max_delay)
        
        # Convert delay time to samples
        delay_samples = int(delay_time * self.sample_rate)
        
        # Create output buffer
        if feedback > 0:
            # For feedback, we need a longer buffer
            max_feedback_repeats = int(math.log(0.01) / math.log(feedback)) + 1
            output_length = len(audio_data) + delay_samples * max_feedback_repeats
            output = np.zeros(output_length)
        else:
            # For simple delay, we just need input length + delay
            output_length = len(audio_data) + delay_samples
            output = np.zeros(output_length)
        
        # Copy original signal
        output[:len(audio_data)] = audio_data
        
        if feedback > 0:
            # Apply feedback delay
            attenuation = 1.0
            for i in range(1, max_feedback_repeats + 1):
                if attenuation < 0.01:  # Stop when too quiet
                    break
                    
                start_idx = i * delay_samples
                end_idx = start_idx + len(audio_data)
                
                if end_idx <= len(output):
                    attenuation *= feedback
                    output[start_idx:end_idx] += audio_data * attenuation
        
        return output
    
    def apply_echo(self, audio_data: np.ndarray, num_echoes: int = 3, 
                  initial_delay: float = 0.3, decay: float = 0.7,
                  spacing: float = 0.1) -> np.ndarray:
        """
        Apply echo effect with multiple decaying echoes.
        
        Args:
            audio_data: Input audio samples
            num_echoes: Number of echo repetitions
            initial_delay: Time of first echo in seconds
            decay: Volume decay factor for each echo
            spacing: Time between echo repetitions in seconds
            
        Returns:
            Processed audio data
        """
        if not self.effect_params["echo"]["enabled"] or num_echoes <= 0:
            return audio_data
        
        # Use effect parameters if not specified
        if initial_delay is None:
            initial_delay = self.effect_params["echo"]["delay_time"]
        
        if decay is None:
            decay = self.effect_params["echo"]["decay"]
        
        # Calculate total output length
        total_delay_time = initial_delay + spacing * (num_echoes - 1)
        total_delay_samples = int(total_delay_time * self.sample_rate)
        output_length = len(audio_data) + total_delay_samples
        
        # Create output buffer
        output = np.zeros(output_length)
        
        # Copy original signal
        output[:len(audio_data)] = audio_data.copy()
        
        # Apply echoes
        current_delay = initial_delay
        current_amplitude = decay  # Start with decay for first echo
        
        for i in range(num_echoes):
            # Calculate delay in samples
            delay_samples = int(current_delay * self.sample_rate)
            
            # Apply echo
            start_idx = delay_samples
            end_idx = start_idx + len(audio_data)
            
            if end_idx <= len(output):
                output[start_idx:end_idx] += audio_data * current_amplitude
                current_amplitude *= decay
            
            # Increment delay for next echo
            current_delay += spacing
        
        return output
    
    def apply_reverb(self, audio_data: np.ndarray, room_size: float = None, 
                    damping: float = None) -> np.ndarray:
        """
        Apply a simple reverb effect to audio data.
        
        This is a simplified implementation that approximates reverb using
        multiple echoes with randomized parameters.
        
        Args:
            audio_data: Input audio samples
            room_size: Virtual room size (0.0 to 1.0)
            damping: High frequency damping (0.0 to 1.0)
            
        Returns:
            Processed audio data
        """
        if not self.effect_params["reverb"]["enabled"]:
            return audio_data
        
        # Use effect parameters if not specified
        if room_size is None:
            room_size = self.effect_params["reverb"]["room_size"]
        
        if damping is None:
            damping = self.effect_params["reverb"]["damping"]
        
        # Validate parameters
        room_size = max(0.1, min(1.0, room_size))
        damping = max(0.1, min(0.99, damping))
        
        # Calculate reverb parameters from room size
        reverb_time = 0.2 + room_size * 2.0  # 0.2 to 2.2 seconds
        num_reflections = int(5 + room_size * 15)  # 5 to 20 reflections
        
        # Create early reflections
        reflections = []
        np.random.seed(42)  # Fixed seed for reproducibility
        
        # Scale for the randomization
        time_spread = 0.05 + room_size * 0.2
        
        for i in range(num_reflections):
            # Random delay between 0.01 and 0.5 seconds scaled by room size
            delay = 0.01 + np.random.random() * reverb_time * 0.5
            
            # Random attenuation with higher damping for longer delays
            # (simulating air and wall absorption)
            atten = (1.0 - delay / reverb_time) * (1.0 - damping * 0.5)
            atten = max(0.01, atten)
            
            reflections.append((delay, atten))
        
        # Sort by delay time
        reflections.sort(key=lambda x: x[0])
        
        # Calculate total output length
        max_delay = reflections[-1][0]
        total_delay_samples = int(max_delay * self.sample_rate)
        output_length = len(audio_data) + total_delay_samples
        
        # Create output buffer
        output = np.zeros(output_length)
        
        # Apply dry signal
        dry_level = self.effect_params["reverb"]["dry_level"]
        output[:len(audio_data)] = audio_data * dry_level
        
        # Apply wet signal (reflections)
        wet_level = self.effect_params["reverb"]["wet_level"]
        
        for delay, attenuation in reflections:
            # Convert delay to samples
            delay_samples = int(delay * self.sample_rate)
            
            # Create slightly filtered version for this reflection
            if damping > 0.5:
                # Crude low-pass filtering by simple averaging (more damping = more filtering)
                filter_width = int((damping - 0.5) * 10) + 1
                reflection_data = np.convolve(audio_data, 
                                           np.ones(filter_width)/filter_width, 
                                           mode='same')
            else:
                reflection_data = audio_data.copy()
            
            # Apply reflection
            start_idx = delay_samples
            end_idx = start_idx + len(audio_data)
            
            if end_idx <= len(output):
                output[start_idx:end_idx] += reflection_data * attenuation * wet_level
        
        return output
    
    def apply_all_effects(self, audio_data: np.ndarray, 
                         parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply all enabled effects to audio data based on provided parameters.
        
        Args:
            audio_data: Input audio samples
            parameters: Dictionary of effect parameters
            
        Returns:
            Processed audio data
        """
        result = audio_data.copy()
        
        # Extract parameters
        delay_params = parameters.get("delay", {})
        echo_params = parameters.get("echo", {})
        reverb_params = parameters.get("reverb", {})
        
        # Apply delay if enabled
        if self.effect_params["delay"]["enabled"]:
            delay_time = delay_params.get("time", 0.0)
            feedback = delay_params.get("feedback", 0.0)
            
            if delay_time > 0:
                result = self.apply_delay(result, delay_time, feedback)
        
        # Apply echo if enabled
        if self.effect_params["echo"]["enabled"]:
            num_echoes = echo_params.get("num_echoes", 3)
            initial_delay = echo_params.get("initial_delay", 
                                         self.effect_params["echo"]["delay_time"])
            decay = echo_params.get("decay", self.effect_params["echo"]["decay"])
            spacing = echo_params.get("spacing", 0.1)
            
            if num_echoes > 0:
                result = self.apply_echo(result, num_echoes, initial_delay, decay, spacing)
        
        # Apply reverb if enabled
        if self.effect_params["reverb"]["enabled"]:
            room_size = reverb_params.get("room_size", self.effect_params["reverb"]["room_size"])
            damping = reverb_params.get("damping", self.effect_params["reverb"]["damping"])
            
            result = self.apply_reverb(result, room_size, damping)
        
        return result
    
    def configure_effects(self, effect_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure effect parameters.
        
        Args:
            effect_config: Dictionary of effect parameters
        """
        # Update delay parameters
        if "delay" in effect_config:
            for key, value in effect_config["delay"].items():
                if key in self.effect_params["delay"]:
                    self.effect_params["delay"][key] = value
        
        # Update echo parameters
        if "echo" in effect_config:
            for key, value in effect_config["echo"].items():
                if key in self.effect_params["echo"]:
                    self.effect_params["echo"][key] = value
        
        # Update reverb parameters
        if "reverb" in effect_config:
            for key, value in effect_config["reverb"].items():
                if key in self.effect_params["reverb"]:
                    self.effect_params["reverb"][key] = value
    
    def process_for_sound_path(self, audio_data: np.ndarray, 
                              sound_path: Dict[str, Any]) -> np.ndarray:
        """
        Process audio data for a specific sound path with appropriate effects.
        
        Args:
            audio_data: Input audio samples
            sound_path: Sound path data from ray tracing
            
        Returns:
            Processed audio data
        """
        # Extract relevant path data
        distance = sound_path.get("distance", 1.0)
        reflection_count = sound_path.get("reflection_count", 0)
        energy = sound_path.get("energy", 1.0)
        delay = sound_path.get("delay", 0.0)
        
        # Prepare effect parameters
        effect_params = {
            "delay": {
                "time": delay,
                "feedback": 0.0  # No feedback for initial delay
            },
            "echo": {
                "num_echoes": reflection_count,
                "initial_delay": delay + 0.05,  # Add a small offset after initial delay
                "decay": 0.7 * energy,  # Decay based on remaining energy
                "spacing": 0.05 + 0.02 * distance  # More spacing for longer distances
            },
            "reverb": {
                "room_size": min(1.0, distance / 10.0),  # Larger room size for longer distances
                "damping": 0.5 + (1.0 - energy) * 0.3  # More damping for lower energy
            }
        }
        
        # Apply all effects
        return self.apply_all_effects(audio_data, effect_params)
    
    def enable_effect(self, effect_name: str, enabled: bool = True) -> None:
        """
        Enable or disable a specific effect.
        
        Args:
            effect_name: Name of the effect ("delay", "echo", or "reverb")
            enabled: Whether the effect should be enabled
        """
        if effect_name in self.effect_params:
            self.effect_params[effect_name]["enabled"] = enabled 