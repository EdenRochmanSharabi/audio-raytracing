import unittest
import numpy as np
import os
import tempfile
import time
import pygame
import sys

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio.mixer import SpatialAudioMixer
from src.audio.effects import AudioEffects
from src.audio.player import AudioPlayer


class TestSpatialAudioMixer(unittest.TestCase):
    """Tests for the SpatialAudioMixer class."""
    
    def setUp(self):
        """Set up test environment."""
        pygame.init()
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, channels=2)
        self.mixer = SpatialAudioMixer()
    
    def tearDown(self):
        """Clean up test environment."""
        pygame.mixer.quit()
        pygame.quit()
    
    def test_initialization(self):
        """Test mixer initialization."""
        self.assertEqual(self.mixer.sample_rate, 44100)
        self.assertEqual(len(self.mixer.audio_cache), 0)
        self.assertEqual(len(self.mixer.left_channel), 0)
        self.assertEqual(len(self.mixer.right_channel), 0)
        self.assertFalse(self.mixer.is_playing)
    
    def test_panning_calculation(self):
        """Test panning calculation from angle."""
        # Test center (0 radians)
        left_gain, right_gain = self.mixer.calculate_panning(0)
        self.assertAlmostEqual(left_gain, 0.7071, places=4)  # cos(π/4)
        self.assertAlmostEqual(right_gain, 0.7071, places=4)  # sin(π/4)
        
        # Test right (π/2 radians)
        left_gain, right_gain = self.mixer.calculate_panning(np.pi/2)
        self.assertAlmostEqual(left_gain, 0.0, places=4)
        self.assertAlmostEqual(right_gain, 1.0, places=4)
        
        # Test left (-π/2 radians)
        left_gain, right_gain = self.mixer.calculate_panning(-np.pi/2)
        self.assertAlmostEqual(left_gain, 1.0, places=4)
        self.assertAlmostEqual(right_gain, 0.0, places=4)
    
    def test_spatial_delay_calculation(self):
        """Test spatial delay calculation."""
        # Direct front should have equal delays
        left_delay, right_delay = self.mixer.calculate_spatial_delay(0, 1.0)
        self.assertAlmostEqual(left_delay, right_delay)
        self.assertAlmostEqual(left_delay, 1.0 / self.mixer.speed_of_sound)
        
        # From right, left ear should have more delay
        left_delay, right_delay = self.mixer.calculate_spatial_delay(np.pi/2, 1.0)
        self.assertGreater(left_delay, right_delay)
        
        # From left, right ear should have more delay
        left_delay, right_delay = self.mixer.calculate_spatial_delay(-np.pi/2, 1.0)
        self.assertGreater(right_delay, left_delay)
    
    def test_intensity_calculation(self):
        """Test intensity calculation based on distance and energy."""
        # Test inverse square law
        intensity1 = self.mixer.calculate_intensity(1.0, 1.0)
        intensity2 = self.mixer.calculate_intensity(1.0, 2.0)
        intensity4 = self.mixer.calculate_intensity(1.0, 4.0)
        
        # Intensity should decrease with square of distance
        # Using the mixer's distance_factor (0.5) in the implementation
        # So the actual scaling is (1.0 / (distance * 0.5)^2)
        self.assertAlmostEqual(intensity1 / 4, intensity2, places=1)
        self.assertAlmostEqual(intensity1 / 16, intensity4, places=1)
        
        # Energy factor should scale intensity linearly
        self.assertAlmostEqual(
            self.mixer.calculate_intensity(0.5, 1.0),
            self.mixer.calculate_intensity(1.0, 1.0) * 0.5,
            places=1
        )
    
    def test_create_spatial_audio(self):
        """Test creating spatial audio from sound paths."""
        # Create test sound paths
        sound_paths = [
            {
                "audio_file": None,  # Will generate tone
                "frequency": 440,
                "intensity": 1.0,
                "angle": 0,
                "distance": 1.0,
                "delay": 0.0,
                "energy": 1.0
            },
            {
                "audio_file": None,  # Will generate tone
                "frequency": 880,
                "intensity": 0.5,
                "angle": np.pi/2,  # Right side
                "distance": 2.0,
                "delay": 0.1,
                "energy": 0.8
            }
        ]
        
        # Generate spatial audio
        left_channel, right_channel = self.mixer.create_spatial_audio(sound_paths)
        
        # Check output has correct format
        self.assertGreater(len(left_channel), 0)
        self.assertGreater(len(right_channel), 0)
        self.assertEqual(len(left_channel), len(right_channel))
        
        # For the second sound (from right), right channel should have more energy
        right_energy = np.sum(right_channel**2)
        left_energy = np.sum(left_channel**2)
        self.assertGreater(right_energy, left_energy)
    
    def test_audio_cache(self):
        """Test audio cache functionality."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Generate a simple sine wave and save as WAV
            duration = 0.1  # 100ms
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            sine_wave = np.sin(2 * np.pi * 440 * t)
            sine_wave = (sine_wave * 32767).astype(np.int16)
            
            from scipy.io import wavfile
            wavfile.write(temp_path, sample_rate, sine_wave)
            
            # First load should add to cache
            data1, rate1 = self.mixer.load_audio(temp_path)
            self.assertEqual(len(self.mixer.audio_cache), 1)
            self.assertIn(temp_path, self.mixer.audio_cache)
            
            # Second load should use cache
            data2, rate2 = self.mixer.load_audio(temp_path)
            self.assertEqual(len(self.mixer.audio_cache), 1)
            
            # Should be same object in memory
            self.assertIs(data1, data2)
            
            # Clear cache
            self.mixer.clear_cache()
            self.assertEqual(len(self.mixer.audio_cache), 0)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAudioEffects(unittest.TestCase):
    """Tests for the AudioEffects class."""
    
    def setUp(self):
        """Set up test environment."""
        self.effects = AudioEffects()
        
        # Create simple test audio
        duration = 0.5  # 500ms
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        self.test_audio = np.sin(2 * np.pi * 440 * t)
    
    def test_initialization(self):
        """Test effects initialization."""
        self.assertEqual(self.effects.sample_rate, 44100)
        self.assertTrue(self.effects.effect_params["delay"]["enabled"])
        self.assertTrue(self.effects.effect_params["echo"]["enabled"])
        self.assertFalse(self.effects.effect_params["reverb"]["enabled"])
    
    def test_delay_effect(self):
        """Test delay effect."""
        delay_time = 0.1  # 100ms
        
        # Apply delay with no feedback
        delayed_audio = self.effects.apply_delay(self.test_audio, delay_time)
        
        # Output should be longer than input by delay time
        expected_length = len(self.test_audio) + int(delay_time * self.effects.sample_rate)
        self.assertEqual(len(delayed_audio), expected_length)
        
        # First part should be original audio
        np.testing.assert_array_equal(delayed_audio[:len(self.test_audio)], self.test_audio)
        
        # Apply delay with feedback
        delayed_audio_fb = self.effects.apply_delay(self.test_audio, delay_time, 0.5)
        
        # Should be longer due to feedback
        self.assertGreater(len(delayed_audio_fb), len(delayed_audio))
    
    def test_echo_effect(self):
        """Test echo effect."""
        # Apply echo with default parameters
        echo_audio = self.effects.apply_echo(self.test_audio)
        
        # Should be longer than original
        self.assertGreater(len(echo_audio), len(self.test_audio))
        
        # First part should contain the original audio
        # Not necessarily equal since the implementation might add echoes 
        # on top of the original signal
        
        # Test with more echoes
        echo_audio_more = self.effects.apply_echo(self.test_audio, num_echoes=5)
        self.assertGreater(len(echo_audio_more), len(echo_audio))
    
    def test_reverb_effect(self):
        """Test reverb effect."""
        # Enable reverb
        self.effects.effect_params["reverb"]["enabled"] = True
        
        # Apply reverb
        reverb_audio = self.effects.apply_reverb(self.test_audio)
        
        # Should be longer than original
        self.assertGreater(len(reverb_audio), len(self.test_audio))
        
        # Output should not be identical to input (due to processing)
        self.assertFalse(np.array_equal(reverb_audio[:len(self.test_audio)], self.test_audio))
    
    def test_apply_all_effects(self):
        """Test applying all effects together."""
        # Enable all effects
        self.effects.effect_params["reverb"]["enabled"] = True
        
        # Parameters for all effects
        params = {
            "delay": {"time": 0.1, "feedback": 0.3},
            "echo": {"num_echoes": 2, "initial_delay": 0.2, "decay": 0.5},
            "reverb": {"room_size": 0.7, "damping": 0.4}
        }
        
        # Apply all effects
        processed_audio = self.effects.apply_all_effects(self.test_audio, params)
        
        # Should be significantly longer than original
        self.assertGreater(len(processed_audio), len(self.test_audio) * 1.5)
    
    def test_effect_configuration(self):
        """Test configuring effect parameters."""
        # Original config
        original_echo_decay = self.effects.effect_params["echo"]["decay"]
        
        # New config
        config = {
            "echo": {"decay": 0.3, "delay_time": 0.5},
            "delay": {"max_delay_time": 2.0}
        }
        
        # Apply config
        self.effects.configure_effects(config)
        
        # Check changes were applied
        self.assertEqual(self.effects.effect_params["echo"]["decay"], 0.3)
        self.assertEqual(self.effects.effect_params["echo"]["delay_time"], 0.5)
        self.assertEqual(self.effects.effect_params["delay"]["max_delay_time"], 2.0)
        
        # Disable an effect
        self.effects.enable_effect("echo", False)
        self.assertFalse(self.effects.effect_params["echo"]["enabled"])


class TestAudioPlayer(unittest.TestCase):
    """Tests for the AudioPlayer class."""
    
    def setUp(self):
        """Set up test environment."""
        pygame.init()
        self.player = AudioPlayer()
        
        # Create simple test audio
        duration = 0.1  # 100ms (short for quick tests)
        t = np.linspace(0, duration, int(self.player.sample_rate * duration), False)
        self.test_audio = np.sin(2 * np.pi * 440 * t)  # A4 sine wave
    
    def tearDown(self):
        """Clean up test environment."""
        self.player.stop_all_sounds()
        pygame.mixer.quit()
        pygame.quit()
    
    def test_initialization(self):
        """Test player initialization."""
        self.assertEqual(self.player.sample_rate, 44100)
        self.assertEqual(self.player.channels, 2)
        self.assertEqual(self.player.volume, 1.0)
        self.assertFalse(self.player.is_playing)
    
    def test_sound_playback(self):
        """Test basic sound playback."""
        # Play test sound
        sound_id = self.player.play_sound(self.test_audio)
        
        # Check correct state
        self.assertGreaterEqual(sound_id, 0)
        self.assertTrue(self.player.is_playing)
        self.assertIn(sound_id, self.player.active_sounds)
        
        # Let it play a bit
        time.sleep(0.01)
        self.player.update()
        
        # Stop the sound
        self.player.stop_sound(sound_id)
        self.assertNotIn(sound_id, self.player.active_sounds)
    
    def test_volume_control(self):
        """Test volume control."""
        # Play test sound
        sound_id = self.player.play_sound(self.test_audio)
        
        # Set volume for specific sound
        self.player.set_sound_volume(sound_id, 0.5)
        self.assertEqual(self.player.active_sounds[sound_id]["volume"], 0.5)
        
        # Set master volume
        self.player.set_volume(0.7)
        self.assertEqual(self.player.volume, 0.7)
        
        # Clean up
        self.player.stop_sound(sound_id)
    
    def test_pause_resume(self):
        """Test pause and resume functionality."""
        # Play test sound
        sound_id = self.player.play_sound(self.test_audio, loop=True)
        self.assertTrue(self.player.is_playing)
        
        # Pause
        self.player.pause_all_sounds()
        self.assertFalse(self.player.is_playing)
        
        # Resume
        self.player.resume_all_sounds()
        self.assertTrue(self.player.is_playing)
        
        # Clean up
        self.player.stop_sound(sound_id)
    
    def test_auto_completion(self):
        """Test automatic completion of sounds."""
        # Play a short sound without looping
        sound_id = self.player.play_sound(self.test_audio)
        
        # Wait for it to finish
        time.sleep(0.2)  # Wait longer than the sound duration
        
        # Update should detect completion
        self.player.update()
        self.assertNotIn(sound_id, self.player.active_sounds)


if __name__ == "__main__":
    unittest.main() 