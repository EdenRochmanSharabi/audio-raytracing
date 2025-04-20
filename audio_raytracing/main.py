#!/usr/bin/env python3
import os
import sys
import time
import math
import argparse
import pygame
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment import Environment
from src.obstacles import RectangleObstacle, CircleObstacle, LineObstacle
from src.player import Player
from src.sound_source import SoundSource
from src.ray_tracer import RayTracer
from src.physics.simulation import Simulation
from src.audio.mixer import SpatialAudioMixer
from src.audio.effects import AudioEffects
from src.audio.player import AudioPlayer


class AudioRayTracingApp:
    """
    Main application class for the Audio Ray Tracing simulation.
    
    This class handles the pygame initialization, main loop, and user interactions.
    """
    
    def __init__(self, args):
        """
        Initialize the application.
        
        Args:
            args: Command line arguments
        """
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Audio Ray Tracing Simulation")
        
        # Set up the screen
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Set up fonts
        self.font = pygame.font.SysFont("Arial", 16)
        
        # Create environment
        if args.environment and os.path.exists(args.environment):
            self.environment = Environment.load_from_file(args.environment)
        else:
            self.environment = self.create_default_environment()
        
        # Create simulation
        self.simulation = Simulation(self.environment)
        
        # Set parameters from command line args
        if args.rays:
            self.simulation.set_ray_count(args.rays)
        if args.speed:
            self.environment.sound_speed = args.speed
            self.simulation.propagation_model.speed_of_sound = args.speed
        if args.animation_speed:
            self.simulation.set_animation_speed(args.animation_speed)
        
        # Initialize colors
        self.colors = {
            "background": (0, 0, 0),
            "text": (255, 255, 255),
            "obstacle": (100, 100, 100),
            "player": (0, 255, 0),
            "sound_source": (255, 0, 0),
            "ray": (255, 255, 0, 128),  # Yellow with alpha
            "boundary": (50, 50, 50)
        }
        
        # UI state
        self.show_help = False
        self.show_stats = True
        self.paused = False
        
        # Movement state
        self.keys_pressed = set()
        
        # Whether to trace rays on startup
        self.initial_trace_done = False
        
        # Initialize audio components
        self.audio_mixer = SpatialAudioMixer()
        self.audio_effects = AudioEffects()
        self.audio_player = AudioPlayer()
        
        # Audio state
        self.audio_enabled = True
        self.last_sound_process_time = 0
        self.sound_process_interval = 0.1  # Process sounds every 100ms
        
        # Sound paths from last ray trace
        self.sound_paths = []
        
        # Audio configuration
        if args.mute:
            self.audio_enabled = False
            
        if args.volume:
            self.audio_player.set_volume(args.volume)
    
    def create_default_environment(self) -> Environment:
        """
        Create a default environment for the simulation.
        
        Returns:
            Environment object with default setup
        """
        # Create environment
        env = Environment(width=self.width, height=self.height, name="Default Room")
        
        # Create player
        player = Player(position=(400, 300), orientation=0)
        env.set_player(player)
        
        # Create sound source
        source = SoundSource(
            position=(200, 200),
            name="Source 1",
            audio_file="assets/audio.wav"  # Use the existing audio.mp3 file
        )
        env.add_sound_source(source)
        
        # Create some obstacles
        # Rectangle in the middle
        rect = RectangleObstacle(
            position=(400, 200),
            dimensions=(100, 30),
            rotation=45,
            material="wood"
        )
        env.add_obstacle(rect)
        
        # Circle obstacle
        circle = CircleObstacle(
            position=(600, 400),
            radius=50,
            material="glass"
        )
        env.add_obstacle(circle)
        
        # Line obstacle
        line = LineObstacle(
            start=(100, 400),
            end=(300, 500),
            material="concrete"
        )
        env.add_obstacle(line)
        
        return env
    
    def run(self):
        """Run the main application loop."""
        running = True
        
        # Start simulation
        self.simulation.start()
        
        # Main loop
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    self.keys_pressed.add(event.key)
                    
                    # Handle single key presses
                    if not self.handle_keydown(event.key):
                        running = False
                
                elif event.type == pygame.KEYUP:
                    if event.key in self.keys_pressed:
                        self.keys_pressed.remove(event.key)
            
            # Handle continuous key presses (movement)
            self.handle_movement()
            
            # Perform initial ray trace if not done
            if not self.initial_trace_done:
                self.simulation.trace_rays()
                self.initial_trace_done = True
            
            # Update simulation
            dt = self.simulation.update()
            
            # Process audio if enabled
            if self.audio_enabled:
                current_time = time.time()
                if current_time - self.last_sound_process_time > self.sound_process_interval:
                    self.process_audio()
                    self.last_sound_process_time = current_time
            
            # Update audio player (check for completed sounds)
            self.audio_player.update()
            
            # Render
            self.render()
            
            # Cap at 60 FPS
            self.clock.tick(60)
        
        # Clean up pygame
        pygame.quit()
    
    def handle_movement(self):
        """Handle continuous key presses for player movement."""
        player = self.environment.player
        if not player:
            return
            
        # Movement speed
        speed = player.speed
        
        # Handle arrow keys for movement
        if pygame.K_UP in self.keys_pressed:
            self.simulation.move_player(0, -speed)
        if pygame.K_DOWN in self.keys_pressed:
            self.simulation.move_player(0, speed)
        if pygame.K_LEFT in self.keys_pressed:
            self.simulation.move_player(-speed, 0)
        if pygame.K_RIGHT in self.keys_pressed:
            self.simulation.move_player(speed, 0)
        
        # Handle rotation
        if pygame.K_q in self.keys_pressed:
            self.simulation.rotate_player(-player.rotation_speed)
        if pygame.K_e in self.keys_pressed:
            self.simulation.rotate_player(player.rotation_speed)
    
    def render(self):
        """Render the simulation to the screen."""
        # Clear screen
        self.screen.fill(self.colors["background"])
        
        # Draw environment boundaries
        for x1, y1, x2, y2, material in self.environment.boundaries:
            pygame.draw.line(self.screen, self.colors["boundary"], (x1, y1), (x2, y2), 2)
        
        # Draw obstacles
        for obstacle in self.environment.obstacles:
            if isinstance(obstacle, RectangleObstacle):
                pygame.draw.polygon(self.screen, self.colors["obstacle"], obstacle.vertices)
            elif isinstance(obstacle, CircleObstacle):
                pygame.draw.circle(self.screen, self.colors["obstacle"], 
                                  obstacle.position, obstacle.radius)
            elif isinstance(obstacle, LineObstacle):
                pygame.draw.line(self.screen, self.colors["obstacle"], 
                               obstacle.start, obstacle.end, 3)
        
        # Draw ray paths
        ray_segments = self.simulation.ray_tracer.get_visible_ray_segments()
        for segment in ray_segments:
            # Adjust alpha based on energy
            alpha = int(segment["energy"] * 128) + 32  # Range 32-160
            color = list(self.colors["ray"])
            color[3] = alpha
            
            # Create surface for alpha blending
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.line(s, color, segment["start"], segment["end"], 1)
            self.screen.blit(s, (0, 0))
        
        # Draw sound sources
        for source in self.environment.sound_sources:
            pygame.draw.circle(self.screen, self.colors["sound_source"], 
                              source.position, 6)
            
            # Draw direction indicator if directional
            if source.directional:
                # Calculate endpoint of direction line
                end_x = source.position[0] + 15 * math.cos(source.direction_rad)
                end_y = source.position[1] + 15 * math.sin(source.direction_rad)
                pygame.draw.line(self.screen, self.colors["sound_source"], 
                               source.position, (end_x, end_y), 2)
        
        # Draw player
        if self.environment.player:
            player = self.environment.player
            pygame.draw.circle(self.screen, self.colors["player"], 
                              player.position, 8)
            
            # Draw direction indicator
            dir_x = player.position[0] + 15 * math.cos(player.orientation_rad)
            dir_y = player.position[1] + 15 * math.sin(player.orientation_rad)
            pygame.draw.line(self.screen, self.colors["player"], 
                           player.position, (dir_x, dir_y), 2)
        
        # Draw statistics
        if self.show_stats:
            self.render_stats()
        
        # Draw help
        if self.show_help:
            self.render_help()
        
        # Update display
        pygame.display.flip()
    
    def render_stats(self):
        """Render statistics overlay."""
        stats = self.simulation.get_performance_stats()
        
        # Prepare stats text
        stats_text = [
            f"FPS: {stats['fps']}",
            f"Simulation Time: {stats['simulation_time']:.2f}s",
            f"Ray Count: {stats['ray_count']}",
            f"Animation Speed: {self.simulation.animation_speed:.1f}x",
            f"Received Sounds: {stats['received_sounds']}",
            f"Audio: {'Enabled' if self.audio_enabled else 'Disabled'}"
        ]
        
        # Render stats
        y = 10
        for text in stats_text:
            text_surf = self.font.render(text, True, self.colors["text"])
            self.screen.blit(text_surf, (10, y))
            y += 20
        
        # Render state
        state_text = "PAUSED" if self.paused else "RUNNING"
        state_surf = self.font.render(state_text, True, self.colors["text"])
        self.screen.blit(state_surf, (self.width - state_surf.get_width() - 10, 10))
    
    def render_help(self):
        """Render help overlay."""
        help_text = [
            "Controls:",
            "Arrow Keys: Move player",
            "Q/E: Rotate player",
            "Space: Pause/Resume",
            "R: Reset simulation",
            "+/-: Adjust animation speed",
            "M: Toggle audio",
            "H: Toggle help",
            "F1: Toggle stats",
            "ESC: Quit"
        ]
        
        # Create semi-transparent overlay
        s = pygame.Surface((300, len(help_text) * 20 + 20), pygame.SRCALPHA)
        s.fill((0, 0, 0, 192))  # Black with alpha
        
        # Render help text
        y = 10
        for text in help_text:
            text_surf = self.font.render(text, True, self.colors["text"])
            s.blit(text_surf, (10, y))
            y += 20
        
        # Position in center of screen
        self.screen.blit(s, ((self.width - s.get_width()) // 2, 
                           (self.height - s.get_height()) // 2))
    
    def handle_keydown(self, key):
        """Handle single key press events."""
        if key == pygame.K_ESCAPE:
            return False  # Signal to exit
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif key == pygame.K_F1:
            self.show_stats = not self.show_stats
        elif key == pygame.K_SPACE:
            if self.paused:
                self.simulation.resume()
            else:
                self.simulation.pause()
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.simulation.reset()
            self.initial_trace_done = False
            # Stop all audio
            self.audio_player.stop_all_sounds()
        elif key == pygame.K_EQUALS:  # + key
            self.simulation.set_animation_speed(self.simulation.animation_speed * 1.5)
        elif key == pygame.K_MINUS:
            self.simulation.set_animation_speed(self.simulation.animation_speed / 1.5)
        elif key == pygame.K_m:
            self.audio_enabled = not self.audio_enabled
            if not self.audio_enabled:
                self.audio_player.stop_all_sounds()
        
        return True  # Continue running
    
    def process_audio(self):
        """Process ray-traced sound paths into spatial audio."""
        try:
            # Get received sounds from simulation
            sound_paths = self.simulation.get_received_sounds()
            
            # Debug info
            print(f"Sound paths: {len(sound_paths)}")
            
            # Alternative: Get sound paths directly from ray tracer if simulation method fails
            if not sound_paths and self.simulation.ray_tracer.sound_path_data:
                sound_paths = self.simulation.ray_tracer.sound_path_data
                print(f"Using direct ray tracer data: {len(sound_paths)} paths")
            
            if sound_paths:
                print(f"First sound path: {sound_paths[0]}")
            
            # Skip if no sounds or audio disabled
            if not sound_paths or not self.audio_enabled:
                return
            
            # Only process if sounds have changed
            if sound_paths == self.sound_paths:
                return
                
            self.sound_paths = sound_paths
            
            # If we have valid sound paths, create spatial audio
            if sound_paths:
                try:
                    # Create spatial audio
                    left_channel, right_channel = self.audio_mixer.create_spatial_audio(sound_paths)
                    
                    # Debug info
                    print(f"Audio channels: L={len(left_channel)}, R={len(right_channel)}")
                    
                    # Apply audio effects
                    if len(left_channel) > 0 and len(right_channel) > 0:
                        try:
                            # Configure effects based on environment
                            effect_config = {
                                "echo": {
                                    "enabled": False,  # Disable echo
                                    "delay_time": 0.3,
                                    "decay": 0.5
                                },
                                "reverb": {
                                    "enabled": False,  # Disable reverb
                                    "room_size": min(1.0, self.environment.width / 1000),
                                    "damping": 0.5
                                }
                            }
                            self.audio_effects.configure_effects(effect_config)
                            
                            # Apply effects to each channel
                            left_channel = self.audio_effects.apply_all_effects(
                                left_channel, 
                                {"echo": {"num_echoes": 2}, "reverb": {}}
                            )
                            right_channel = self.audio_effects.apply_all_effects(
                                right_channel, 
                                {"echo": {"num_echoes": 2}, "reverb": {}}
                            )
                            
                            # Debug info
                            print(f"After effects: L={len(left_channel)}, R={len(right_channel)}")
                        except Exception as effect_error:
                            print(f"Error applying audio effects: {effect_error}")
                            # Continue with unprocessed audio
                        
                        try:
                            # Stop any currently playing sounds
                            self.audio_player.stop_all_sounds()
                            
                            # Play the processed audio
                            sound_id = self.audio_player.play_sound(
                                np.column_stack((left_channel, right_channel)), 
                                volume=1.0
                            )
                            print(f"Playing sound with ID: {sound_id}")
                            print(f"Player state: is_playing={self.audio_player.is_playing}, active_sounds={len(self.audio_player.active_sounds)}")
                        except Exception as play_error:
                            print(f"Error playing audio: {play_error}")
                except Exception as audio_error:
                    print(f"Error creating spatial audio: {audio_error}")
        except Exception as e:
            print(f"Error in process_audio: {e}")
            # Disable audio if there's a critical failure
            self.audio_enabled = False
            print("Audio disabled due to errors")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio Ray Tracing Simulation")
    
    parser.add_argument('-e', '--environment', type=str,
                       help='Path to environment JSON file')
    parser.add_argument('-r', '--rays', type=int, default=360,
                       help='Number of rays per sound source')
    parser.add_argument('-s', '--speed', type=float, default=343.0,
                       help='Speed of sound in meters/second')
    parser.add_argument('-a', '--animation-speed', type=float, default=1.0,
                       help='Animation speed multiplier')
    parser.add_argument('-m', '--mute', action='store_true',
                       help='Start with audio muted')
    parser.add_argument('-v', '--volume', type=float, default=1.0,
                       help='Set audio volume (0.0 to 1.0)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = AudioRayTracingApp(args)
    app.run() 