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


class AudioRayTracingVisualApp:
    """
    Visual-only version of the Audio Ray Tracing simulation.
    
    This class handles the pygame initialization, main loop, and user interactions.
    The audio components are removed to avoid dependencies on audio libraries.
    """
    
    def __init__(self, args):
        """
        Initialize the application.
        
        Args:
            args: Command line arguments
        """
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Audio Ray Tracing Simulation (Visual Only)")
        
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
        
        # Create ray tracer
        self.ray_tracer = RayTracer(self.environment)
        
        # Set parameters from command line args
        if args.rays:
            self.ray_tracer.ray_count = args.rays
        if args.speed:
            self.environment.sound_speed = args.speed
        if args.animation_speed:
            self.ray_tracer.animation_speed = args.animation_speed
        
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
        
        # Timing and performance
        self.last_trace_time = time.time()
        self.fps = 0
        self.simulation_time = 0
    
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
            audio_file="assets/audio.wav"  # Reference file, not used in visual-only mode
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
        prev_time = time.time()
        
        # Main loop
        while running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            
            # Calculate FPS
            if dt > 0:
                self.fps = int(1 / dt)
            
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
                self.ray_tracer.trace_rays()
                self.initial_trace_done = True
            
            # Update ray tracer animation
            if not self.paused:
                self.ray_tracer.update_animation(dt)
                self.simulation_time += dt
            
            # Render
            self.render()
            
            # Cap at 60 FPS
            self.clock.tick(60)
        
        # Clean up pygame
        pygame.quit()
    
    def handle_movement(self):
        """Handle continuous key presses for player movement."""
        player = self.environment.player
        if not player or self.paused:
            return
            
        # Movement speed
        speed = player.speed
        
        # Handle arrow keys for movement
        if pygame.K_UP in self.keys_pressed:
            player.move(0, -speed)
            self.ray_tracer.trace_rays()  # Re-trace rays when player moves
        if pygame.K_DOWN in self.keys_pressed:
            player.move(0, speed)
            self.ray_tracer.trace_rays()
        if pygame.K_LEFT in self.keys_pressed:
            player.move(-speed, 0)
            self.ray_tracer.trace_rays()
        if pygame.K_RIGHT in self.keys_pressed:
            player.move(speed, 0)
            self.ray_tracer.trace_rays()
        
        # Handle rotation
        if pygame.K_q in self.keys_pressed:
            player.rotate(-player.rotation_speed)
            self.ray_tracer.trace_rays()
        if pygame.K_e in self.keys_pressed:
            player.rotate(player.rotation_speed)
            self.ray_tracer.trace_rays()
    
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
        ray_segments = self.ray_tracer.get_visible_ray_segments()
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
        # Get ray information
        ray_count = len(self.ray_tracer.rays) if hasattr(self.ray_tracer, 'rays') else 0
        sound_path_count = len(self.ray_tracer.sound_path_data) if hasattr(self.ray_tracer, 'sound_path_data') else 0
        
        # Prepare stats text
        stats_text = [
            f"FPS: {self.fps}",
            f"Simulation Time: {self.simulation_time:.2f}s",
            f"Ray Count: {ray_count}",
            f"Animation Speed: {self.ray_tracer.animation_speed:.1f}x",
            f"Received Sounds: {sound_path_count}",
            f"Audio: Disabled (Visual Only)"
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
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.ray_tracer.reset_animation()
            self.initial_trace_done = False
            self.simulation_time = 0
        elif key == pygame.K_EQUALS:  # + key
            self.ray_tracer.animation_speed *= 1.5
        elif key == pygame.K_MINUS:
            self.ray_tracer.animation_speed /= 1.5
        
        return True  # Continue running


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio Ray Tracing Simulation (Visual Only)")
    
    parser.add_argument('-e', '--environment', type=str,
                       help='Path to environment JSON file')
    parser.add_argument('-r', '--rays', type=int, default=360,
                       help='Number of rays per sound source')
    parser.add_argument('-s', '--speed', type=float, default=343.0,
                       help='Speed of sound in meters/second')
    parser.add_argument('-a', '--animation-speed', type=float, default=1.0,
                       help='Animation speed multiplier')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = AudioRayTracingVisualApp(args)
    app.run() 