import math
import numpy as np
import time
from typing import Dict, Tuple, List, Optional, Any

from src.environment import Environment
from src.player import Player
from src.sound_source import SoundSource
from src.ray_tracer import RayTracer
from src.physics.reflection import ReflectionModel
from src.physics.propagation import PropagationModel


class Simulation:
    """
    Main simulation class that coordinates the physics models and updates the environment.
    
    This class handles the time stepping, updates for simulation objects, and
    coordinates the ray tracing and audio processing.
    """
    
    def __init__(self, environment: Environment):
        """
        Initialize the simulation with an environment.
        
        Args:
            environment: The environment to simulate
        """
        self.environment = environment
        
        # Create the physics models
        self.reflection_model = ReflectionModel()
        self.propagation_model = PropagationModel(speed_of_sound=environment.sound_speed)
        self.ray_tracer = RayTracer(environment)
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.simulation_time = 0.0  # Current simulation time in seconds
        self.time_scale = 1.0  # Time scale factor (1.0 = real-time)
        
        # Previous frame time for calculating time step
        self.previous_time = None
        
        # Limit for excessive time steps (prevents large jumps)
        self.max_time_step = 0.1  # seconds
        
        # Animation properties
        self.animation_speed = 1.0  # Visualization speed of sound propagation
        
        # Performance metrics
        self.frame_count = 0
        self.last_fps_update = 0
        self.fps = 0
    
    def start(self) -> None:
        """Start or resume the simulation."""
        self.is_running = True
        self.is_paused = False
        self.previous_time = time.time()
        
        # Start ray tracing animation
        self.ray_tracer.start_animation()
    
    def pause(self) -> None:
        """Pause the simulation."""
        self.is_paused = True
        
        # Pause ray tracing animation
        self.ray_tracer.pause_animation()
    
    def resume(self) -> None:
        """Resume the simulation from a paused state."""
        if self.is_running and self.is_paused:
            self.is_paused = False
            self.previous_time = time.time()
            
            # Resume ray tracing animation
            self.ray_tracer.is_animating = True
    
    def stop(self) -> None:
        """Stop the simulation."""
        self.is_running = False
        self.is_paused = False
        self.simulation_time = 0.0
        
        # Reset animation
        self.ray_tracer.reset_animation()
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        self.simulation_time = 0.0
        
        # Reset animation
        self.ray_tracer.reset_animation()
    
    def update(self) -> float:
        """
        Update the simulation for one time step.
        
        Returns:
            Actual time step in seconds
        """
        # Skip update if not running or paused
        if not self.is_running or self.is_paused:
            return 0.0
        
        # Calculate time step
        current_time = time.time()
        if self.previous_time is None:
            self.previous_time = current_time
            return 0.0
            
        dt = (current_time - self.previous_time) * self.time_scale
        self.previous_time = current_time
        
        # Limit dt to prevent large jumps
        dt = min(dt, self.max_time_step)
        
        # Update simulation time
        self.simulation_time += dt
        
        # Update animation
        self.ray_tracer.update_animation(dt)
        
        # Track performance
        self.frame_count += 1
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update = current_time
        
        return dt
    
    def trace_rays(self) -> List[Dict[str, Any]]:
        """
        Perform ray tracing calculation.
        
        Returns:
            Ray tracing visualization data
        """
        # Run the ray tracer
        self.ray_tracer.trace_rays()
        
        # Get ray visualization data
        return self.ray_tracer.get_visible_ray_segments()
    
    def set_animation_speed(self, speed: float) -> None:
        """
        Set the animation speed for sound propagation visualization.
        
        Args:
            speed: Animation speed multiplier
        """
        self.animation_speed = max(0.1, speed)  # Minimum speed of 0.1
        self.ray_tracer.animation_speed = self.animation_speed
    
    def set_time_scale(self, scale: float) -> None:
        """
        Set the time scale for the simulation.
        
        Args:
            scale: Time scale factor (1.0 = real-time)
        """
        self.time_scale = max(0.01, scale)  # Minimum scale of 0.01
    
    def get_received_sounds(self) -> List[Dict[str, Any]]:
        """
        Get the sounds received by the player at the current time.
        
        Returns:
            List of sound data objects
        """
        return self.ray_tracer.get_received_sound_at_time()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get simulation performance statistics.
        
        Returns:
            Dictionary of performance stats
        """
        return {
            "fps": self.fps,
            "simulation_time": self.simulation_time,
            "ray_count": len(self.ray_tracer.rays),
            "received_sounds": len(self.ray_tracer.sound_path_data)
        }
    
    def update_environment(self, new_environment: Environment) -> None:
        """
        Update the environment being simulated.
        
        Args:
            new_environment: New environment to simulate
        """
        self.environment = new_environment
        self.ray_tracer.environment = new_environment
        self.propagation_model.speed_of_sound = new_environment.sound_speed
        
        # Reset simulation state
        self.reset()
    
    def move_player(self, dx: float, dy: float) -> None:
        """
        Move the player by the specified delta.
        
        Args:
            dx: Change in x position
            dy: Change in y position
        """
        if self.environment.player:
            self.environment.player.move(dx, dy)
            
            # Ray tracing needs to be recalculated after player moves
            if not self.is_paused:
                self.trace_rays()
    
    def rotate_player(self, angle: float) -> None:
        """
        Rotate the player by the specified angle.
        
        Args:
            angle: Angle to rotate in degrees
        """
        if self.environment.player:
            self.environment.player.rotate(angle)
    
    def add_sound_source(self, source: SoundSource) -> None:
        """
        Add a new sound source to the environment.
        
        Args:
            source: Sound source to add
        """
        self.environment.add_sound_source(source)
        
        # Ray tracing needs to be recalculated after adding source
        if not self.is_paused:
            self.trace_rays()
    
    def set_ray_count(self, count: int) -> None:
        """
        Set the number of rays to cast from each sound source.
        
        Args:
            count: Number of rays
        """
        self.ray_tracer.ray_count = max(8, min(1000, count))  # Limit between 8 and 1000
    
    def set_max_reflections(self, reflections: int) -> None:
        """
        Set the maximum number of reflections per ray.
        
        Args:
            reflections: Maximum reflection count
        """
        self.ray_tracer.max_reflections = max(0, min(10, reflections))  # Limit between 0 and 10 