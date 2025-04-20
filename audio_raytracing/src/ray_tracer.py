import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import time

from src.environment import Environment
from src.obstacles import Obstacle
from src.player import Player
from src.sound_source import SoundSource


class Ray:
    """
    Represents a single sound ray for ray tracing.
    
    Rays are emitted from sound sources and travel through the environment,
    potentially reflecting off obstacles before reaching the player/listener.
    """
    
    def __init__(self, origin: Tuple[float, float], direction: Tuple[float, float], 
                 energy: float = 1.0, source=None):
        """
        Initialize a ray.
        
        Args:
            origin: (x, y) coordinates of the ray's starting point
            direction: (dx, dy) normalized direction vector
            energy: Initial energy of the ray (0.0 to 1.0)
            source: Reference to the source that emitted this ray
        """
        self.origin = origin
        
        # Normalize the direction vector
        dir_magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
        if dir_magnitude > 0:
            self.direction = (direction[0]/dir_magnitude, direction[1]/dir_magnitude)
        else:
            self.direction = (1.0, 0.0)  # Default direction if zero vector provided
        
        self.energy = energy
        self.source = source
        
        # Ray path history (for visualization)
        self.path = [origin]
        
        # Reflection count
        self.reflections = 0
    
    def get_point_at_distance(self, distance: float) -> Tuple[float, float]:
        """
        Calculate the point at a certain distance along the ray.
        
        Args:
            distance: Distance from origin
            
        Returns:
            (x, y) coordinates of the point
        """
        return (
            self.origin[0] + self.direction[0] * distance,
            self.origin[1] + self.direction[1] * distance
        )
    
    def reflect(self, intersection: Tuple[float, float], normal: Tuple[float, float], 
                energy_factor: float) -> None:
        """
        Update the ray for a reflection at the given intersection point.
        
        Args:
            intersection: (x, y) coordinates of reflection point
            normal: (nx, ny) normalized surface normal vector
            energy_factor: Factor to multiply the ray's energy by (absorption)
        """
        # Calculate reflection direction using the formula: r = d - 2(d·n)n
        # where d is incident direction, n is surface normal
        dot_product = self.direction[0] * normal[0] + self.direction[1] * normal[1]
        reflection_dir = (
            self.direction[0] - 2 * dot_product * normal[0],
            self.direction[1] - 2 * dot_product * normal[1]
        )
        
        # Update ray properties
        self.origin = intersection
        self.direction = reflection_dir
        self.energy *= energy_factor
        self.reflections += 1
        
        # Add reflection point to path
        self.path.append(intersection)


class RayTracer:
    """
    Implements the ray tracing algorithm for sound propagation.
    
    This class handles casting rays from sound sources, tracking their paths
    through the environment, detecting intersections with obstacles, and
    calculating sound properties at the listener position.
    """
    
    def __init__(self, environment: Environment):
        """
        Initialize the ray tracer.
        
        Args:
            environment: The environment to trace rays in
        """
        self.environment = environment
        
        # Ray tracing parameters
        self.ray_count = 360  # Number of rays to cast per sound source
        self.max_reflections = 3  # Maximum number of reflections per ray
        self.max_distance = 1000.0  # Maximum distance a ray can travel
        self.min_energy = 0.0001  # Minimum energy threshold for ray propagation (lowered from 0.01)
        
        # Current ray tracing results
        self.rays = []  # List of all rays
        self.sound_path_data = []  # Data for rays that hit the player
        
        # Animation state
        self.is_animating = False
        self.animation_speed = 1.0  # Speed multiplier for visualization
        self.current_animation_time = 0.0  # Current time in animation
        
        # Cache lists for performance
        self.sound_sources = []
        self.obstacles = []
        
        # Physics
        self.speed_of_sound = 343.0  # m/s in air at room temperature
    
    def trace_rays(self) -> List[Ray]:
        """
        Perform ray tracing from the player toward sound sources in the environment,
        ensuring all visible rays come from the player.
        
        Returns:
            List of all rays traced
        """
        # Reset ray tracing data
        self.rays = []
        self.sound_path_data = []
        
        # Get environment objects
        sources = self.environment.sound_sources
        obstacles = self.environment.obstacles
        player = self.environment.player
        boundaries = self.environment.boundaries
        
        # Store sound sources for later lookup
        self.sound_sources = sources
        
        # Create obstacle-like objects from environment boundaries
        boundary_obstacles = []
        for x1, y1, x2, y2, material in boundaries:
            from src.obstacles import LineObstacle
            boundary = LineObstacle((x1, y1), (x2, y2), material)
            boundary_obstacles.append(boundary)
        
        # Combine regular obstacles and boundaries
        all_obstacles = obstacles + boundary_obstacles
        
        # Cache active sources
        active_sources = [source for source in sources if source.active]
        if not active_sources or not player:
            return self.rays
        
        # Calculate direct path data for audio (but don't create visible rays)
        for source in active_sources:
            # Calculate direct path
            to_player = (
                player.position[0] - source.position[0],
                player.position[1] - source.position[1]
            )
            distance = math.sqrt(to_player[0]**2 + to_player[1]**2)
            if distance > 0:
                direction = (to_player[0] / distance, to_player[1] / distance)
                
                # Check for obstacles in the direct path
                has_direct_path = True
                
                # Check for obstacle intersections without creating a visible ray
                for obstacle in all_obstacles:
                    intersection_data = obstacle.intersects_ray(source.position, direction)
                    if intersection_data:
                        hit_distance, _, _ = intersection_data
                        if hit_distance < distance and hit_distance > 0.1:  # Some small threshold
                            has_direct_path = False
                            break
                
                # Always add direct path data for audio with adjusted energy based on obstacles
                energy_factor = 1.0 if has_direct_path else 0.5  # Reduced energy if obstructed
                
                # Calculate sound intensity for direct path
                intensity = self._calculate_distance_attenuation(distance)
                
                # Calculate delay
                delay = distance / self.environment.sound_speed
                
                # Add to sound path data (but not to visible rays list)
                self.sound_path_data.append({
                    "direct_audio_only": True,  # Mark as audio-only path (not visually represented)
                    "source": source,
                    "distance": distance,
                    "intensity": intensity * energy_factor,
                    "delay": delay,
                    "energy": energy_factor,
                    "reflection_count": 0
                })
            
        # Cast rays from player in all directions (for reflections and visualization)
        ray_directions = []
        angle_step = 2 * math.pi / self.ray_count
        for i in range(self.ray_count):
            angle = i * angle_step
            dx = math.cos(angle)
            dy = math.sin(angle)
            ray_directions.append((dx, dy))
        
        # Create and trace each ray from player
        for direction in ray_directions:
            # Create ray from player
            ray = Ray(player.position, direction, 1.0, None)  # No source initially
            self.rays.append(ray)
            
            # Trace this ray through the environment
            self._trace_ray_from_player(ray, all_obstacles, active_sources, player)
        
        return self.rays
        
    def _trace_ray_from_player(self, ray: Ray, obstacles: List[Obstacle], 
                              sources: List[SoundSource], player: Player) -> None:
        """
        Trace a ray from player, looking for collisions with sound sources and obstacles.
        When a source is hit, the ray is reflected back to check if it reaches the player.
        
        Args:
            ray: The ray to trace, starting from player
            obstacles: List of all obstacles (including boundaries)
            sources: List of active sound sources
            player: The player/listener
        """
        # Continue tracing until ray is too weak or has too many reflections
        while (ray.energy >= self.min_energy and 
               ray.reflections <= self.max_reflections):
            
            # Find the closest intersection with any obstacle
            closest_distance = float('inf')
            closest_intersection = None
            closest_normal = None
            closest_obstacle = None
            
            # Check obstacles
            for obstacle in obstacles:
                intersection_data = obstacle.intersects_ray(ray.origin, ray.direction)
                if intersection_data:
                    distance, intersection, normal = intersection_data
                    
                    # Check if this is the closest intersection so far
                    if distance < closest_distance and distance > 1e-6:  # Avoid self-intersection
                        closest_distance = distance
                        closest_intersection = intersection
                        closest_normal = normal
                        closest_obstacle = obstacle
            
            # Check if ray reaches any sound source before hitting obstacles
            closest_source = None
            source_distance = float('inf')
            source_hit_point = None
            
            for source in sources:
                # Simple circle-based collision for sound sources
                source_radius = 10.0  # Radius to detect collision with source
                
                # Calculate vector from ray origin to source
                to_source = (
                    source.position[0] - ray.origin[0],
                    source.position[1] - ray.origin[1]
                )
                
                # Project this vector onto the ray direction
                dot_product = to_source[0] * ray.direction[0] + to_source[1] * ray.direction[1]
                
                # If negative, source is behind the ray
                if dot_product < 0:
                    continue
                
                # Calculate closest approach point
                proj_x = ray.origin[0] + ray.direction[0] * dot_product
                proj_y = ray.origin[1] + ray.direction[1] * dot_product
                
                # Calculate squared distance from source to closest approach point
                dx = proj_x - source.position[0]
                dy = proj_y - source.position[1]
                distance_squared = dx*dx + dy*dy
                
                # Check if ray passes within source radius
                if distance_squared <= source_radius * source_radius and dot_product < source_distance:
                    # Calculate actual hit distance
                    source_distance = dot_product
                    source_hit_point = (proj_x, proj_y)
                    closest_source = source
            
            # If ray hits a source before an obstacle
            if closest_source and source_distance < closest_distance:
                # Add source hit point to ray path
                ray.path.append(source_hit_point)
                
                # Now cast a ray back to the player to see if it reaches
                back_origin = source_hit_point
                to_player = (
                    player.position[0] - back_origin[0],
                    player.position[1] - back_origin[1]
                )
                # Normalize
                distance_to_player = math.sqrt(to_player[0]**2 + to_player[1]**2)
                if distance_to_player > 0:
                    back_direction = (
                        to_player[0] / distance_to_player,
                        to_player[1] / distance_to_player
                    )
                    
                    # Create a temporary ray for back-tracing
                    back_ray = Ray(back_origin, back_direction, ray.energy, closest_source)
                    
                    # Check if this ray reaches the player without obstacles
                    back_hit, back_distance = self._check_player_hit(back_ray, player)
                    
                    if back_hit:
                        # Add player hit point to original ray path
                        hit_point = back_ray.get_point_at_distance(back_distance)
                        ray.path.append(hit_point)
                        
                        # Apply source properties to the ray
                        ray.source = closest_source
                        
                        # Calculate sound intensity based on total distance traveled
                        total_distance = source_distance + back_distance
                        intensity = self._calculate_distance_attenuation(total_distance)
                        
                        # Calculate propagation delay
                        total_path_length = self._calculate_ray_path_length(ray) + back_distance
                        delay = total_path_length / self.environment.sound_speed
                        
                        # Apply directional factor from sound source
                        angle_to_player = math.atan2(back_direction[1], back_direction[0])
                        directional_factor = closest_source.directional_factor(angle_to_player)
                        intensity *= directional_factor * closest_source.volume
                        
                        # Record sound path data for audio processing
                        self.sound_path_data.append({
                            "ray": ray,
                            "distance": total_distance,
                            "intensity": intensity,
                            "delay": delay,
                            "hit_point": hit_point,
                            "energy": ray.energy,
                            "reflection_count": ray.reflections
                        })
                
                # Ray has been fully traced, stop
                break
                
            elif closest_intersection:
                # Get material properties for reflection
                material = closest_obstacle.material
                material_props = self.environment.get_material_properties(material)
                reflection_factor = material_props["reflection"]
                
                # Apply reflection
                ray.reflect(closest_intersection, closest_normal, reflection_factor)
                
                # Check if the ray has traveled too far
                total_distance = self._calculate_ray_path_length(ray)
                if total_distance > self.max_distance:
                    break
            else:
                # No intersection found, ray goes out of bounds
                break
    
    def _check_player_hit(self, ray: Ray, player: Player) -> Tuple[bool, float]:
        """
        Check if a ray hits the player.
        
        Args:
            ray: The ray to check
            player: The player/listener
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
        """
        # Use a larger radius to make sound detection more generous
        player_radius = 20.0  # Increased from 5.0 to make player much easier to hit
        
        # Calculate vector from ray origin to player
        to_player = (
            player.position[0] - ray.origin[0],
            player.position[1] - ray.origin[1]
        )
        
        # Project this vector onto the ray direction
        dot_product = to_player[0] * ray.direction[0] + to_player[1] * ray.direction[1]
        
        # If negative, player is behind the ray
        if dot_product < 0:
            return False, float('inf')
        
        # Calculate closest approach point
        proj_x = ray.origin[0] + ray.direction[0] * dot_product
        proj_y = ray.origin[1] + ray.direction[1] * dot_product
        
        # Calculate squared distance from player to closest approach point
        dx = proj_x - player.position[0]
        dy = proj_y - player.position[1]
        distance_squared = dx*dx + dy*dy
        
        # Check if ray passes within player radius
        if distance_squared <= player_radius * player_radius:
            # Calculate actual hit distance 
            hit_distance = dot_product - math.sqrt(player_radius*player_radius - distance_squared)
            
            # If hit distance is negative, ray origin is inside player
            if hit_distance < 0:
                hit_distance = 0
            
            return True, hit_distance
        
        return False, float('inf')
    
    def _calculate_distance_attenuation(self, distance: float) -> float:
        """
        Calculate attenuation factor based on distance.
        
        Uses a modified inverse law for sound intensity to make sound audible at greater distances.
        
        Args:
            distance: Distance from source to receiver
            
        Returns:
            Attenuation factor (0.0 to 1.0)
        """
        # Avoid division by zero
        if distance < 1.0:
            distance = 1.0
        
        # Modified falloff (gentler than inverse square law)
        # Use 1/d instead of 1/d² for gameplay purposes
        attenuation = 1.0 / distance
        
        # Apply a smaller ambient absorption effect
        ambient_factor = 0.2 * self.environment.ambient_absorption
        attenuation *= math.exp(-ambient_factor * distance)
        
        return attenuation
    
    def _calculate_propagation_delay(self, ray: Ray) -> float:
        """
        Calculate the time delay for sound to travel along the ray's path.
        
        Args:
            ray: The ray to calculate delay for
            
        Returns:
            Delay time in seconds
        """
        path_length = self._calculate_ray_path_length(ray)
        speed_of_sound = self.environment.sound_speed
        
        return path_length / speed_of_sound
    
    def _calculate_ray_path_length(self, ray: Ray) -> float:
        """
        Calculate the total length of a ray's path.
        
        Args:
            ray: The ray to calculate path length for
            
        Returns:
            Path length in distance units
        """
        total_length = 0.0
        for i in range(1, len(ray.path)):
            x1, y1 = ray.path[i-1]
            x2, y2 = ray.path[i]
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_length += segment_length
        
        return total_length
    
    def update_animation(self, dt: float) -> None:
        """
        Update the ray tracing animation.
        
        Args:
            dt: Time step in seconds
        """
        if self.is_animating:
            self.current_animation_time += dt * self.animation_speed
    
    def start_animation(self) -> None:
        """Start the ray tracing animation."""
        self.current_animation_time = 0.0
        self.is_animating = True
    
    def pause_animation(self) -> None:
        """Pause the ray tracing animation."""
        self.is_animating = False
    
    def reset_animation(self) -> None:
        """Reset the ray tracing animation to the beginning."""
        self.current_animation_time = 0.0
    
    def get_visible_ray_segments(self) -> List[Dict[str, Any]]:
        """
        Get ray segments that should be visible at the current animation time.
        
        Returns:
            List of visible ray segment data for rendering
        """
        if not self.rays:
            return []
        
        visible_segments = []
        speed_of_sound = self.environment.sound_speed * self.animation_speed
        current_distance = self.current_animation_time * speed_of_sound
        
        for ray in self.rays:
            path = ray.path
            if len(path) < 2:
                continue
            
            # Calculate cumulative distances along the path
            cumulative_distances = [0.0]
            total_distance = 0.0
            
            for i in range(1, len(path)):
                x1, y1 = path[i-1]
                x2, y2 = path[i]
                segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                total_distance += segment_length
                cumulative_distances.append(total_distance)
            
            # Determine which segments are visible
            for i in range(1, len(path)):
                start_distance = cumulative_distances[i-1]
                end_distance = cumulative_distances[i]
                
                # Skip segments that sound hasn't reached yet
                if start_distance > current_distance:
                    continue
                
                # For segments that sound is currently traveling through
                if start_distance <= current_distance <= end_distance:
                    # Calculate how far along this segment the sound has traveled
                    fraction = (current_distance - start_distance) / (end_distance - start_distance)
                    interpolated_end = (
                        path[i-1][0] + fraction * (path[i][0] - path[i-1][0]),
                        path[i-1][1] + fraction * (path[i][1] - path[i-1][1])
                    )
                    
                    visible_segments.append({
                        "start": path[i-1],
                        "end": interpolated_end,
                        "energy": ray.energy,
                        "source": ray.source
                    })
                
                # Segments sound has completely passed through
                elif end_distance < current_distance:
                    visible_segments.append({
                        "start": path[i-1],
                        "end": path[i],
                        "energy": ray.energy,
                        "source": ray.source
                    })
        
        return visible_segments
    
    def get_received_sound_at_time(self) -> List[Dict[str, Any]]:
        """
        Get the sounds received by the player at the current time.
        
        Returns:
            List of sound data objects, each containing source, intensity, angle, etc.
        """
        received_sounds = []
        
        # Get the number of paths for debugging
        print(f"Total sound_path_data: {len(self.sound_path_data)}")
        
        for path_data in self.sound_path_data:
            # Check if this is a direct audio-only path
            if path_data.get("direct_audio_only", False):
                source = path_data.get("source")
                if source:
                    # For direct audio paths, create sound data directly
                    sound_data = {
                        "source_id": self.sound_sources.index(source) if hasattr(self, 'sound_sources') and source in self.sound_sources else 0,
                        "intensity": path_data.get("intensity", 1.0),
                        "angle": 0.0,  # Direct path angle
                        "distance": path_data.get("distance", 1.0),
                        "delay": path_data.get("delay", 0.0),
                        "energy": path_data.get("energy", 1.0),
                        "reflection_count": path_data.get("reflection_count", 0),
                        "audio_file": source.audio_file if hasattr(source, 'audio_file') else None,
                        "frequency": source.frequency if hasattr(source, 'frequency') else 440
                    }
                    received_sounds.append(sound_data)
                continue
        
            # For animation mode, filter based on time
            if self.is_animating:
                # Calculate time required for sound to travel the path
                ray = path_data.get("ray")
                if ray:
                    total_distance = self._calculate_ray_path_length(ray)
                    travel_time = total_distance / self.environment.sound_speed
                    
                    # Skip paths that aren't received yet in animation
                    if travel_time > self.current_animation_time:
                        continue
            
            # Add to received sounds
            source = path_data.get("ray", {}).source if "ray" in path_data else None
            
            # Skip paths without valid source
            if not source:
                continue
                
            sound_data = {
                "source_id": self.sound_sources.index(source) if hasattr(self, 'sound_sources') and source in self.sound_sources else 0,
                "intensity": path_data.get("intensity", 1.0),
                "angle": path_data.get("angle", 0.0),
                "distance": path_data.get("distance", 1.0),
                "delay": path_data.get("delay", 0.0),
                "energy": path_data.get("energy", 1.0),
                "reflection_count": path_data.get("ray").reflections if "ray" in path_data else 0,
                "audio_file": source.audio_file if hasattr(source, 'audio_file') else None,
                "frequency": source.frequency if hasattr(source, 'frequency') else 440
            }
            
            received_sounds.append(sound_data)
        
        # Debug output
        print(f"Returned received_sounds: {len(received_sounds)}")
        
        return received_sounds 