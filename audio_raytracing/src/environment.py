import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional


class Environment:
    """
    Represents a 2D environment for audio ray tracing simulation.
    
    This class manages the environment boundaries, keeps track of all objects
    within the environment (obstacles, sound sources, and the player),
    and provides methods for environment management.
    """
    
    def __init__(self, width: int = 800, height: int = 600, name: str = "Default Environment"):
        """
        Initialize a new environment with specified dimensions.
        
        Args:
            width: Width of the environment in pixels/units
            height: Height of the environment in pixels/units
            name: Name of the environment
        """
        self.width = width
        self.height = height
        self.name = name
        
        # Lists to store objects in the environment
        self.obstacles = []
        self.sound_sources = []
        self.player = None
        
        # Environment properties
        self.sound_speed = 343.0  # m/s (speed of sound in air at room temperature)
        self.ambient_absorption = 0.0001  # Ambient absorption coefficient
        
        # Boundaries defined as walls
        self.boundaries = [
            # [x1, y1, x2, y2, material]
            [0, 0, width, 0, "wall"],           # Top wall
            [width, 0, width, height, "wall"],  # Right wall
            [width, height, 0, height, "wall"], # Bottom wall
            [0, height, 0, 0, "wall"]           # Left wall
        ]
        
        # Material properties for reflection and absorption
        self.materials = {
            "wall": {"reflection": 0.9, "absorption": 0.1},
            "wood": {"reflection": 0.8, "absorption": 0.2},
            "glass": {"reflection": 0.95, "absorption": 0.05},
            "carpet": {"reflection": 0.4, "absorption": 0.6},
            "concrete": {"reflection": 0.97, "absorption": 0.03}
        }
    
    def add_obstacle(self, obstacle) -> None:
        """
        Add an obstacle to the environment.
        
        Args:
            obstacle: Obstacle object to add
        """
        self.obstacles.append(obstacle)
    
    def add_sound_source(self, sound_source) -> None:
        """
        Add a sound source to the environment.
        
        Args:
            sound_source: SoundSource object to add
        """
        self.sound_sources.append(sound_source)
    
    def set_player(self, player) -> None:
        """
        Set the player/listener in the environment.
        
        Args:
            player: Player object to set
        """
        self.player = player
    
    def add_material(self, name: str, reflection: float, absorption: float) -> None:
        """
        Add a new material type to the environment.
        
        Args:
            name: Material name
            reflection: Reflection coefficient (0.0 to 1.0)
            absorption: Absorption coefficient (0.0 to 1.0)
        """
        if reflection + absorption > 1.0:
            # Normalize values if they sum to more than 1
            total = reflection + absorption
            reflection /= total
            absorption /= total
            
        self.materials[name] = {
            "reflection": reflection,
            "absorption": absorption
        }
    
    def get_material_properties(self, material_name: str) -> Dict[str, float]:
        """
        Get the properties of a specified material.
        
        Args:
            material_name: Name of the material
            
        Returns:
            Dictionary containing reflection and absorption coefficients
        """
        return self.materials.get(material_name, self.materials["wall"])
    
    def is_point_in_bounds(self, x: float, y: float) -> bool:
        """
        Check if a point is within the environment boundaries.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            True if point is within boundaries, False otherwise
        """
        return 0 <= x <= self.width and 0 <= y <= self.height
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the environment configuration to a JSON file.
        
        Args:
            filepath: Path to save the file
        """
        env_data = {
            "name": self.name,
            "dimensions": [self.width, self.height],
            "sound_speed": self.sound_speed,
            "ambient_absorption": self.ambient_absorption,
            "materials": self.materials,
            "boundaries": self.boundaries
        }
        
        # Add obstacles, player, and sound sources if they exist
        if self.obstacles:
            env_data["obstacles"] = [obstacle.to_dict() for obstacle in self.obstacles]
            
        if self.player:
            env_data["player"] = self.player.to_dict()
            
        if self.sound_sources:
            env_data["sound_sources"] = [source.to_dict() for source in self.sound_sources]
        
        with open(filepath, 'w') as f:
            json.dump(env_data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Environment':
        """
        Load an environment configuration from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Environment object with loaded configuration
        """
        from src.obstacles import Obstacle
        from src.player import Player
        from src.sound_source import SoundSource
        
        with open(filepath, 'r') as f:
            env_data = json.load(f)
        
        # Create environment with basic properties
        width, height = env_data.get("dimensions", [800, 600])
        env = cls(width=width, height=height, name=env_data.get("name", "Loaded Environment"))
        
        # Set environment properties
        env.sound_speed = env_data.get("sound_speed", 343.0)
        env.ambient_absorption = env_data.get("ambient_absorption", 0.0001)
        
        # Load materials if present
        if "materials" in env_data:
            env.materials = env_data["materials"]
            
        # Load boundaries if present
        if "boundaries" in env_data:
            env.boundaries = env_data["boundaries"]
        
        # Load obstacles if present
        if "obstacles" in env_data:
            for obstacle_data in env_data["obstacles"]:
                obstacle = Obstacle.from_dict(obstacle_data)
                env.add_obstacle(obstacle)
        
        # Load player if present
        if "player" in env_data:
            player = Player.from_dict(env_data["player"])
            env.set_player(player)
        
        # Load sound sources if present
        if "sound_sources" in env_data:
            for source_data in env_data["sound_sources"]:
                source = SoundSource.from_dict(source_data)
                env.add_sound_source(source)
                
        return env
    
    def __str__(self) -> str:
        """String representation of the environment."""
        return (f"Environment: {self.name} ({self.width}x{self.height}), "
                f"{len(self.obstacles)} obstacles, "
                f"{len(self.sound_sources)} sound sources, "
                f"{'Player exists' if self.player else 'No player'}") 