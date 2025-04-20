import math
from typing import Dict, Any, Tuple, Optional


class Player:
    """
    Represents the listener/player in the audio ray tracing environment.
    
    The player has a position in the environment and an orientation.
    It acts as the receiver for sound waves that propagate through the
    environment.
    """
    
    def __init__(self, position: Tuple[float, float] = (400, 300), orientation: float = 0):
        """
        Initialize a player/listener.
        
        Args:
            position: (x, y) coordinates of the player in the environment
            orientation: Angle in degrees (0 is right, 90 is up, etc.)
        """
        self.position = position
        self.orientation = orientation  # in degrees
        self.orientation_rad = math.radians(orientation)
        
        # Player movement properties
        self.speed = 5.0  # units per step
        self.rotation_speed = 5.0  # degrees per step
        
        # Hearing properties
        self.hearing_sensitivity = 1.0  # Multiplier for received sound intensity
        self.direction_sensitivity = 0.5  # How much direction affects hearing (0-1)
    
    def move(self, dx: float, dy: float) -> None:
        """
        Move the player by the specified delta.
        
        Args:
            dx: Change in x position
            dy: Change in y position
        """
        self.position = (self.position[0] + dx, self.position[1] + dy)
    
    def move_forward(self, distance: float = None) -> None:
        """
        Move the player forward in the direction they are facing.
        
        Args:
            distance: Distance to move (if None, uses self.speed)
        """
        if distance is None:
            distance = self.speed
            
        dx = distance * math.cos(self.orientation_rad)
        dy = distance * math.sin(self.orientation_rad)
        self.move(dx, dy)
    
    def rotate(self, angle: float) -> None:
        """
        Rotate the player by the specified angle.
        
        Args:
            angle: Angle to rotate in degrees (positive is counterclockwise)
        """
        self.orientation = (self.orientation + angle) % 360
        self.orientation_rad = math.radians(self.orientation)
    
    def set_orientation(self, angle: float) -> None:
        """
        Set the player's orientation to a specific angle.
        
        Args:
            angle: New orientation angle in degrees
        """
        self.orientation = angle % 360
        self.orientation_rad = math.radians(self.orientation)
    
    def get_direction_vector(self) -> Tuple[float, float]:
        """
        Get the unit vector representing the player's orientation.
        
        Returns:
            (dx, dy) normalized direction vector
        """
        return (math.cos(self.orientation_rad), math.sin(self.orientation_rad))
    
    def get_ear_positions(self, ear_distance: float = 1.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate positions of left and right ears based on orientation.
        
        This is useful for stereo audio simulation.
        
        Args:
            ear_distance: Distance between ears in world units
            
        Returns:
            Tuple of ((left_ear_x, left_ear_y), (right_ear_x, right_ear_y))
        """
        # Calculate perpendicular vector (for ear separation)
        perp_angle = self.orientation_rad + math.pi/2
        perp_x = math.cos(perp_angle)
        perp_y = math.sin(perp_angle)
        
        half_distance = ear_distance / 2
        left_ear = (
            self.position[0] + perp_x * half_distance,
            self.position[1] + perp_y * half_distance
        )
        
        right_ear = (
            self.position[0] - perp_x * half_distance,
            self.position[1] - perp_y * half_distance
        )
        
        return (left_ear, right_ear)
    
    def calculate_hearing_factor(self, source_position: Tuple[float, float]) -> float:
        """
        Calculate a factor representing how well the player can hear from a source.
        
        This takes into account the player's orientation relative to the source.
        Sound is heard better when it comes from in front of the player.
        
        Args:
            source_position: (x, y) coordinates of the sound source
            
        Returns:
            Hearing factor from 0.0 to 1.0
        """
        if self.direction_sensitivity == 0:
            return 1.0  # No directional sensitivity
        
        # Calculate vector from player to source
        dx = source_position[0] - self.position[0]
        dy = source_position[1] - self.position[1]
        
        # Calculate angle between player's orientation and source direction
        if dx == 0 and dy == 0:
            return 1.0  # Source is at same position as player
        
        source_angle = math.atan2(dy, dx)
        angle_diff = abs(source_angle - self.orientation_rad)
        
        # Normalize angle difference to [0, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        angle_diff = abs(angle_diff)
        
        # Calculate hearing factor (1.0 when source is directly in front,
        # decreasing as source moves to sides/back)
        # The formula maps [0, pi] to [1.0, 1.0 - direction_sensitivity]
        hearing_factor = 1.0 - (angle_diff / math.pi) * self.direction_sensitivity
        
        return hearing_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert player to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the player
        """
        return {
            "position": self.position,
            "orientation": self.orientation,
            "speed": self.speed,
            "rotation_speed": self.rotation_speed,
            "hearing_sensitivity": self.hearing_sensitivity,
            "direction_sensitivity": self.direction_sensitivity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Player':
        """
        Create a player from dictionary data.
        
        Args:
            data: Dictionary with player properties
            
        Returns:
            Player instance
        """
        player = cls(
            position=tuple(data["position"]),
            orientation=data.get("orientation", 0)
        )
        
        # Set optional properties if present
        if "speed" in data:
            player.speed = data["speed"]
        if "rotation_speed" in data:
            player.rotation_speed = data["rotation_speed"]
        if "hearing_sensitivity" in data:
            player.hearing_sensitivity = data["hearing_sensitivity"]
        if "direction_sensitivity" in data:
            player.direction_sensitivity = data["direction_sensitivity"]
            
        return player 