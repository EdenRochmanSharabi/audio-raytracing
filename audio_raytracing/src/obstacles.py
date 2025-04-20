import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import math


class Obstacle:
    """
    Base class for obstacles in the audio ray tracing environment.
    
    Obstacles are objects that sound rays can interact with through
    reflection and absorption. Each obstacle has a position, shape,
    and acoustic properties.
    """
    
    def __init__(self, position: Tuple[float, float], material: str = "wall"):
        """
        Initialize an obstacle.
        
        Args:
            position: (x, y) coordinates of the obstacle's center
            material: Material type affecting acoustic properties
        """
        self.position = position
        self.material = material
    
    def intersects_ray(self, ray_origin: Tuple[float, float], ray_dir: Tuple[float, float]) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
        """
        Check if a ray intersects this obstacle.
        
        Args:
            ray_origin: (x, y) origin point of the ray
            ray_dir: (dx, dy) direction vector of the ray
            
        Returns:
            If intersection occurs, returns (distance, intersection_point, normal_vector)
            If no intersection, returns None
        """
        # Must be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert obstacle to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the obstacle
        """
        # Basic properties all obstacles have
        return {
            "type": self.__class__.__name__.lower(),
            "position": self.position,
            "material": self.material
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Obstacle':
        """
        Create an obstacle from dictionary data.
        
        Args:
            data: Dictionary with obstacle properties
            
        Returns:
            Appropriate Obstacle subclass instance
        """
        obstacle_type = data.get("type", "rectangle").lower()
        
        if obstacle_type == "rectangle":
            return RectangleObstacle(
                position=tuple(data["position"]),
                dimensions=tuple(data["dimensions"]),
                rotation=data.get("rotation", 0),
                material=data.get("material", "wall")
            )
        elif obstacle_type == "circle":
            return CircleObstacle(
                position=tuple(data["position"]),
                radius=data["radius"],
                material=data.get("material", "wall")
            )
        elif obstacle_type == "line":
            return LineObstacle(
                start=tuple(data["start"]),
                end=tuple(data["end"]),
                material=data.get("material", "wall")
            )
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle_type}")


class RectangleObstacle(Obstacle):
    """Rectangular obstacle with width, height and rotation."""
    
    def __init__(self, position: Tuple[float, float], dimensions: Tuple[float, float], 
                 rotation: float = 0, material: str = "wall"):
        """
        Initialize a rectangular obstacle.
        
        Args:
            position: (x, y) center of the rectangle
            dimensions: (width, height) of the rectangle
            rotation: Rotation angle in degrees
            material: Material type affecting acoustic properties
        """
        super().__init__(position, material)
        self.dimensions = dimensions
        self.rotation = rotation  # in degrees
        self.rotation_rad = math.radians(rotation)
        
        # Precompute the rectangle vertices
        self._update_vertices()
    
    def _update_vertices(self):
        """Update the rectangle vertices based on position, dimensions and rotation."""
        w, h = self.dimensions
        x, y = self.position
        
        # Calculate corners relative to center (before rotation)
        half_w, half_h = w/2, h/2
        corners = [
            (-half_w, -half_h),  # Top-left
            (half_w, -half_h),   # Top-right
            (half_w, half_h),    # Bottom-right
            (-half_w, half_h)    # Bottom-left
        ]
        
        # Apply rotation and translate to absolute position
        cos_rot = math.cos(self.rotation_rad)
        sin_rot = math.sin(self.rotation_rad)
        
        self.vertices = []
        for dx, dy in corners:
            # Rotate
            rx = dx * cos_rot - dy * sin_rot
            ry = dx * sin_rot + dy * cos_rot
            # Translate
            self.vertices.append((x + rx, y + ry))
        
        # Create line segments (edges) from vertices
        self.edges = []
        for i in range(4):
            self.edges.append((
                self.vertices[i],
                self.vertices[(i+1) % 4]
            ))
    
    def intersects_ray(self, ray_origin: Tuple[float, float], ray_dir: Tuple[float, float]) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
        """
        Check if a ray intersects this rectangle by checking intersection with each edge.
        
        Args:
            ray_origin: (x, y) origin point of the ray
            ray_dir: (dx, dy) direction vector of the ray
            
        Returns:
            If intersection occurs, returns (distance, intersection_point, normal_vector)
            If no intersection, returns None
        """
        closest_distance = float('inf')
        closest_intersection = None
        closest_normal = None
        
        # Check intersection with each edge
        for i, ((x1, y1), (x2, y2)) in enumerate(self.edges):
            # Line segment ray intersection
            x3, y3 = ray_origin
            x4, y4 = ray_origin[0] + ray_dir[0], ray_origin[1] + ray_dir[1]
            
            # Calculate determinant
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            # If lines are parallel
            if den == 0:
                continue
            
            # Calculate intersection parameters
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
            
            # Check if intersection occurs within the line segment and along the ray direction
            if 0 <= t <= 1 and u >= 0:
                # Calculate intersection point
                intersection_x = x1 + t * (x2 - x1)
                intersection_y = y1 + t * (y2 - y1)
                intersection = (intersection_x, intersection_y)
                
                # Calculate distance from ray origin to intersection
                dx = intersection_x - ray_origin[0]
                dy = intersection_y - ray_origin[1]
                distance = math.sqrt(dx * dx + dy * dy)
                
                # Calculate normal vector (perpendicular to the edge, pointing outward)
                edge_dx = x2 - x1
                edge_dy = y2 - y1
                normal = (-edge_dy, edge_dx)  # Rotate 90 degrees
                
                # Normalize the normal vector
                normal_magnitude = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1])
                if normal_magnitude > 0:
                    normal = (normal[0] / normal_magnitude, normal[1] / normal_magnitude)
                
                # Ensure normal points away from the ray
                dot_product = normal[0] * ray_dir[0] + normal[1] * ray_dir[1]
                if dot_product > 0:
                    normal = (-normal[0], -normal[1])
                
                # Keep track of the closest intersection
                if distance < closest_distance:
                    closest_distance = distance
                    closest_intersection = intersection
                    closest_normal = normal
        
        if closest_intersection:
            return (closest_distance, closest_intersection, closest_normal)
        
        return None
    
    def set_rotation(self, angle: float) -> None:
        """
        Set the rotation angle of the rectangle.
        
        Args:
            angle: Rotation angle in degrees
        """
        self.rotation = angle
        self.rotation_rad = math.radians(angle)
        self._update_vertices()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert rectangle to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the rectangle
        """
        base_dict = super().to_dict()
        base_dict.update({
            "dimensions": self.dimensions,
            "rotation": self.rotation
        })
        return base_dict


class CircleObstacle(Obstacle):
    """Circular obstacle with a radius."""
    
    def __init__(self, position: Tuple[float, float], radius: float, material: str = "wall"):
        """
        Initialize a circular obstacle.
        
        Args:
            position: (x, y) center of the circle
            radius: Radius of the circle
            material: Material type affecting acoustic properties
        """
        super().__init__(position, material)
        self.radius = radius
    
    def intersects_ray(self, ray_origin: Tuple[float, float], ray_dir: Tuple[float, float]) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
        """
        Check if a ray intersects this circle using the quadratic formula.
        
        Args:
            ray_origin: (x, y) origin point of the ray
            ray_dir: (dx, dy) direction vector of the ray
            
        Returns:
            If intersection occurs, returns (distance, intersection_point, normal_vector)
            If no intersection, returns None
        """
        # Normalize direction vector
        dir_magnitude = math.sqrt(ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1])
        if dir_magnitude == 0:
            return None
        
        normalized_dir = (ray_dir[0] / dir_magnitude, ray_dir[1] / dir_magnitude)
        
        # Vector from ray origin to circle center
        oc = (ray_origin[0] - self.position[0], ray_origin[1] - self.position[1])
        
        # Quadratic formula coefficients
        a = 1  # Since normalized_dir is normalized
        b = 2 * (oc[0] * normalized_dir[0] + oc[1] * normalized_dir[1])
        c = oc[0] * oc[0] + oc[1] * oc[1] - self.radius * self.radius
        
        # Calculate discriminant
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            # No intersection
            return None
        
        # Find the closest intersection point
        t1 = (-b - math.sqrt(discriminant)) / (2 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)
        
        # Take the smallest non-negative value of t
        t = None
        if t1 >= 0:
            t = t1
        elif t2 >= 0:
            t = t2
        
        if t is None:
            # No intersection in the positive direction
            return None
        
        # Calculate intersection point
        intersection = (
            ray_origin[0] + t * normalized_dir[0],
            ray_origin[1] + t * normalized_dir[1]
        )
        
        # Calculate normal vector (pointing outward from circle center)
        normal = (
            (intersection[0] - self.position[0]) / self.radius,
            (intersection[1] - self.position[1]) / self.radius
        )
        
        # Calculate distance from ray origin to intersection
        distance = t * dir_magnitude
        
        return (distance, intersection, normal)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert circle to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the circle
        """
        base_dict = super().to_dict()
        base_dict.update({
            "radius": self.radius
        })
        return base_dict


class LineObstacle(Obstacle):
    """Line segment obstacle with start and end points."""
    
    def __init__(self, start: Tuple[float, float], end: Tuple[float, float], material: str = "wall"):
        """
        Initialize a line segment obstacle.
        
        Args:
            start: (x, y) coordinates of line start
            end: (x, y) coordinates of line end
            material: Material type affecting acoustic properties
        """
        # Calculate center point for the parent class
        position = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2
        )
        super().__init__(position, material)
        
        self.start = start
        self.end = end
        
        # Calculate line properties
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        self.length = math.sqrt(dx*dx + dy*dy)
        
        # Pre-compute normal vector (perpendicular to line)
        if self.length > 0:
            self.normal = (-dy / self.length, dx / self.length)
        else:
            self.normal = (0, 1)  # Default for zero-length line
    
    def intersects_ray(self, ray_origin: Tuple[float, float], ray_dir: Tuple[float, float]) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
        """
        Check if a ray intersects this line segment.
        
        Args:
            ray_origin: (x, y) origin point of the ray
            ray_dir: (dx, dy) direction vector of the ray
            
        Returns:
            If intersection occurs, returns (distance, intersection_point, normal_vector)
            If no intersection, returns None
        """
        x1, y1 = self.start
        x2, y2 = self.end
        x3, y3 = ray_origin
        x4, y4 = ray_origin[0] + ray_dir[0], ray_origin[1] + ray_dir[1]
        
        # Calculate determinant
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # If lines are parallel
        if den == 0:
            return None
        
        # Calculate intersection parameters
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        
        # Check if intersection occurs within the line segment and along the ray direction
        if 0 <= t <= 1 and u >= 0:
            # Calculate intersection point
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            intersection = (intersection_x, intersection_y)
            
            # Calculate distance from ray origin to intersection
            dx = intersection_x - ray_origin[0]
            dy = intersection_y - ray_origin[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Ensure normal points away from the ray
            normal = self.normal
            dot_product = normal[0] * ray_dir[0] + normal[1] * ray_dir[1]
            if dot_product > 0:
                normal = (-normal[0], -normal[1])
            
            return (distance, intersection, normal)
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert line to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the line
        """
        base_dict = super().to_dict()
        base_dict.update({
            "start": self.start,
            "end": self.end
        })
        return base_dict 