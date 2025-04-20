import math
import numpy as np
from typing import Dict, Tuple, List, Optional


class ReflectionModel:
    """
    Implements the physics model for sound reflection and absorption.
    
    This class handles the calculation of reflection angles and energy loss
    when sound rays interact with surfaces of different materials.
    """
    
    def __init__(self):
        """Initialize the reflection model with default properties."""
        # Default material properties (reflection and absorption coefficients)
        self.default_materials = {
            "wall": {"reflection": 0.9, "absorption": 0.1},
            "wood": {"reflection": 0.8, "absorption": 0.2},
            "glass": {"reflection": 0.95, "absorption": 0.05},
            "carpet": {"reflection": 0.4, "absorption": 0.6},
            "concrete": {"reflection": 0.97, "absorption": 0.03}
        }
        
        # Frequency-dependent absorption coefficients (optional)
        self.frequency_absorption = {
            "wall": {125: 0.05, 250: 0.07, 500: 0.09, 1000: 0.12, 2000: 0.15, 4000: 0.18},
            "wood": {125: 0.15, 250: 0.17, 500: 0.20, 1000: 0.22, 2000: 0.25, 4000: 0.30},
            "glass": {125: 0.02, 250: 0.03, 500: 0.04, 1000: 0.05, 2000: 0.06, 4000: 0.07},
            "carpet": {125: 0.10, 250: 0.25, 500: 0.45, 1000: 0.65, 2000: 0.75, 4000: 0.80},
            "concrete": {125: 0.01, 250: 0.01, 500: 0.02, 1000: 0.02, 2000: 0.03, 4000: 0.04}
        }
    
    def get_material_properties(self, material: str) -> Dict[str, float]:
        """
        Get the reflection and absorption properties for a material.
        
        Args:
            material: Name of the material
            
        Returns:
            Dictionary with reflection and absorption coefficients
        """
        return self.default_materials.get(material, self.default_materials["wall"])
    
    def get_frequency_absorption(self, material: str, frequency: float) -> float:
        """
        Get the absorption coefficient for a material at a specific frequency.
        
        Args:
            material: Name of the material
            frequency: Sound frequency in Hz
            
        Returns:
            Absorption coefficient (0.0 to 1.0)
        """
        # If material not found, use wall as default
        if material not in self.frequency_absorption:
            material = "wall"
            
        material_freq_data = self.frequency_absorption[material]
        
        # Find the nearest frequency bands
        frequencies = sorted(material_freq_data.keys())
        
        if frequency <= frequencies[0]:
            return material_freq_data[frequencies[0]]
        elif frequency >= frequencies[-1]:
            return material_freq_data[frequencies[-1]]
        
        # Interpolate between the two nearest frequency bands
        for i in range(len(frequencies) - 1):
            if frequencies[i] <= frequency <= frequencies[i+1]:
                f1, f2 = frequencies[i], frequencies[i+1]
                a1, a2 = material_freq_data[f1], material_freq_data[f2]
                # Linear interpolation
                t = (frequency - f1) / (f2 - f1)
                return a1 + t * (a2 - a1)
        
        # Fallback (should not reach here)
        return self.default_materials[material]["absorption"]
    
    def calculate_reflection_direction(self, incident_dir: Tuple[float, float], 
                                       normal: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate the reflection direction using the law of reflection.
        
        The law of reflection states that the angle of incidence equals
        the angle of reflection, with respect to the surface normal.
        
        Args:
            incident_dir: (dx, dy) normalized incident direction vector
            normal: (nx, ny) normalized surface normal vector
            
        Returns:
            (rx, ry) normalized reflection direction vector
        """
        # Calculate reflection using the formula: r = d - 2(d·n)n
        # where d is incident direction, n is surface normal
        # Ensure vectors are normalized
        d_mag = math.sqrt(incident_dir[0]**2 + incident_dir[1]**2)
        n_mag = math.sqrt(normal[0]**2 + normal[1]**2)
        
        if d_mag == 0 or n_mag == 0:
            return (-incident_dir[0], -incident_dir[1])  # Just reverse direction as fallback
            
        d = (incident_dir[0] / d_mag, incident_dir[1] / d_mag)
        n = (normal[0] / n_mag, normal[1] / n_mag)
        
        # Calculate dot product
        dot_product = d[0] * n[0] + d[1] * n[1]
        
        # Calculate reflection vector
        reflection_dir = (
            d[0] - 2 * dot_product * n[0],
            d[1] - 2 * dot_product * n[1]
        )
        
        # Normalize the reflection direction
        mag = math.sqrt(reflection_dir[0]**2 + reflection_dir[1]**2)
        if mag > 0:
            reflection_dir = (
                reflection_dir[0] / mag,
                reflection_dir[1] / mag
            )
        
        return reflection_dir
    
    def calculate_energy_loss(self, energy: float, material: str, angle_of_incidence: float,
                             frequency: Optional[float] = None) -> float:
        """
        Calculate the energy loss during reflection based on material properties.
        
        Args:
            energy: Incoming ray energy (0.0 to 1.0)
            material: Material of the reflecting surface
            angle_of_incidence: Angle between ray and surface normal (radians)
            frequency: Sound frequency in Hz (for frequency-dependent absorption)
            
        Returns:
            Remaining energy after reflection (0.0 to 1.0)
        """
        material_props = self.get_material_properties(material)
        reflection_coef = material_props["reflection"]
        
        # Reflection coefficient varies with angle of incidence
        # At grazing angles (close to 90°), reflection is generally higher
        angle_factor = 1.0
        if angle_of_incidence is not None:
            # Convert to degrees for easier understanding
            angle_deg = math.degrees(abs(angle_of_incidence))
            
            # Increase reflection as angle approaches 90 degrees (grazing)
            # This is a simplified model; real-world behavior is more complex
            if angle_deg > 45:
                # Linear increase from 1.0 at 45° to 1.2 at 90°
                angle_factor = 1.0 + (angle_deg - 45) / 45 * 0.2
        
        # Apply frequency-dependent absorption if frequency is provided
        if frequency is not None:
            absorption_coef = self.get_frequency_absorption(material, frequency)
            # Combine with standard absorption
            absorption_coef = (absorption_coef + material_props["absorption"]) / 2
            reflection_coef = 1.0 - absorption_coef
        
        # Calculate remaining energy
        remaining_energy = energy * reflection_coef * angle_factor
        
        # Ensure energy remains in valid range
        return max(0.0, min(1.0, remaining_energy))
    
    def add_material(self, name: str, reflection: float, absorption: float, 
                    frequency_data: Optional[Dict[int, float]] = None) -> None:
        """
        Add a new material to the model.
        
        Args:
            name: Material name
            reflection: Reflection coefficient (0.0 to 1.0)
            absorption: Absorption coefficient (0.0 to 1.0)
            frequency_data: Optional dictionary mapping frequencies to absorption values
        """
        # Normalize coefficients if they sum to more than 1
        if reflection + absorption > 1.0:
            total = reflection + absorption
            reflection /= total
            absorption /= total
            
        self.default_materials[name] = {
            "reflection": reflection,
            "absorption": absorption
        }
        
        # Add frequency-dependent data if provided
        if frequency_data:
            self.frequency_absorption[name] = frequency_data 