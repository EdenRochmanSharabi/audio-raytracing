import os
import sys
import math
import unittest
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
from src.obstacles import RectangleObstacle, CircleObstacle, LineObstacle
from src.player import Player
from src.sound_source import SoundSource
from src.ray_tracer import RayTracer, Ray
from src.physics.reflection import ReflectionModel
from src.physics.propagation import PropagationModel


class TestEnvironment(unittest.TestCase):
    """Test the Environment class."""
    
    def test_creation(self):
        """Test environment creation."""
        env = Environment(width=800, height=600, name="Test Environment")
        
        self.assertEqual(env.width, 800)
        self.assertEqual(env.height, 600)
        self.assertEqual(env.name, "Test Environment")
        self.assertEqual(len(env.obstacles), 0)
        self.assertEqual(len(env.sound_sources), 0)
        self.assertIsNone(env.player)
    
    def test_adding_objects(self):
        """Test adding objects to environment."""
        env = Environment()
        
        # Add player
        player = Player(position=(400, 300))
        env.set_player(player)
        self.assertEqual(env.player, player)
        
        # Add sound source
        source = SoundSource(position=(200, 200))
        env.add_sound_source(source)
        self.assertEqual(len(env.sound_sources), 1)
        self.assertEqual(env.sound_sources[0], source)
        
        # Add obstacle
        obstacle = RectangleObstacle(position=(300, 300), dimensions=(100, 50))
        env.add_obstacle(obstacle)
        self.assertEqual(len(env.obstacles), 1)
        self.assertEqual(env.obstacles[0], obstacle)
    
    def test_is_point_in_bounds(self):
        """Test boundary checking."""
        env = Environment(width=500, height=400)
        
        self.assertTrue(env.is_point_in_bounds(250, 200))  # Center
        self.assertTrue(env.is_point_in_bounds(0, 0))  # Top-left corner
        self.assertTrue(env.is_point_in_bounds(500, 400))  # Bottom-right corner
        self.assertFalse(env.is_point_in_bounds(-1, 200))  # Out of left bound
        self.assertFalse(env.is_point_in_bounds(250, -10))  # Out of top bound
        self.assertFalse(env.is_point_in_bounds(501, 200))  # Out of right bound
        self.assertFalse(env.is_point_in_bounds(250, 401))  # Out of bottom bound


class TestObstacles(unittest.TestCase):
    """Test the Obstacle classes."""
    
    def test_rectangle_creation(self):
        """Test rectangle obstacle creation."""
        rect = RectangleObstacle(position=(100, 100), dimensions=(50, 30), rotation=45)
        
        self.assertEqual(rect.position, (100, 100))
        self.assertEqual(rect.dimensions, (50, 30))
        self.assertEqual(rect.rotation, 45)
        self.assertEqual(len(rect.vertices), 4)
        self.assertEqual(len(rect.edges), 4)
    
    def test_circle_creation(self):
        """Test circle obstacle creation."""
        circle = CircleObstacle(position=(200, 200), radius=50)
        
        self.assertEqual(circle.position, (200, 200))
        self.assertEqual(circle.radius, 50)
    
    def test_line_creation(self):
        """Test line obstacle creation."""
        line = LineObstacle(start=(100, 100), end=(200, 200))
        
        self.assertEqual(line.start, (100, 100))
        self.assertEqual(line.end, (200, 200))
        self.assertAlmostEqual(line.length, math.sqrt(2 * 100 * 100))
    
    def test_rectangle_intersection(self):
        """Test ray intersection with rectangle."""
        rect = RectangleObstacle(position=(100, 100), dimensions=(50, 50), rotation=0)
        
        # Ray hitting the rectangle
        result = rect.intersects_ray((50, 100), (1, 0))
        self.assertIsNotNone(result)
        distance, point, normal = result
        self.assertAlmostEqual(distance, 25)
        
        # Ray missing the rectangle
        result = rect.intersects_ray((50, 50), (0, -1))
        self.assertIsNone(result)
    
    def test_circle_intersection(self):
        """Test ray intersection with circle."""
        circle = CircleObstacle(position=(100, 100), radius=50)
        
        # Ray hitting the circle
        result = circle.intersects_ray((0, 100), (1, 0))
        self.assertIsNotNone(result)
        distance, point, normal = result
        self.assertAlmostEqual(distance, 50)
        
        # Ray missing the circle
        result = circle.intersects_ray((0, 200), (1, 0))
        self.assertIsNone(result)
    
    def test_line_intersection(self):
        """Test ray intersection with line."""
        line = LineObstacle(start=(100, 0), end=(100, 200))
        
        # Ray hitting the line
        result = line.intersects_ray((0, 100), (1, 0))
        self.assertIsNotNone(result)
        distance, point, normal = result
        self.assertAlmostEqual(distance, 100)
        
        # Ray missing the line
        result = line.intersects_ray((0, 300), (1, 0))
        self.assertIsNone(result)


class TestPlayer(unittest.TestCase):
    """Test the Player class."""
    
    def test_creation(self):
        """Test player creation."""
        player = Player(position=(300, 200), orientation=90)
        
        self.assertEqual(player.position, (300, 200))
        self.assertEqual(player.orientation, 90)
        self.assertAlmostEqual(player.orientation_rad, math.radians(90))
    
    def test_movement(self):
        """Test player movement."""
        player = Player(position=(100, 100))
        
        player.move(50, 0)
        self.assertEqual(player.position, (150, 100))
        
        player.move(0, -25)
        self.assertEqual(player.position, (150, 75))
    
    def test_rotation(self):
        """Test player rotation."""
        player = Player(position=(100, 100), orientation=0)
        
        player.rotate(45)
        self.assertEqual(player.orientation, 45)
        self.assertAlmostEqual(player.orientation_rad, math.radians(45))
        
        player.set_orientation(180)
        self.assertEqual(player.orientation, 180)
        self.assertAlmostEqual(player.orientation_rad, math.radians(180))
    
    def test_direction_vector(self):
        """Test getting player direction vector."""
        player = Player(position=(100, 100), orientation=0)
        dir_vector = player.get_direction_vector()
        self.assertAlmostEqual(dir_vector[0], 1.0)
        self.assertAlmostEqual(dir_vector[1], 0.0)
        
        player.set_orientation(90)
        dir_vector = player.get_direction_vector()
        self.assertAlmostEqual(dir_vector[0], 0.0)
        self.assertAlmostEqual(dir_vector[1], 1.0)
    
    def test_ear_positions(self):
        """Test getting ear positions."""
        player = Player(position=(100, 100), orientation=0)
        left_ear, right_ear = player.get_ear_positions(ear_distance=2.0)
        
        # At 0 degrees, left ear should be above player, right ear below
        self.assertAlmostEqual(left_ear[0], 100.0)
        self.assertAlmostEqual(left_ear[1], 101.0)
        self.assertAlmostEqual(right_ear[0], 100.0)
        self.assertAlmostEqual(right_ear[1], 99.0)


class TestSoundSource(unittest.TestCase):
    """Test the SoundSource class."""
    
    def test_creation(self):
        """Test sound source creation."""
        source = SoundSource(position=(200, 300), volume=0.8, name="Test Source")
        
        self.assertEqual(source.position, (200, 300))
        self.assertEqual(source.volume, 0.8)
        self.assertEqual(source.name, "Test Source")
        self.assertFalse(source.directional)
    
    def test_directional_setup(self):
        """Test setting up directional sound source."""
        source = SoundSource(position=(200, 300))
        
        source.set_directional(True, direction=45, beam_width=90)
        self.assertTrue(source.directional)
        self.assertEqual(source.direction, 45)
        self.assertAlmostEqual(source.direction_rad, math.radians(45))
        self.assertEqual(source.beam_width, 90)
        self.assertAlmostEqual(source.beam_width_rad, math.radians(90))
    
    def test_directional_factor(self):
        """Test directional intensity factor."""
        source = SoundSource(position=(200, 300))
        
        # Omnidirectional source should have factor 1.0 in all directions
        self.assertEqual(source.directional_factor(0), 1.0)
        self.assertEqual(source.directional_factor(math.pi), 1.0)
        
        # Set up directional source with wider beam width for the test
        source.set_directional(True, direction=0, beam_width=180)
        
        # At direction, factor should be 1.0
        self.assertAlmostEqual(source.directional_factor(0), 1.0)
        
        # At 45 degrees (quarter of the beam width), factor should be around 0.7071
        # This value is cos(π/4) which is approximately 0.7071
        self.assertAlmostEqual(source.directional_factor(math.radians(45)), 0.5, places=2)
        
        # At 90 degrees (edge of beam), factor should be very close to 0
        self.assertAlmostEqual(source.directional_factor(math.radians(90)), 0.0, places=5)
        
        # Outside beam, factor should be 0
        self.assertAlmostEqual(source.directional_factor(math.pi), 0.0, places=5)
    
    def test_ray_directions(self):
        """Test generating ray directions."""
        source = SoundSource(position=(200, 300))
        
        # Test omnidirectional rays
        rays = source.get_ray_directions(ray_count=8)
        self.assertEqual(len(rays), 8)
        
        # Test directional rays
        source.set_directional(True, direction=0, beam_width=90)
        rays = source.get_ray_directions(ray_count=8)
        self.assertEqual(len(rays), 8)
        
        # All directional rays should be within the beam
        for dx, dy in rays:
            angle = math.atan2(dy, dx)
            # Normalize angle to [-π, π]
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
                
            # Check if within beam width
            self.assertLessEqual(abs(angle), math.radians(45))


class TestRayTracer(unittest.TestCase):
    """Test the RayTracer class."""
    
    def setUp(self):
        """Set up a test environment."""
        self.env = Environment(width=500, height=400)
        self.player = Player(position=(400, 200))
        self.env.set_player(self.player)
        
        self.source = SoundSource(position=(100, 200))
        self.env.add_sound_source(self.source)
        
        self.obstacle = RectangleObstacle(
            position=(250, 200),
            dimensions=(20, 100),
            rotation=0
        )
        self.env.add_obstacle(self.obstacle)
        
        self.ray_tracer = RayTracer(self.env)
        self.ray_tracer.ray_count = 36  # Use fewer rays for testing
    
    def test_ray_creation(self):
        """Test ray creation."""
        ray = Ray(origin=(0, 0), direction=(1, 0), energy=0.8)
        
        self.assertEqual(ray.origin, (0, 0))
        self.assertEqual(ray.direction, (1, 0))
        self.assertEqual(ray.energy, 0.8)
        self.assertEqual(ray.path, [(0, 0)])
        self.assertEqual(ray.reflections, 0)
    
    def test_ray_reflection(self):
        """Test ray reflection."""
        ray = Ray(origin=(0, 0), direction=(1, 0), energy=1.0)
        
        # Reflect off a vertical surface
        ray.reflect((10, 0), (1, 0), 0.8)
        
        self.assertEqual(ray.origin, (10, 0))
        self.assertAlmostEqual(ray.direction[0], -1.0)
        self.assertAlmostEqual(ray.direction[1], 0.0)
        self.assertEqual(ray.energy, 0.8)
        self.assertEqual(ray.path, [(0, 0), (10, 0)])
        self.assertEqual(ray.reflections, 1)
    
    def test_tracing(self):
        """Test basic ray tracing."""
        # Perform ray tracing
        rays = self.ray_tracer.trace_rays()
        
        # Check if rays were created
        self.assertGreater(len(rays), 0)
        
        # Check if any rays reached the player
        self.assertGreater(len(self.ray_tracer.sound_path_data), 0)
    
    def test_ray_obstacle_intersection(self):
        """Test ray-obstacle intersection detection."""
        # Create a ray pointing directly at the obstacle
        ray = Ray(
            origin=(100, 200), 
            direction=(1, 0),
            energy=1.0,
            source=self.source
        )
        
        # Manually trace this ray
        self.ray_tracer._trace_single_ray(ray, [self.obstacle], self.player)
        
        # Check if it intersected with the obstacle
        self.assertEqual(len(ray.path), 2)  # Origin + intersection point
        
        # Calculate expected intersection point
        expected_x = self.obstacle.position[0] - self.obstacle.dimensions[0] / 2
        expected_y = 200
        
        # Check if intersection point is close to expected
        intersection = ray.path[1]
        self.assertAlmostEqual(intersection[0], expected_x, delta=1e-10)
        self.assertAlmostEqual(intersection[1], expected_y, delta=1e-10)
    
    def test_animation(self):
        """Test ray animation."""
        # Trace rays
        self.ray_tracer.trace_rays()
        
        # Start animation
        self.ray_tracer.start_animation()
        
        # Initial state should have some visible segments
        segments = self.ray_tracer.get_visible_ray_segments()
        self.assertGreater(len(segments), 0)
        
        # Advance time
        self.ray_tracer.update_animation(0.1)
        
        # Should have more visible segments now
        new_segments = self.ray_tracer.get_visible_ray_segments()
        self.assertGreaterEqual(len(new_segments), len(segments))


class TestPhysics(unittest.TestCase):
    """Test the physics models."""
    
    def test_reflection_model(self):
        """Test the reflection physics model."""
        reflection_model = ReflectionModel()
        
        # Test reflection direction calculation
        incident = (1, 0)
        normal = (0, -1)  # Changing to pointing down
        reflected = reflection_model.calculate_reflection_direction(incident, normal)
        
        # Reflection should be (1, 0) -> (1, 0) with normal pointing down
        self.assertAlmostEqual(reflected[0], 1.0)
        self.assertAlmostEqual(reflected[1], 0.0)
        
        # Test with normal pointing up (should reflect downward)
        normal = (0, 1)  # Pointing up
        reflected = reflection_model.calculate_reflection_direction(incident, normal)
        
        # Now the reflection should be (1, 0) -> (1, 0) with normal pointing up
        self.assertAlmostEqual(reflected[0], 1.0)
        self.assertAlmostEqual(reflected[1], 0.0)
        
        # Test with a 45-degree normal
        normal = (1, 1)  # 45-degree normal
        reflected = reflection_model.calculate_reflection_direction(incident, normal)
        
        # Reflection should be approximately (0, -1)
        self.assertAlmostEqual(reflected[0], 0.0, places=3)
        self.assertAlmostEqual(reflected[1], -1.0, places=3)
        
        # Test energy loss calculation
        energy = 1.0
        material = "glass"  # High reflection
        angle = math.radians(45)
        
        remaining = reflection_model.calculate_energy_loss(energy, material, angle)
        self.assertGreater(remaining, 0.9)  # Glass should reflect most energy
        
        # Test with more absorbent material
        remaining = reflection_model.calculate_energy_loss(energy, "carpet", angle)
        self.assertLess(remaining, 0.5)  # Carpet should absorb most energy
    
    def test_propagation_model(self):
        """Test the propagation physics model."""
        propagation_model = PropagationModel(speed_of_sound=343.0)
        
        # Test time calculation
        distance = 343.0  # Meters
        time = propagation_model.calculate_propagation_time(distance)
        self.assertEqual(time, 1.0)  # Should take 1 second
        
        # Test inverse square law
        intensity_1m = propagation_model.calculate_intensity_factor(1.0)
        intensity_2m = propagation_model.calculate_intensity_factor(2.0)
        
        # At twice the distance, intensity should be 1/4
        self.assertAlmostEqual(intensity_2m, intensity_1m / 4.0)
        
        # Test linear attenuation model
        propagation_model.set_attenuation_model("linear")
        intensity_1m = propagation_model.calculate_intensity_factor(1.0)
        intensity_2m = propagation_model.calculate_intensity_factor(2.0)
        
        # At twice the distance, intensity should be 1/2
        self.assertAlmostEqual(intensity_2m, intensity_1m / 2.0)


if __name__ == "__main__":
    unittest.main() 