# Audio Ray Tracing Simulation

A 2D proof-of-concept for simulating sound propagation through ray tracing techniques, enabling realistic spatial audio based on physical principles.

## Overview

This project demonstrates audio ray tracing - a technique for simulating how sound waves travel through an environment, bounce off obstacles, and reach your ears. Unlike typical game audio that simply fades sounds based on distance, this simulation creates a more immersive experience by calculating actual sound paths through the environment.

![Ray tracing visualization showing sound paths](screenshots/ray_paths.png)

## How Sound Travels in Our Simulation

When you listen to sounds in the real world, you're actually hearing sound waves that have traveled along various paths to reach your ears. Our simulation recreates this natural phenomenon:

1. **Sound paths matter**: Sound doesn't just travel in straight lines - it bounces off walls and objects before reaching you
2. **Distance affects volume**: The further you are from a sound source, the quieter it becomes
3. **Materials change sound**: Different surfaces (wood, concrete, glass) absorb or reflect sound differently
4. **Sound takes time**: You hear distant sounds slightly later than close ones because sound needs time to travel

The simulation traces rays from your position (the listener) outward in all directions. When these rays hit a sound source, they bounce back toward you. If they successfully return to your position, you'll hear that sound with the appropriate volume, delay, and spatial characteristics.

![Sound rays emanating from player position](screenshots/player_rays.png)

This approach creates truly spatial audio - sounds coming from your left actually sound like they're coming from your left, and sounds that bounce off multiple surfaces create a natural sense of reverberation and space.

## Features

- Real-time visualization of sound rays and how they travel
- Two-way path tracing for accurate sound perception
- Realistic sound reflection and absorption based on physics
- Spatial audio mixing using the traced sound paths
- Customizable environment with different obstacle types
- Interactive movement through the sound environment

![Multiple reflection paths showing complex sound behavior](screenshots/reflections.png)

## Getting Started

### Prerequisites

- Python 3.10 or 3.12 (Note: Python 3.13 is not currently supported due to the removal of the `audioop` module)
- Dependencies listed in `requirements.txt`

### Installation

1. Clone this repository
```bash
git clone https://github.com/EdenRochman/audio-raytracing.git
cd audio-raytracing
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r audio_raytracing/requirements.txt
```

### Running the Simulation

For visual-only mode (no audio):
```bash
cd audio_raytracing
python main_visual.py
```

For full version with audio:
```bash
cd audio_raytracing
python main.py
```

Command line options:
- `--rays N`: Sets the number of rays to cast (default: 360)
- `--reflection-limit N`: Maximum number of reflections per ray (default: 3)
- `--animation-speed S`: Controls the animation speed (default: 1.0)
- `--mute`: Start with audio muted
- `--volume V`: Set audio volume (0.0 to 1.0)

### Controls

- **Arrow keys**: Move the player/listener
- **WASD**: Move the nearest sound source
- **Q/E**: Rotate player
- **Space**: Pause/Resume
- **R**: Reset the simulation
- **+/-**: Adjust animation speed
- **H**: Toggle help overlay
- **F1**: Toggle stats display
- **M**: Toggle audio mute
- **Esc**: Quit the simulation

## Project Structure

The core components are organized as follows:

- `src/`: Core simulation code
  - `environment.py`: Environment and simulation space
  - `obstacles.py`: Various obstacle types and collision detection
  - `player.py`: Player/listener position and orientation
  - `sound_source.py`: Sound source characteristics
  - `ray_tracer.py`: Ray tracing logic
  - `physics/`: Physical models for sound
  - `audio/`: Audio processing and playback
- `tests/`: Unit tests for core components
- `assets/`: Audio files and resources
- `examples/`: Example environments and demonstrations

## Recent Improvements

- **Two-way Ray Tracing**: Rays now start from your position and bounce off sound sources back to you
- **Direct Sound Paths**: Added calculations for direct paths from sources to player for continuous audio
- **Better Spatial Audio**: Improved how sounds are positioned in space with better left/right balance
- **Increased Range**: Reduced minimum energy threshold so sound can travel further

## Future Plans

- 3D sound propagation
- Sound bending around corners for low frequencies
- Real-time audio synthesis
- Custom sound design tools
- Performance improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project draws inspiration from both game audio and architectural acoustics
- Thanks to everyone who contributed ideas and feedback 