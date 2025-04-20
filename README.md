# Audio Ray Tracing Simulation

A 2D proof-of-concept for simulating sound propagation through ray tracing techniques, enabling realistic spatial audio based on physical principles.

## Overview

This project demonstrates audio ray tracing - a technique for simulating how sound waves propagate through an environment, interact with obstacles, and reach a listener. Unlike traditional audio engines that use simple distance-based models, this simulation accounts for sound reflection, absorption, and path tracing for more realistic spatial audio experiences.

## Scientific Background

### Sound Propagation Physics

Sound waves propagate through a medium (typically air) as pressure oscillations. In this simulation, we model sound propagation using a geometric ray-based approach, which is an approximation that works well at frequencies where the wavelength is small compared to the obstacles in the environment.

Key physical principles modeled:

1. **Inverse Square Law**: Sound intensity decreases with the square of the distance from the source
   - Implemented as: `intensity = initial_energy / (distance^2)`

2. **Sound Reflection**: Sound waves reflect off surfaces with the angle of reflection equal to the angle of incidence
   - Reflection vectors calculated using: `reflection = incident - 2 * dot(incident, normal) * normal`
   - Each reflection reduces energy based on the material's absorption coefficient

3. **Energy Absorption**: Different materials absorb different amounts of sound energy
   - Materials have an absorption coefficient between 0 (perfect reflection) and 1 (complete absorption)
   - Each reflection: `new_energy = energy * (1 - absorption)`

4. **Propagation Delay**: Sound travels at approximately 343 m/s in air at room temperature
   - Time delay = distance / speed of sound

### Bidirectional Ray Tracing Implementation

The simulation uses a bidirectional ray tracing approach to model sound propagation accurately:

1. Rays are cast from the listener/player in all directions
2. When these rays intersect with sound sources, they're reflected back toward the player
3. If these reflected rays reach the player, the corresponding sound path is considered audible
4. The delay and intensity are calculated based on the total path length

In addition, direct sound paths from sources to the player are calculated to ensure continuous audio. This bidirectional approach provides more accurate simulation of how humans perceive sound by only processing audio paths that actually reach the listener.

## Features

- Real-time visualization of sound rays and propagation
- Bidirectional path tracing for accurate sound perception
- Physically-based sound reflection and absorption
- Spatial audio mixing based on ray tracing data
- Customizable environment with various obstacle types
- Interactive player movement and environment exploration

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

- **Bidirectional Ray Tracing**: Rays now originate from the player and reflect from sound sources back to the player
- **Direct Path Calculation**: Added direct path calculations from sources to player for continuous audio
- **Enhanced Spatial Audio**: Improved audio cues with interaural time and level differences
- **Performance Optimization**: Reduced minimum energy threshold for better sound propagation

## Future Directions

- 3D sound propagation
- Diffraction modeling for low frequencies
- Real-time audio synthesis
- Custom sound design tools
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project utilizes principles from geometric acoustics and computer graphics
- Inspired by ray tracing techniques used in visual rendering and architectural acoustics 