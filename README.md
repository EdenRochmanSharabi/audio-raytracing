# Audio Ray Tracing Simulation

A 2D simulation of sound propagation using ray tracing techniques, demonstrating the principles of acoustic wave propagation in environments with obstacles.

## Scientific Background

### Acoustic Wave Propagation

Sound is a pressure wave that propagates through a medium like air. While sound waves actually propagate as spherical wavefronts in 3D environments, we can reasonably approximate sound propagation behavior using geometric ray tracing in certain frequency ranges. This simulation implements a ray-based approach to model how sound travels, reflects, and attenuates in a 2D environment.

### Ray Tracing for Sound

Ray tracing is traditionally used in computer graphics to simulate light paths, but it can also model sound propagation when certain conditions are met:

1. **Geometric Acoustics Assumption**: When sound wavelengths are small compared to obstacles (high-frequency sounds), sound behaves similarly to rays.
2. **Law of Reflection**: Sound reflects off surfaces following the same principle as light: angle of incidence equals angle of reflection.
3. **Energy Attenuation**: Sound energy decreases with distance (typically following the inverse square law) and is partially absorbed at each reflection.

### Physical Principles Implemented

This simulation implements several key acoustic phenomena:

#### 1. Distance-Based Attenuation

Sound intensity decreases as it travels through space. In free-field conditions, this follows the inverse square law:

```
I ∝ 1/d²
```

Where `I` is intensity and `d` is distance. For gameplay purposes, this simulation uses a modified attenuation model:

```
I ∝ 1/d
```

This allows sound to be audible over greater distances while still providing directional cues.

#### 2. Material-Based Absorption

When sound hits a surface, some energy is absorbed based on the material properties. Each material has an absorption coefficient (α) that represents the fraction of energy absorbed:

```
Ereflected = Eincident × (1-α)
```

Different materials (wood, concrete, glass, etc.) have different absorption coefficients, affecting how sound reflects and propagates.

#### 3. Spatial Audio Processing

The simulation creates a 3D audio experience by calculating:

- **Interaural Time Difference (ITD)**: The difference in arrival time of sound at each ear
- **Interaural Level Difference (ILD)**: The difference in sound intensity at each ear
- **Direct and Reflected Paths**: Combining direct sound paths with reflected paths that arrive later

#### 4. Propagation Delay

Sound travels at approximately 343 m/s in air at room temperature. The simulation calculates arrival time for each sound path based on distance:

```
delay = distance / speed_of_sound
```

This creates realistic timing for direct sounds and reflections.

## Implementation Approach

The simulation uses a hybrid approach that balances scientific accuracy with real-time performance:

1. **Ray-Based Propagation**: Casting rays from sound sources to model sound paths
2. **Direct Path Calculation**: Always calculating a direct path from source to listener for continuous audio
3. **Reflection Modeling**: Tracing rays as they bounce off obstacles with energy attenuation
4. **Spatial Audio Rendering**: Converting ray tracing data into stereo audio with proper spatial cues

## Controls

- **Arrow Keys**: Move player
- **Q/E**: Rotate player
- **Space**: Pause/Resume
- **R**: Reset simulation
- **+/-**: Adjust animation speed
- **H**: Toggle help
- **F1**: Toggle stats
- **ESC**: Quit
- **M**: Toggle audio

## Running the Simulation

```bash
# Install dependencies
pip install -r audio_raytracing/requirements.txt

# Run the simulation
python audio_raytracing/main.py

# Run with custom parameters
python audio_raytracing/main.py --rays 360 --animation-speed 1.0
```

## Command Line Options

- `-r, --rays`: Number of rays per sound source (default: 360)
- `-s, --speed`: Speed of sound in meters/second (default: 343.0)
- `-a, --animation-speed`: Animation speed multiplier (default: 1.0)
- `-e, --environment`: Path to environment JSON file (optional)
- `-m, --mute`: Start with audio muted
- `-v, --volume`: Set audio volume (0.0 to 1.0)

## Dependencies

- Python 3.10 or later (best with Python 3.12 or earlier)
- pygame
- numpy
- scipy
- pydub (for audio processing)

## Project Structure

```
audio_raytracing/
├── src/                   # Core simulation components
│   ├── environment.py     # 2D environment definition
│   ├── obstacles.py       # Obstacle classes with collision detection
│   ├── player.py          # Player/listener implementation
│   ├── sound_source.py    # Sound source implementation
│   ├── ray_tracer.py      # Ray tracing algorithm
│   └── audio/             # Audio processing components
│       ├── mixer.py       # Spatial audio mixer
│       └── player.py      # Audio playback
├── assets/                # Audio files and resources
├── tests/                 # Unit tests
├── main.py                # Main application entry point
└── requirements.txt       # Project dependencies
```

## Limitations and Future Work

This simulation is a simplified model of acoustic behavior and has several limitations:

1. **Frequency-Dependent Effects**: The current model doesn't account for frequency-dependent behaviors like diffraction around obstacles.
2. **Simplified Materials**: Material properties are reduced to a single reflection coefficient rather than frequency-dependent absorption.
3. **2D Only**: Real sound propagates in 3D, including reflections from floors and ceilings.

Future work could include:
- Frequency-dependent absorption and reflection
- Diffraction effects for low-frequency sounds
- Room impulse response generation for more realistic reverb
- 3D environment modeling 