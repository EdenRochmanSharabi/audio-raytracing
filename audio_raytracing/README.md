# Audio Ray Tracing Simulation

A 2D simulation of sound propagation using ray tracing techniques.

## Running the Simulation

### Visual-Only Mode

If you're experiencing issues with audio dependencies (especially on Python 3.13+), you can run the visual-only version:

```bash
python main_visual.py
```

This version provides all the visualization features without requiring audio libraries.

Command line options:
- `-r, --rays`: Number of rays per sound source (default: 360)
- `-s, --speed`: Speed of sound in meters/second (default: 343.0)
- `-a, --animation-speed`: Animation speed multiplier (default: 1.0)
- `-e, --environment`: Path to environment JSON file (optional)

### Full Version (with Audio)

To run the full version with audio support (requires additional dependencies):

```bash
python main.py
```

Additional options for the full version:
- `-m, --mute`: Start with audio muted
- `-v, --volume`: Set audio volume (0.0 to 1.0)

## Controls

- **Arrow Keys**: Move player
- **Q/E**: Rotate player
- **Space**: Pause/Resume
- **R**: Reset simulation
- **+/-**: Adjust animation speed
- **H**: Toggle help
- **F1**: Toggle stats
- **ESC**: Quit
- **M**: Toggle audio (full version only)

## Dependencies for Full Version

The full audio version requires:
- Python 3.12 or earlier (Python 3.13 removed some audio modules)
- pydub
- pygame
- numpy

## Python 3.13 Compatibility Note

Python 3.13 removed the `audioop` module which pydub relies on. If you're using Python 3.13, you should use the visual-only version of the application. 