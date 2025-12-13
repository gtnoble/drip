# Drip

**Physics-based rainfall sound generator for noise masking and ambience**

Generate pleasant, non-repetitive binaural rainfall audio using simplified wave mechanics. Each raindrop is modeled as a point source with physics-derived acoustic properties, creating natural spatial imaging through wave propagation.

## Features

- **Physics-based synthesis**: Drop sizes follow Marshall-Palmer distribution, frequencies determined by Minnaert resonance or impact physics
- **Binaural rendering**: Independent wave propagation to each ear creates realistic spatial imaging
- **Never repeats**: Generates unique audio each time (unless seeded)
- **Multiple surface types**: Water (bubble resonance), hard surfaces (impact transients), capillary waves, pink/white noise
- **Perceptually optimized**: Optional hearing threshold filtering removes inaudible drops
- **Unix-style**: Composes with standard audio tools (sox, ffmpeg)
- **Fast generation**: Parallel processing, ~1-10× realtime

## Installation

```bash
# Clone or download
git clone <repository-url>
cd drip

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- NumPy ≥1.20.0
- SciPy ≥1.7.0

## Quick Start

```bash
# Generate 10 minutes of moderate rain
./drip.py -d 600 --rain-rate 10 -o rain.wav

# Heavy rain with higher drop density
./drip.py -d 300 --rain-rate 25 --drop-rate 10000 -o heavy_rain.wav

# Light drizzle (fewer, smaller drops)
./drip.py -d 600 --rain-rate 5 --drop-rate 2000 -o drizzle.wav

# Hard surface impacts (no bubble resonance)
./drip.py -d 300 --surface-type hard -o roof_rain.wav

# Generate and normalize in pipeline
./drip.py -d 600 --rain-rate 10 | sox -t wav - rain_normalized.wav norm
```

## Usage

```
drip.py -d DURATION [options]

Required:
  -d, --duration SECONDS      Duration of output audio

Key Parameters:
  --rain-rate MM/HR          Rainfall rate (controls drop size distribution)
                             Default: 10.0 mm/hr
                             Typical range: 1-50 (light drizzle to heavy rain)

  --drop-rate DROPS/SEC      Number of drops per second (controls density)
                             Default: 5000.0
                             Typical range: 1000-20000

  -q, --quality-factor Q     Resonance quality (2=splashy, 20=tonal)
                             Default: 10.0
                             Only applies to water/capillary surfaces

  --surface-type TYPE        Impact surface:
                             - water: Minnaert bubble resonance (default)
                             - hard: Ricker wavelet (rigid surfaces)
                             - capillary: Surface tension resonance
                             - pink_noise: Gaussian-windowed pink noise
                             - white_noise: Gaussian-windowed white noise

Spatial Parameters:
  -e, --ear-separation M     Distance between ears (default: 0.17m)
  -l, --listener-height M    Height above raindrop plane (default: 1.7m)

Output:
  -o, --output FILE          Output WAV file (default: stdout)
  -r, --sample-rate HZ       Sample rate (default: 44100)

Performance:
  -w, --workers N            Worker processes (default: auto-detect)
  -s, --seed N               Random seed for reproducibility

Advanced:
  --no-hearing-threshold     Disable perceptual filtering
  --mh-burn-in N            M-H sampler burn-in iterations (default: 1000)
  --mh-position-scale M      M-H spatial proposal scale (default: 5.0m)
```

## Examples

### Basic Workflows

```bash
# Quick test (30 seconds)
./drip.py -d 30 --rain-rate 15 -o test.wav

# Long-form ambience (1 hour)
./drip.py -d 3600 --rain-rate 10 --drop-rate 3000 -o rain_1hr.wav

# Different intensities
./drip.py -d 600 --rain-rate 5 --drop-rate 2000 -o light.wav   # Light
./drip.py -d 600 --rain-rate 15 --drop-rate 5000 -o moderate.wav # Moderate
./drip.py -d 600 --rain-rate 30 --drop-rate 10000 -o heavy.wav  # Heavy
```

### Surface Types

```bash
# Water surface (bubble resonance)
./drip.py -d 300 --surface-type water -q 10 -o water.wav

# Hard surface (roof, pavement)
./drip.py -d 300 --surface-type hard -o hard_surface.wav

# Capillary waves (surface tension)
./drip.py -d 300 --surface-type capillary -q 15 -o capillary.wav

# Broadband noise impacts
./drip.py -d 300 --surface-type pink_noise -o pink.wav
./drip.py -d 300 --surface-type white_noise -o white.wav
```

### Unix Pipeline Integration

```bash
# Normalize to -3dB
./drip.py -d 600 --rain-rate 10 | sox -t wav - rain_norm.wav norm -3

# Convert to MP3
./drip.py -d 600 --rain-rate 10 | ffmpeg -i pipe:0 -b:a 192k rain.mp3

# Apply EQ (boost low frequencies)
./drip.py -d 600 --rain-rate 10 | sox -t wav - rain_eq.wav bass +6

# Create seamless loop (crossfade)
./drip.py -d 600 --rain-rate 10 -o rain.wav
sox rain.wav rain.wav rain_loop.wav splice 600,0.5

# Mix multiple layers
./drip.py -d 300 --rain-rate 20 --surface-type water -o layer1.wav
./drip.py -d 300 --rain-rate 10 --surface-type hard -o layer2.wav
sox -m layer1.wav layer2.wav mixed.wav
```

### Reproducible Generation

```bash
# Same seed produces identical output
./drip.py -d 60 --rain-rate 10 -s 42 -o version1.wav
./drip.py -d 60 --rain-rate 10 -s 42 -o version2.wav
# version1.wav and version2.wav are identical
```

## How It Works

### Physical Model

Each raindrop is modeled using simplified physics:

1. **Drop Size Distribution**: Marshall-Palmer exponential distribution
   ```
   N(D) = N₀ exp(-Λ D)
   where Λ = 4.1 R^(-0.21)
   ```

2. **Acoustic Frequency** (depends on surface type):
   - **Water**: Minnaert bubble resonance: `f = 3.26 / R_bubble` (Hz)
   - **Hard**: Impact frequency: `f = v / (2R)` where v is terminal velocity
   - **Capillary**: Surface tension resonance: `f = √(σ/m)`

3. **Acoustic Energy**: Small fraction (~0.1%) of kinetic energy
   ```
   E_acoustic = 0.001 × (½ m v²)
   ```

4. **Wave Propagation**:
   - Geometric spreading: `1/r²` (hemispherical radiation)
   - Air absorption: Frequency-dependent attenuation
   - Time delay: `t = distance / 343 m/s`

### Synthesis Pipeline

```
Parameters → M-H Sampling → Grain Synthesis → Wave Propagation → Binaural Mixing → WAV Output
```

1. **Metropolis-Hastings Sampling**: Generates drop positions and sizes that are audible
2. **Grain Generation**: Filtered impulse (water) or Ricker wavelet (hard surface)
3. **Propagation**: Independent calculation for each ear
4. **Mixing**: Linear superposition of all grains

### Perceptual Optimization

The Metropolis-Hastings sampler (enabled by default) only generates drops that are:
- Below Nyquist frequency
- Above absolute threshold of hearing (frequency-dependent)
- Audible at listener position

This dramatically reduces computation while maintaining perceptual quality.

## Parameters Explained

### Rain Rate vs Drop Rate

- **`--rain-rate`** (mm/hr): Controls the **size distribution** of drops
  - Higher rate → more large drops in the distribution
  - Physical parameter based on meteorology
  - Affects timbre (larger drops have lower frequencies)

- **`--drop-rate`** (drops/sec): Controls the **density** of the sound
  - Higher rate → more drops per second
  - Directly controls how "busy" the sound is
  - Independent of drop size distribution

**Example**: Heavy rain might have high rain-rate (large drops) and high drop-rate (many drops), while light drizzle has low values for both.

### Quality Factor (Q)

Controls the resonance bandwidth (water/capillary surfaces only):
- **Low Q (2-5)**: Splashy, broadband, natural rain sound
- **Medium Q (8-12)**: Balanced, pleasant resonance
- **High Q (15-20)**: Tonal, ringing, musical

### Listener Height

Height above the raindrop plane affects:
- Spatial distribution (higher = narrower audible region)
- Distance-dependent filtering (higher = more air absorption)
- Typical values: 0.5m (sitting), 1.7m (standing), 3.0m (elevated)

## Diagnostic Tools

### Audible Region Visualization

Visualize which drops are audible vs inaudible:

```bash
./plot_audible_region.py -o audible_region.png
./plot_audible_region.py --surface-type hard -o audible_hard.png
```

Shows a 2D plot of radius vs distance, with black = audible, white = inaudible.

## Technical Details

### Grain Synthesis Methods

**Water Surface** (`--surface-type water`):
- Filtered impulse through bandpass filter
- Filter centered at Minnaert frequency
- Bandwidth controlled by Q factor
- Models bubble oscillation

**Hard Surface** (`--surface-type hard`):
- Ricker wavelet (2nd derivative of Gaussian)
- Frequency based on contact time
- Models impulsive collision transient
- No resonance (Q ignored)

**Capillary** (`--surface-type capillary`):
- Same as water but frequency from surface tension
- Models ripple oscillations
- Higher frequencies than bubble resonance

**Noise** (`--surface-type pink_noise` or `white_noise`):
- Gaussian-windowed noise
- Window duration from contact time
- Pink noise: ~1/f spectrum (Voss-McCartney algorithm)
- White noise: flat spectrum

### Air Absorption Model

Simplified frequency-dependent absorption:
```python
alpha(f) ≈ 0.02 × (f/1000)^1.5  dB/100m
gain = 10^(-alpha × distance/100 / 20)
```

Empirical fit for typical atmospheric conditions (20°C, 50% RH).

### Metropolis-Hastings Sampling

Uses M-H algorithm to sample from:
```
p(R, x, y) ∝ p_Marshall-Palmer(R) × I_audible(R, x, y)
```

Where:
- `p_Marshall-Palmer`: Exponential size distribution
- `I_audible`: Indicator function (1 if audible, 0 otherwise)

This ensures efficient sampling of only audible drops.

## Limitations

**Physical Approximations**:
- Single resonance mode per drop (water surfaces)
- Hemispherical radiation pattern (no directionality)
- Simplified air absorption (no humidity/temperature variation)
- 2D plane geometry (no drop height variation)
- No ground reflections

**Technical Constraints**:
- Pre-generation only (not real-time)
- WAV output only (use external tools for format conversion)
- No built-in effects (use sox/ffmpeg for processing)

These are intentional simplifications following the Unix philosophy: do one thing well.

## Contributing

This project follows the Unix philosophy and emphasizes physical accuracy over arbitrary parameters. When contributing:

- Maintain physics-based approach
- Add physical parameters only (no "wetness" sliders)
- Keep dependencies minimal
- Preserve composability with standard tools
