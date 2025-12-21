# Drip

**Physics-based rainfall sound generator for noise masking and ambience**

Generate pleasant, non-repetitive binaural rainfall audio using simplified wave mechanics. Each raindrop is modeled as a point source with physics-derived acoustic properties, creating natural spatial imaging through wave propagation.

## Features

- **Physics-based synthesis**: Drop sizes follow Marshall-Palmer distribution, frequencies determined by Minnaert resonance
- **Binaural rendering**: Independent wave propagation to each ear creates realistic spatial imaging
- **Never repeats**: Generates unique audio each time (unless seeded)
- **Multiple surface types**: Water (bubble resonance), capillary waves, pink/white noise
- **Perceptually optimized**: Hearing threshold filtering removes inaudible drops
- **Unix-style**: Composes with standard audio tools (sox, ffmpeg)
- **High performance**: Streaming architecture with zero grain buffer allocation

## Installation

```bash
# Clone or download
git clone <repository-url>
cd drip

# Install D compiler (Ubuntu/Debian)
sudo apt install ldc dub

# On macOS
brew install ldc dub

# Build project
dub build

# Or run directly with dub
dub run -- --duration 60 --rain-rate 10
```

**Requirements:**
- D compiler (LDC, DMD, or GDC)
- DUB (D's package manager)
- Dependencies handled automatically by DUB
  - mir-algorithm: Numerical operations and ndslice
  - mir-random: Random number generation and probability distributions
  - wave-d: WAV file I/O
  - darg: Command-line argument parsing

## Quick Start

```bash
# Generate 10 minutes of moderate rain
./drip -d 600 --rain-rate 10 -o rain.wav

# Heavy rain with higher drop density
./drip -d 300 --rain-rate 25 --drop-rate 10000 -o heavy_rain.wav

# Light drizzle (fewer, smaller drops)
./drip -d 600 --rain-rate 5 --drop-rate 2000 -o drizzle.wav

# Capillary wave surface (surface tension dominated)
./drip -d 300 --surface-type capillary -o capillary_rain.wav

# Generate and normalize in pipeline
./drip -d 600 --rain-rate 10 | sox -t wav - rain_normalized.wav norm
```

## Usage

```
./drip -d DURATION [options]

# Or via DUB:
dub run -- -d DURATION [options]

Required:
  -d, --duration SECONDS      Duration of output audio

Key Parameters:
  --rain-rate MM/HR          Rainfall rate (controls drop size distribution)
                             Default: 10.0 mm/hr
                             Typical range: 1-50 (light drizzle to heavy rain)

  --drop-rate DROPS/SEC      Number of drops per second (controls density)
                             Default: 5000.0
                             Typical range: 1000-20000

  --surface-type TYPE        Impact surface:
                             - water: Minnaert bubble resonance (default)
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
./drip -d 30 --rain-rate 15 -o test.wav

# Long-form ambience (1 hour)
./drip -d 3600 --rain-rate 10 --drop-rate 3000 -o rain_1hr.wav

# Different intensities
./drip -d 600 --rain-rate 5 --drop-rate 2000 -o light.wav   # Light
./drip -d 600 --rain-rate 15 --drop-rate 5000 -o moderate.wav # Moderate
./drip -d 600 --rain-rate 30 --drop-rate 10000 -o heavy.wav  # Heavy
```

### Surface Types

```bash
# Water surface (bubble resonance)
./drip -d 300 --surface-type water -o water.wav

# Capillary waves (surface tension)
./drip -d 300 --surface-type capillary -o capillary.wav

# Broadband noise impacts
./drip -d 300 --surface-type pink_noise -o pink.wav
./drip -d 300 --surface-type white_noise -o white.wav

### Reproducible Generation

```bash
# Same seed produces identical output
./drip -d 60 --rain-rate 10 --seed 42 -o version1.wav
./drip -d 60 --rain-rate 10 --seed 42 -o version2.wav
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
2. **Streaming Grain Generation**: 
   - Water/Capillary: Unit impulse → biquad filter (single-pass)
   - Pink/White Noise: Voss-McCartney/Gaussian → Gaussian window
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

The D implementation uses physics-based quality factors calculated from drop properties:
- **Water surface**: Q based on Minnaert resonance model (acoustic radiation resistance vs viscous resistance)
- **Capillary surface**: Q based on surface tension mechanics (mass inductance vs tension capacitance)
- **Noise surfaces**: Q not applicable (broadband by nature)

These physics-based Q factors vary with drop size, producing natural variation without manual control.

### Listener Height

Height above the raindrop plane affects:
- Spatial distribution (higher = narrower audible region)
- Distance-dependent filtering (higher = more air absorption)
- Typical values: 0.5m (sitting), 1.7m (standing), 3.0m (elevated)

## Technical Details

### Grain Synthesis Methods

**Water Surface** (`--surface-type water`):
- Filtered impulse through bandpass filter
- Filter centered at Minnaert frequency
- Bandwidth controlled by Q factor
- Models bubble oscillation

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
```
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
