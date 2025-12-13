#!/usr/bin/env python
"""
Drip: Physics-based rainfall sound generator

Synthesizes binaural rainfall audio using simplified wave mechanics.
Drop sizes follow Marshall-Palmer distribution, Minnaert resonance 
determines frequencies, and hemispherical wave propagation creates
realistic spatial imaging.
"""

import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from scipy.stats import qmc
import argparse
import sys
import os
import time
from multiprocessing import Pool, Manager, cpu_count


# ==============================================================================
# Physics Module
# ==============================================================================

def minnaert_frequency(drop_radius):
    """
    Calculate Minnaert resonance frequency for air bubble
    
    f = 3.26 / R_bubble (simplified formula)
    
    Args:
        drop_radius: Drop radius in meters
        
    Returns:
        Frequency in Hz
    """
    # Bubble radius ≈ 0.7 × drop radius (Newton's impact depth approximation)
    bubble_radius = 0.7 * drop_radius
    return 3.26 / bubble_radius


def capillary_frequency(drop_radius):
    """
    Calculate capillary wave resonance frequency from surface tension
    
    f = sqrt(σ / m) where σ is surface tension and m is drop mass
    
    Args:
        drop_radius: Drop radius in meters
        
    Returns:
        Frequency in Hz
    """
    surface_tension = 0.072  # N/m (water-air interface at 20°C)
    rho_water = 1000  # kg/m³
    
    # Drop mass
    volume = (4/3) * np.pi * drop_radius**3
    mass = volume * rho_water
    
    # Capillary frequency
    return np.sqrt(surface_tension / mass)


def terminal_velocity(drop_radius):
    """
    Calculate terminal velocity of raindrop in air
    
    Simplified model assuming spherical drops with constant drag coefficient.
    Based on balance between gravitational and drag forces:
    v_terminal = sqrt((8/3) * (ρ_water/ρ_air) * g * R / C_d)
    
    Args:
        drop_radius: Drop radius in meters
        
    Returns:
        Velocity in m/s
    """
    rho_water = 1000  # kg/m³
    rho_air = 1.225   # kg/m³
    g = 9.81          # m/s²
    C_d = 0.47        # Drag coefficient for sphere
    
    v_terminal = np.sqrt((8/3) * (rho_water/rho_air) * g * drop_radius / C_d)
    return v_terminal


def calculate_kinetic_energy(drop_radius):
    """
    Calculate kinetic energy of falling raindrop
    
    E_kinetic = 0.5 * m * v^2
    
    Args:
        drop_radius: Drop radius in meters
        
    Returns:
        Kinetic energy in Joules
    """
    rho_water = 1000  # kg/m³
    
    # Drop mass
    volume = (4/3) * np.pi * drop_radius**3
    mass = volume * rho_water
    
    # Terminal velocity
    velocity = terminal_velocity(drop_radius)
    
    # Kinetic energy
    E_kinetic = 0.5 * mass * velocity**2
    
    return E_kinetic


def acoustic_amplitude(drop_radius, surface_type='water', Q=10.0, acoustic_efficiency=0.001):
    """
    Calculate acoustic energy from drop kinetic energy
    
    Args:
        drop_radius: Drop radius in meters
        surface_type: 'water' or 'hard' - determines duration calculation
        Q: Quality factor (only used for water surfaces)
        acoustic_efficiency: Fraction of kinetic energy converted to sound
        
    Returns:
        Acoustic energy in Joules
    """
    rho_water = 1000  # kg/m³
    
    # Drop mass
    volume = (4/3) * np.pi * drop_radius**3
    mass = volume * rho_water
    
    # Terminal velocity
    velocity = terminal_velocity(drop_radius)
    
    # Kinetic energy
    E_kinetic = 0.5 * mass * velocity**2
    
    # Acoustic energy (small fraction)
    E_acoustic = acoustic_efficiency * E_kinetic
    
    return E_acoustic


def a_weighting_db(frequency):
    """
    Calculate A-weighting in dB (IEC 61672-1:2013)
    
    Standard frequency weighting curve used in sound level measurement.
    Normalized to 0 dB at 1 kHz. Can be validated against published tables.
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        A-weighting value in dB
        
    Reference values for validation:
        100 Hz: -19.1 dB
        1000 Hz: 0.0 dB
        3000 Hz: +1.2 dB
        10000 Hz: -0.8 dB
    """
    f = frequency
    # IEC 61672-1:2013 reference frequencies
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    
    numerator = f4**2 * f**4
    denominator = ((f**2 + f1**2) * 
                   np.sqrt((f**2 + f2**2) * (f**2 + f3**2)) * 
                   (f**2 + f4**2))
    
    A = numerator / denominator
    return 20 * np.log10(A) + 2.0


def hearing_threshold_db_spl(frequency):
    """
    Calculate approximate audibility threshold using A-weighting
    
    Uses A-weighting curve anchored to typical absolute threshold at 1 kHz.
    While A-weighting is technically a loudness weighting (not a true absolute
    threshold curve), it provides:
    1. Verifiable formula (IEC 61672-1:2013 standard)
    2. Reasonable frequency-dependent sensitivity approximation
    3. Higher sensitivity around 2-5 kHz, reduced at low/high frequencies
    
    Anchor point: ~4 dB SPL at 1 kHz (typical absolute threshold)
    Formula: threshold(f) = 4 dB - A_weighting(f)
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        Threshold in dB SPL (re: 20 µPa)
    """
    if frequency < 20 or frequency > 20000:
        return 80.0  # Outside audible range
    
    # Reference threshold at 1 kHz is approximately 4 dB SPL
    # Apply inverse A-weighting to get frequency-dependent threshold
    return 4.0 - a_weighting_db(frequency)


def hearing_threshold_pa(frequency):
    """
    Calculate absolute threshold of hearing in Pascals
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        Threshold pressure in Pascals (Pa)
    """
    threshold_db = hearing_threshold_db_spl(frequency)
    p_ref = 20e-6  # Reference pressure: 20 micropascals
    return p_ref * 10 ** (threshold_db / 20.0)


def sample_drop_sizes(n_drops, rain_rate_mm_hr, max_diameter_mm=10.0):
    """
    Sample drop sizes from Marshall-Palmer distribution
    
    N(D) = N₀ exp(-Λ D)
    where Λ = 4.1 R^(-0.21)
    
    Args:
        n_drops: Number of drops to generate
        rain_rate_mm_hr: Rainfall rate in mm/hour
        max_diameter_mm: Maximum physically possible diameter in mm (default: 10.0)
        
    Returns:
        Array of drop radii in meters
    """
    # Marshall-Palmer slope parameter
    Lambda = 4.1 * rain_rate_mm_hr**(-0.21)
    
    # Sample from exponential distribution
    U = np.random.uniform(0, 1, n_drops)
    D_mm = -np.log(U) / Lambda
    
    # Filter out physically impossible large drops
    D_mm = np.minimum(D_mm, max_diameter_mm)
    
    # Convert diameter to radius, mm to meters
    R_m = (D_mm / 2) / 1000
    
    return R_m


def sample_drop_sizes_sobol(n_drops, rain_rate_mm_hr, max_diameter_mm=10.0):
    """
    Sample drop sizes from Marshall-Palmer distribution using Sobol sequence
    
    Uses quasi-random Sobol sequence for better coverage of the distribution
    compared to pseudo-random sampling. This is particularly useful for grain
    libraries where we want representative samples across the entire range.
    
    N(D) = N₀ exp(-Λ D)
    where Λ = 4.1 R^(-0.21)
    
    Args:
        n_drops: Number of drops to generate
        rain_rate_mm_hr: Rainfall rate in mm/hour
        max_diameter_mm: Maximum physically possible diameter in mm (default: 10.0)
        
    Returns:
        Array of drop radii in meters
    """
    # Marshall-Palmer slope parameter
    Lambda = 4.1 * rain_rate_mm_hr**(-0.21)
    
    # Sobol sequence works best with power-of-2 sample counts
    # Round up to nearest power of 2
    n_sobol = 2 ** int(np.ceil(np.log2(n_drops)))
    
    # Sample from Sobol sequence (quasi-random)
    sampler = qmc.Sobol(d=1, scramble=True)
    U = sampler.random(n_sobol).flatten()
    
    # Transform to exponential distribution using inverse CDF
    D_mm = -np.log(1 - U) / Lambda
    
    # Filter out physically impossible large drops
    D_mm = np.minimum(D_mm, max_diameter_mm)
    
    # Convert diameter to radius, mm to meters
    R_m = (D_mm / 2) / 1000
    
    return R_m


def get_dominant_frequency(drop_radius, surface_type='water'):
    """
    Calculate dominant frequency for a drop based on surface type
    
    Args:
        drop_radius: Drop radius in meters
        surface_type: 'water', 'hard', 'capillary', 'pink_noise', or 'white_noise'
        
    Returns:
        Dominant frequency in Hz
    """
    if surface_type == 'hard' or surface_type == 'pink_noise' or surface_type == 'white_noise':
        # Hard surface / pink noise / white noise: frequency from contact time
        diameter = 2 * drop_radius
        velocity = terminal_velocity(drop_radius)
        contact_time = diameter / velocity
        return 1.0 / contact_time
    elif surface_type == 'capillary':
        # Capillary wave resonance from surface tension
        return capillary_frequency(drop_radius)
    else:
        # Water surface: Minnaert bubble resonance
        return minnaert_frequency(drop_radius)


def calculate_impulse_duration(drop_radius, surface_type='water', Q=10.0):
    """
    Calculate acoustic impulse duration based on surface type
    
    Args:
        drop_radius: Drop radius in meters
        surface_type: 'water', 'hard', 'pink_noise', or 'white_noise'
        Q: Quality factor (only used for water surfaces)
        
    Returns:
        Duration in seconds
    """
    if surface_type == 'hard' or surface_type == 'pink_noise' or surface_type == 'white_noise':
        # Hard surface / pink noise / white noise: duration from contact time
        diameter = 2 * drop_radius
        velocity = terminal_velocity(drop_radius)
        contact_time = diameter / velocity
        return 4 * contact_time  # Match grain generation logic
    else:
        # Water surface: decay time from Q factor and Minnaert frequency
        frequency = minnaert_frequency(drop_radius)
        return Q / frequency


# ==============================================================================
# Grain Synthesis
# ==============================================================================

def generate_grain_water(drop_radius, Q, sample_rate, frequency=None):
    """
    Generate acoustic grain as filtered impulse for water surface impacts
    
    Models raindrop impact on water as impulse filtered through resonance.
    Can be used for Minnaert bubble resonance or capillary wave resonance.
    Grain is normalized to unit energy (sum of squares = 1.0).
    
    Args:
        drop_radius: Drop radius in meters
        Q: Quality factor (controls bandwidth: 2=splashy, 20=tonal)
        sample_rate: Audio sample rate (Hz)
        frequency: Resonance frequency in Hz (if None, uses Minnaert frequency)
        
    Returns:
        Numpy array containing grain signal with unit energy
    """
    # Calculate frequency if not provided
    if frequency is None:
        frequency = minnaert_frequency(drop_radius)
    
    # Duration based on decay time (Q/f)
    decay_time = Q / frequency
    duration = min(4 * decay_time, 0.5)  # Cap at 0.5 seconds
    n_samples = int(duration * sample_rate)
    
    if n_samples < 1:
        n_samples = 1
    
    # Create impulse at start
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0
    
    # Design bandpass filter centered at resonance frequency
    nyquist = sample_rate / 2
    bandwidth = frequency / Q
    
    f_low = (frequency - bandwidth/2) / nyquist
    f_high = (frequency + bandwidth/2) / nyquist
    
    # Clamp to valid range and ensure f_low < f_high
    f_low = max(f_low, 0.01)
    f_high = min(f_high, 0.99)
    
    # If bandwidth is too narrow or frequency too low, use minimum valid range
    if f_low >= f_high:
        f_low = 0.01
        f_high = min(0.02, 0.99)  # Minimum valid bandwidth
    
    # 2nd order Butterworth bandpass
    b, a = butter(2, [f_low, f_high], btype='band')
    
    # Filter impulse to get impulse response
    response = lfilter(b, a, impulse)
    
    # Normalize to unit energy
    grain_energy = np.sum(response**2)
    if grain_energy > 0:
        response = response / np.sqrt(grain_energy)
    
    return response


def generate_grain_ricker(drop_radius, sample_rate):
    """
    Generate acoustic grain using Ricker wavelet for hard surface impacts
    
    Models impacts on rigid surfaces where there is no bubble resonance,
    only the initial transient of the collision. The Ricker wavelet 
    (second derivative of Gaussian) represents this impulsive impact with
    zero DC bias. Grain is normalized to unit energy (sum of squares = 1.0).
    
    Impact duration is determined by contact time: t_contact = diameter / velocity
    
    Args:
        drop_radius: Drop radius in meters
        sample_rate: Audio sample rate (Hz)
        
    Returns:
        Numpy array containing grain signal with unit energy
    """
    # Calculate contact time from drop physics
    diameter = 2 * drop_radius
    velocity = terminal_velocity(drop_radius)
    contact_time = diameter / velocity
    
    # Duration should be several contact times for complete wavelet
    duration = min(4 * contact_time, 0.1)  # Cap at 0.1 seconds
    n_samples = int(duration * sample_rate)
    
    if n_samples < 1:
        n_samples = 1
    
    # Dominant frequency: f ~ 1/contact_time
    f_peak = 1.0 / contact_time
    
    # Time array centered at t=0
    t = np.linspace(-duration/2, duration/2, n_samples)
    
    # Ricker wavelet: ψ(t) = (1 - 2a) exp(-a) where a = (πft)²
    a = (np.pi * f_peak * t) ** 2
    ricker = (1 - 2*a) * np.exp(-a)
    
    # Normalize to unit energy
    grain_energy = np.sum(ricker**2)
    if grain_energy > 0:
        ricker = ricker / np.sqrt(grain_energy)
    
    return ricker


def generate_grain_pink_noise(drop_radius, sample_rate):
    """
    Generate acoustic grain using Gaussian-windowed pink noise for surface impacts
    
    Models broadband impact noise on surfaces using pink noise (1/f power spectrum)
    windowed by a Gaussian envelope. The window width is determined by contact time,
    and the signal energy is proportional to drop kinetic energy.
    Grain is normalized to unit energy (sum of squares = 1.0).
    
    Uses Voss-McCartney algorithm for efficient pink noise generation with proper
    handling of filter initialization transients.
    
    Impact duration is determined by contact time: t_contact = diameter / velocity
    
    Args:
        drop_radius: Drop radius in meters
        sample_rate: Audio sample rate (Hz)
        
    Returns:
        Numpy array containing grain signal with unit energy
    """
    from scipy.signal import lfilter, lfilter_zi
    
    # Calculate contact time from drop physics
    diameter = 2 * drop_radius
    velocity = terminal_velocity(drop_radius)
    contact_time = diameter / velocity
    
    # Duration should be several contact times for complete envelope
    duration = min(4 * contact_time, 0.1)  # Cap at 0.1 seconds
    n_samples = int(duration * sample_rate)
    
    if n_samples < 1:
        n_samples = 1
    
    # Generate white noise
    white_noise = np.random.randn(n_samples)
    
    # Pink noise filter using Voss-McCartney algorithm
    # Cascade of first-order lowpass filters at octave-spaced frequencies
    # This creates approximately -3dB/octave slope (pink noise)
    
    # Design cascade of 3 first-order lowpass filters
    # Cutoff frequencies at octave intervals: f_nyquist / 2^k
    nyquist = sample_rate / 2
    pink = white_noise.copy()
    
    # Three poles for good pink noise approximation
    for k in range(1, 4):
        cutoff = nyquist / (2 ** k)
        # First-order Butterworth lowpass: alpha = 2*pi*fc/fs
        alpha = 2 * np.pi * cutoff / sample_rate
        # IIR coefficients: y[n] = (1-a)*y[n-1] + a*x[n]
        a_coeff = alpha / (1 + alpha)
        b_coeff = [a_coeff]
        a_filt = [1, -(1 - a_coeff)]
        
        # Initialize filter state to steady-state for white noise input
        zi = lfilter_zi(b_coeff, a_filt)
        
        # Filter with initial conditions (eliminates transient)
        pink, _ = lfilter(b_coeff, a_filt, pink, zi=zi * pink[0])
    
    # Apply Gaussian window
    # Window parameters: centered at midpoint, σ = duration/6 (99.7% within window)
    t = np.linspace(0, duration, n_samples)
    t_center = duration / 2
    sigma = duration / 6
    gaussian_window = np.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    
    grain = pink * gaussian_window
    
    # Normalize to unit energy
    grain_energy = np.sum(grain**2)
    if grain_energy > 0:
        grain = grain / np.sqrt(grain_energy)
    
    return grain


def generate_grain_white_noise(drop_radius, sample_rate):
    """
    Generate acoustic grain using Gaussian-windowed white noise for surface impacts
    
    Models broadband impact noise on surfaces using white noise (flat power spectrum)
    windowed by a Gaussian envelope. The window width is determined by contact time,
    and the signal energy is proportional to drop kinetic energy.
    Grain is normalized to unit energy (sum of squares = 1.0).
    
    Impact duration is determined by contact time: t_contact = diameter / velocity
    
    Args:
        drop_radius: Drop radius in meters
        sample_rate: Audio sample rate (Hz)
        
    Returns:
        Numpy array containing grain signal with unit energy
    """
    # Calculate contact time from drop physics
    diameter = 2 * drop_radius
    velocity = terminal_velocity(drop_radius)
    contact_time = diameter / velocity
    
    # Duration should be several contact times for complete envelope
    duration = min(4 * contact_time, 0.1)  # Cap at 0.1 seconds
    n_samples = int(duration * sample_rate)
    
    if n_samples < 1:
        n_samples = 1
    
    # Generate white noise
    white_noise = np.random.randn(n_samples)
    
    # Apply Gaussian window
    # Window parameters: centered at midpoint, σ = duration/6 (99.7% within window)
    t = np.linspace(0, duration, n_samples)
    t_center = duration / 2
    sigma = duration / 6
    gaussian_window = np.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    
    grain = white_noise * gaussian_window
    
    # Normalize to unit energy
    grain_energy = np.sum(grain**2)
    if grain_energy > 0:
        grain = grain / np.sqrt(grain_energy)
    
    return grain


# ==============================================================================
# Wave Propagation
# ==============================================================================

def air_absorption_gain(frequency, distance):
    """
    Calculate frequency-dependent air absorption gain
    
    Model: α(f) ≈ k × f^1.5 (classical + molecular absorption)
    Empirical fit for typical atmospheric conditions (20°C, 50% humidity)
    
    Args:
        frequency: Signal frequency in Hz
        distance: Propagation distance in meters
        
    Returns:
        Gain factor (0 to 1)
    """
    if distance < 0.1:
        return 1.0
    
    # Absorption coefficient in dB/100m (empirical)
    # Roughly: 0.1 dB/100m at 1kHz, 1 dB/100m at 4kHz, 5 dB/100m at 10kHz
    f_khz = frequency / 1000.0
    alpha_db_per_100m = 0.02 * f_khz**1.5  # Empirical power law
    
    # Total attenuation
    attenuation_db = alpha_db_per_100m * (distance / 100.0)
    
    # Convert to linear gain
    gain = 10**(-attenuation_db / 20.0)
    
    return gain


def propagate_to_ear(grain, grain_frequency, drop_position, ear_position, 
                     source_energy, output_buffer, event_time, sample_rate):
    """
    Propagate wave from drop to ear and mix into output buffer
    
    Args:
        grain: Source emission signal (unit energy)
        grain_frequency: Dominant frequency of grain (Hz)
        drop_position: (x, y, z) tuple in meters
        ear_position: (x, y, z) tuple in meters
        source_energy: Intrinsic source acoustic energy in Joules
        output_buffer: Output audio buffer to mix into
        event_time: Drop event time in seconds
        sample_rate: Sample rate in Hz
    """
    # Calculate distance (3D Euclidean)
    dx = drop_position[0] - ear_position[0]
    dy = drop_position[1] - ear_position[1]
    dz = drop_position[2] - ear_position[2]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if distance < 0.01:
        distance = 0.01  # Avoid division by zero
    
    # Energy attenuation from geometric spreading (1/r² for hemispherical)
    # and air absorption
    geometric_attenuation = 1.0 / distance**2
    air_gain = air_absorption_gain(grain_frequency, distance)
    
    # Received energy
    received_energy = source_energy * geometric_attenuation * air_gain
    
    # Scale unit-energy grain to have correct energy
    received = grain * np.sqrt(received_energy)
    
    # Time delay (speed of sound = 343 m/s)
    delay = distance / 343.0
    sample_idx = int((event_time + delay) * sample_rate)
    
    # Mix into output buffer
    end_idx = sample_idx + len(received)
    if sample_idx >= 0 and sample_idx < len(output_buffer):
        # Handle buffer boundary
        available_space = len(output_buffer) - sample_idx
        samples_to_add = min(len(received), available_space)
        output_buffer[sample_idx:sample_idx + samples_to_add] += received[:samples_to_add]


# ==============================================================================
# Metropolis-Hastings Sampling
# ==============================================================================

def is_audible(R, x, y, z, ear_L, ear_R, sample_rate, 
               hearing_threshold_enabled, surface_type='water', Q=10.0):
    """
    Check if a drop at (x, y, z) with radius R is audible
    
    Combines two audibility checks:
    1. Nyquist frequency limit
    2. Hearing threshold (frequency + position dependent)
    
    Args:
        R: Drop radius in meters
        x, y, z: Drop position in meters
        ear_L, ear_R: Ear positions as (x, y, z) tuples
        sample_rate: Sample rate in Hz
        hearing_threshold_enabled: Whether to apply hearing threshold
        surface_type: 'water' or 'hard' - determines frequency calculation
        Q: Quality factor (only used for water surfaces)
        
    Returns:
        Boolean indicating if drop is audible
    """
    # Calculate frequency
    frequency = get_dominant_frequency(R, surface_type)
    
    # Nyquist frequency check
    nyquist = sample_rate / 2
    if frequency > nyquist:
        return False
    
    # Hearing threshold check (if enabled)
    if hearing_threshold_enabled:
        threshold_pa = hearing_threshold_pa(frequency)
        source_energy = acoustic_amplitude(R, surface_type, Q)
        
        # Calculate duration for intensity conversion
        duration = calculate_impulse_duration(R, surface_type, Q)
        
        # Check both ears
        for ear_pos in [ear_L, ear_R]:
            dx = x - ear_pos[0]
            dy = y - ear_pos[1]
            dz = z - ear_pos[2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            if distance < 0.01:
                distance = 0.01
            
            # Energy attenuation
            geometric_attenuation = 1.0 / distance**2
            air_gain = air_absorption_gain(frequency, distance)
            received_energy = source_energy * geometric_attenuation * air_gain
            
            # Convert energy to pressure: E -> Power -> Intensity -> Pressure
            power = received_energy / duration
            intensity = power / (2 * np.pi)  # Hemispherical spreading
            rho_c = 415  # Acoustic impedance of air
            received_pa = np.sqrt(intensity * rho_c)
            
            if received_pa >= threshold_pa:
                return True  # Audible at this ear
        
        return False  # Not audible at either ear
    
    return True  # Passed energy and Nyquist checks


def sample_audible_drops_mh(n_drops, duration, rain_rate, ear_separation,
                            listener_height=1.7, sample_rate=44100, 
                            hearing_threshold_enabled=True, burn_in=1000, radius_scale=None, 
                            position_scale=5.0, seed=None, surface_type='water', Q=10.0):
    """
    Generate audible drop parameters using Metropolis-Hastings sampling
    
    Yields drops just-in-time without pre-allocating arrays. Uses symmetric
    Gaussian random walk proposals on infinite 2D plane at z=0, with listener
    at specified height above the plane.
    
    Args:
        n_drops: Number of audible drops to generate
        duration: Total duration in seconds
        rain_rate: Rainfall rate in mm/hr (for Marshall-Palmer distribution)
        ear_separation: Distance between ears in meters
        sample_rate: Sample rate in Hz
        hearing_threshold_enabled: Whether to apply hearing threshold filter
        burn_in: Number of burn-in iterations to discard
        radius_scale: Proposal scale for radius (None = auto from Marshall-Palmer)
        position_scale: Proposal standard deviation for position (meters)
        seed: Random seed
        surface_type: 'water' or 'hard' - determines frequency calculation
        Q: Quality factor (only used for water surfaces)
        
    Yields:
        Tuples of (R, x, y, z, t, acceptance_rate) for each audible drop (z=0 for all drops)
        The last element is the current acceptance rate (accepted/total proposals)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Ear positions (3D: x, y, z)
    ear_L = (-ear_separation / 2, 0, listener_height)
    ear_R = (ear_separation / 2, 0, listener_height)
    
    # Auto-tune radius scale from Marshall-Palmer distribution
    if radius_scale is None:
        Lambda = 4.1 * rain_rate**(-0.21)  # Lambda is in mm^-1
        radius_scale = (1.0 / Lambda) / 1000.0  # Convert mm to meters
    
    # Initialize chain at origin with typical drop size
    Lambda = 4.1 * rain_rate**(-0.21)
    R_current = (1.0 / Lambda) / 2000.0  # Mean radius of Marshall-Palmer in meters (diameter/2/1000)
    x_current = 0.0
    y_current = 0.0
    
    # Maximum physically possible radius (10mm diameter)
    max_radius = 0.005  # 5mm radius in meters
    
    # Find initial audible state (required for M-H)
    max_attempts = 10000
    for attempt in range(max_attempts):
        if is_audible(R_current, x_current, y_current, 0.0, ear_L, ear_R,
                     sample_rate, hearing_threshold_enabled, surface_type, Q):
            break
        # Try different starting position
        R_current = sample_drop_sizes(1, rain_rate)[0]
        r = position_scale * np.sqrt(np.random.uniform())
        theta = np.random.uniform(0, 2*np.pi)
        x_current = r * np.cos(theta)
        y_current = r * np.sin(theta)
    else:
        raise RuntimeError(f"Could not find audible initial state after {max_attempts} attempts. "
                         "Try relaxing thresholds or increasing rain rate.")
    
    # Burn-in phase
    burn_in_accepted = 0
    burn_in_total = 0
    for i in range(burn_in):
        # Propose new state (Gaussian random walk)
        R_proposed = R_current + np.random.normal(0, radius_scale)
        x_proposed = x_current + np.random.normal(0, position_scale)
        y_proposed = y_current + np.random.normal(0, position_scale)
        
        burn_in_total += 1
        
        # Reject negative radii or physically impossible large drops
        if R_proposed <= 0 or R_proposed > max_radius:
            continue
        
        # Acceptance probability
        # For symmetric proposals: α = min(1, p(proposed)/p(current))
        # where p ∝ p_Marshall-Palmer(R) * I_audible(R,x,y)
        
        # Marshall-Palmer probability ratio
        # Marshall-Palmer uses diameter D in mm: N(D) = N₀ exp(-Λ D)
        # Convert radius (m) to diameter (mm): D_mm = 2 * R_m * 1000
        exponent = -Lambda * (R_proposed - R_current) * 2000  # 2*1000: radius→diameter, m→mm
        p_ratio = np.exp(np.clip(exponent, -100, 100))
        
        # Audibility indicator
        audible_proposed = is_audible(R_proposed, x_proposed, y_proposed, 0.0,
                                      ear_L, ear_R, 
                                      sample_rate, hearing_threshold_enabled, surface_type, Q)
        
        if audible_proposed:
            # Accept with probability p_ratio (clamped to 1)
            if np.random.uniform() < min(1.0, p_ratio):
                R_current = R_proposed
                x_current = x_proposed
                y_current = y_proposed
                burn_in_accepted += 1
        # If not audible, automatically reject (implicit in not updating state)
    
    # Generate samples
    accepted = 0
    total_proposals = 0
    generated = 0
    while generated < n_drops:
        # Propose new state
        R_proposed = R_current + np.random.normal(0, radius_scale)
        x_proposed = x_current + np.random.normal(0, position_scale)
        y_proposed = y_current + np.random.normal(0, position_scale)
        
        total_proposals += 1
        
        # Metropolis-Hastings acceptance logic
        # Reject negative radii or physically impossible large drops (implicit rejection)
        if R_proposed > 0 and R_proposed <= max_radius:
            # Marshall-Palmer probability ratio
            # Marshall-Palmer uses diameter D in mm: N(D) = N₀ exp(-Λ D)
            # Convert radius (m) to diameter (mm): D_mm = 2 * R_m * 1000
            exponent = -Lambda * (R_proposed - R_current) * 2000  # 2*1000: radius→diameter, m→mm
            p_ratio = np.exp(np.clip(exponent, -100, 100))
            
            # Audibility check
            audible_proposed = is_audible(R_proposed, x_proposed, y_proposed, 0.0,
                                          ear_L, ear_R,
                                          sample_rate, hearing_threshold_enabled, surface_type, Q)
            
            # Accept if audible and passes M-H test
            if audible_proposed:
                if np.random.uniform() < min(1.0, p_ratio):
                    R_current = R_proposed
                    x_current = x_proposed
                    y_current = y_proposed
                    accepted += 1
        
        # Always yield current state (whether proposal was accepted or rejected)
        t = np.random.uniform(0, duration)
        acceptance_rate = accepted / total_proposals if total_proposals > 0 else 0.0
        yield (R_current, x_current, y_current, 0.0, t, acceptance_rate)
        generated += 1


# ==============================================================================
# Main Synthesis
# ==============================================================================

def _synthesize_rain_worker(worker_id, n_drops, progress_dict, duration, 
                            Q, rain_rate, 
                            ear_separation, listener_height, sample_rate, seed,
                            hearing_threshold_enabled, mh_burn_in, mh_radius_scale,
                            mh_position_scale, surface_type='water'):
    """
    Worker function to synthesize a subset of rain drops using M-H sampling
    
    Args:
        worker_id: Worker identifier for seeding
        n_drops: Number of audible drops this worker should generate
        progress_dict: Shared dictionary for progress reporting
        duration: Total duration in seconds (for timing drops)
        bandwidth: Grain bandwidth 0-100
        Q: Quality factor
        rain_rate: Rainfall rate in mm/hr
        ear_separation: Distance between ears in meters
        sample_rate: Sample rate in Hz
        seed: Base random seed (None or integer)
        hearing_threshold_enabled: Whether to filter by hearing threshold
        mh_burn_in: M-H burn-in iterations
        mh_radius_scale: M-H proposal scale for radius
        mh_position_scale: M-H proposal scale for position
        
    Returns:
        Tuple of (left_channel, right_channel, drops_processed, mh_acceptance_rate) 
    """
    # Set worker-specific seed
    worker_seed = seed + worker_id if seed is not None else None
    
    # Initialize worker's progress counter
    progress_dict[worker_id] = 0
    
    # Pre-generate grain library for this worker
    n_grain_templates = 1000
    grain_library, grain_radii = pregenerate_grain_library(
        n_grain_templates, rain_rate, Q, sample_rate, surface_type
    )
    
    # Initialize output buffers (estimate max delay from typical audible distance)
    n_samples = int(duration * sample_rate)
    max_delay_samples = int((100.0 / 343.0) * sample_rate)  # 100m max distance estimate
    output_L = np.zeros(n_samples + max_delay_samples, dtype=np.float32)
    output_R = np.zeros(n_samples + max_delay_samples, dtype=np.float32)
    
    # Ear positions (3D: x, y, z)
    ear_L = (-ear_separation / 2, 0, listener_height)
    ear_R = (ear_separation / 2, 0, listener_height)
    
    # Create M-H generator
    drop_generator = sample_audible_drops_mh(
        n_drops=n_drops,
        duration=duration,
        rain_rate=rain_rate,
        ear_separation=ear_separation,
        listener_height=listener_height,
        sample_rate=sample_rate,
        hearing_threshold_enabled=hearing_threshold_enabled,
        burn_in=mh_burn_in,
        radius_scale=mh_radius_scale,
        position_scale=mh_position_scale,
        seed=worker_seed,
        surface_type=surface_type,
        Q=Q
    )
    
    # Process drops from M-H sampler
    drops_processed = 0
    progress_interval = max(1, n_drops // 100)
    last_acceptance_rate = 0.0
    
    for R, x, y, z, t, acceptance_rate in drop_generator:
        drops_processed += 1
        last_acceptance_rate = acceptance_rate
        
        # Update progress periodically
        if drops_processed % progress_interval == 0:
            progress_dict[worker_id] = drops_processed
        
        # Calculate acoustic properties
        frequency = get_dominant_frequency(R, surface_type)
        source_energy = acoustic_amplitude(R, surface_type, Q)
        
        # Look up unit-energy grain
        grain = find_nearest_grain(R, grain_radii, grain_library)
        
        # Propagate to both ears
        drop_pos = (x, y, z)
        propagate_to_ear(grain, frequency, drop_pos, ear_L, 
                        source_energy, output_L, t, sample_rate)
        propagate_to_ear(grain, frequency, drop_pos, ear_R, 
                        source_energy, output_R, t, sample_rate)
    
    # Trim to desired length
    output_L = output_L[:n_samples]
    output_R = output_R[:n_samples]
    
    # Final progress update
    progress_dict[worker_id] = drops_processed
    
    return output_L, output_R, drops_processed, last_acceptance_rate


def pregenerate_grain_library(n_grains, rain_rate, Q, sample_rate, surface_type='water'):
    """
    Pre-generate library of grains for different drop sizes
    
    Uses Sobol quasi-random sampling for better coverage of the Marshall-Palmer
    drop size distribution compared to pseudo-random sampling.
    All grains are normalized to unit energy.
    
    Args:
        n_grains: Number of grain templates to generate
        rain_rate: Rain rate in mm/hr (determines drop size distribution)
        Q: Quality factor (only used for water surface)
        sample_rate: Sample rate in Hz
        surface_type: 'water' or 'hard' - determines grain generation method
        
    Returns:
        Tuple of (grain_library, radii) where grain_library[i] corresponds to radii[i]
    """
    # Sample drop sizes from the distribution using Sobol sequence
    radii = sample_drop_sizes_sobol(n_grains, rain_rate)
    
    # Sort by radius for organized lookup
    radii = np.sort(radii)
    
    # Filter out drops above Nyquist frequency
    nyquist = sample_rate / 2
    valid_radii = []
    grain_library = []
    
    for R in radii:
        frequency = get_dominant_frequency(R, surface_type)
        if frequency <= nyquist:
            if surface_type == 'hard':
                grain = generate_grain_ricker(R, sample_rate)
            elif surface_type == 'pink_noise':
                grain = generate_grain_pink_noise(R, sample_rate)
            elif surface_type == 'white_noise':
                grain = generate_grain_white_noise(R, sample_rate)
            else:  # surface_type == 'water' or 'capillary'
                # Pass frequency explicitly for capillary mode
                grain = generate_grain_water(R, Q, sample_rate, frequency=frequency)
            grain_library.append(grain)
            valid_radii.append(R)
    
    return grain_library, np.array(valid_radii)


def pregenerate_air_filters(max_distance, n_filters, sample_rate):
    """
    Pre-generate air absorption filter coefficients for range of distances
    
    DEPRECATED: Replaced by frequency-dependent gain approach
    """
    pass


def find_nearest_grain(target_radius, radii, grain_library):
    """
    Find pre-generated grain closest to target drop size
    
    Args:
        target_radius: Desired drop radius in meters
        radii: Array of radii for which grains were generated
        grain_library: List of pre-generated unit-energy grains
        
    Returns:
        Unit-energy grain from library
    """
    idx = np.searchsorted(radii, target_radius)
    # Clamp to valid range
    idx = min(idx, len(radii) - 1)
    return grain_library[idx]


def synthesize_rain(duration, rain_rate=10.0, Q=10.0, ear_separation=0.17,
                   listener_height=1.7, sample_rate=44100, seed=None, n_workers=None, 
                   hearing_threshold_enabled=True, mh_burn_in=1000, 
                   mh_radius_scale=None, mh_position_scale=5.0, drop_rate=5000.0,
                   surface_type='water'):
    """
    Synthesize binaural rainfall audio using Metropolis-Hastings sampling
    
    Args:
        duration: Length in seconds
        rain_rate: Rainfall rate in mm/hr (controls drop size distribution only)
        Q: Quality factor (controls resonance: low Q=splashy, high Q=tonal)
        ear_separation: Distance between ears in meters
        sample_rate: Audio sample rate in Hz
        seed: Random seed for reproducibility
        n_workers: Number of worker processes (None=auto-detect, 1=single-threaded)
        hearing_threshold_enabled: Filter drops below absolute threshold of hearing (default: True)
        mh_burn_in: M-H burn-in iterations (default: 1000)
        mh_radius_scale: M-H proposal scale for radius (None=auto from Marshall-Palmer)
        mh_position_scale: M-H proposal scale for position in meters (default: 5.0)
        drop_rate: Number of drops per second (default: 5000.0)
        surface_type: 'water' or 'hard' - determines grain synthesis method
        
    Returns:
        Tuple of (left_channel, right_channel) numpy arrays
    """
    # Determine number of workers
    if n_workers is None:
        try:
            n_workers = max(1, cpu_count() - 1)
        except:
            n_workers = 1
    
    # Calculate total drops from drop rate (independent of rain_rate)
    total_drops = int(drop_rate * duration)
    
    print(f"Generating ~{total_drops} audible drops over {duration}s using M-H sampling...", file=sys.stderr)
    print(f"Rain rate: {rain_rate:.1f} mm/hr (drop size distribution)", file=sys.stderr)
    print(f"Drop rate: {drop_rate:.1f} drops/s", file=sys.stderr)
    print(f"Q factor: {Q:.1f}", file=sys.stderr)
    print(f"M-H burn-in: {mh_burn_in}", file=sys.stderr)
    print(f"Workers: {n_workers}", file=sys.stderr)
    
    # Parallel processing
    if n_workers > 1:
        # Distribute drops across workers
        base_drops_per_worker = total_drops // n_workers
        
        # Create shared progress dictionary
        with Manager() as manager:
            progress_dict = manager.dict()
            
            # Build task list
            tasks = []
            for worker_id in range(n_workers):
                # Last worker gets remainder
                if worker_id == n_workers - 1:
                    n_drops = base_drops_per_worker + (total_drops % n_workers)
                else:
                    n_drops = base_drops_per_worker
                
                tasks.append((
                    worker_id, n_drops, progress_dict, duration,
                    Q, rain_rate,
                    ear_separation, listener_height, sample_rate, seed,
                    hearing_threshold_enabled, mh_burn_in, mh_radius_scale,
                    mh_position_scale, surface_type
                ))
            
            # Launch workers
            with Pool(n_workers) as pool:
                result = pool.starmap_async(_synthesize_rain_worker, tasks)
                
                # Poll for progress
                while not result.ready():
                    # Sum progress from all workers
                    current_progress = sum(progress_dict.get(i, 0) for i in range(n_workers))
                    progress_pct = (current_progress / total_drops) * 100 if total_drops > 0 else 0
                    print(f"Progress: {progress_pct:.1f}% ({current_progress}/{total_drops} drops)",
                          file=sys.stderr, end='\r')
                    time.sleep(0.1)
                
                # Get results
                worker_outputs = result.get()
            
            # Calculate actual progress and acceptance rates
            total_processed = sum(drops for _, _, drops, _ in worker_outputs)
            acceptance_rates = [rate for _, _, _, rate in worker_outputs]
            avg_acceptance_rate = np.mean(acceptance_rates) if acceptance_rates else 0.0
            
            print(f"Progress: 100.0% ({total_processed}/{total_drops} drops)    ", file=sys.stderr)
            print(f"M-H acceptance rate: {avg_acceptance_rate*100:.2f}%", file=sys.stderr)
            
            # Sum all worker outputs
            n_samples = int(duration * sample_rate)
            output_L = np.zeros(n_samples, dtype=np.float32)
            output_R = np.zeros(n_samples, dtype=np.float32)
            
            for left, right, _, _ in worker_outputs:
                output_L += left
                output_R += right
    
    else:
        # Single-threaded M-H sampling
        print("Pre-generating grain library...", file=sys.stderr)
        n_grain_templates = 100
        grain_library, grain_radii = pregenerate_grain_library(
            n_grain_templates, rain_rate, Q, sample_rate, surface_type
        )
        print(f"Generated {n_grain_templates} grain templates", file=sys.stderr)
        
        # Initialize output buffers
        n_samples = int(duration * sample_rate)
        max_delay_samples = int((100.0 / 343.0) * sample_rate)  # 100m max distance estimate
        output_L = np.zeros(n_samples + max_delay_samples, dtype=np.float32)
        output_R = np.zeros(n_samples + max_delay_samples, dtype=np.float32)
        
        # Ear positions (3D: x, y, z)
        ear_L = (-ear_separation / 2, 0, listener_height)
        ear_R = (ear_separation / 2, 0, listener_height)
        
        # Create M-H generator
        drop_generator = sample_audible_drops_mh(
            n_drops=total_drops,
            duration=duration,
            rain_rate=rain_rate,
            ear_separation=ear_separation,
            listener_height=listener_height,
            sample_rate=sample_rate,
            hearing_threshold_enabled=hearing_threshold_enabled,
            burn_in=mh_burn_in,
            radius_scale=mh_radius_scale,
            position_scale=mh_position_scale,
            seed=seed,
            surface_type=surface_type,
            Q=Q
        )
        
        # Process drops from M-H sampler
        last_acceptance_rate = 0.0
        for i, (R, x, y, z, t, acceptance_rate) in enumerate(drop_generator):
            last_acceptance_rate = acceptance_rate
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                progress = (i + 1) / total_drops * 100
                print(f"Progress: {progress:.1f}% ({i+1}/{total_drops} drops)", 
                      file=sys.stderr, end='\r')
            
            # Calculate acoustic properties
            frequency = get_dominant_frequency(R, surface_type)
            source_energy = acoustic_amplitude(R, surface_type, Q)
            
            # Look up unit-energy grain
            grain = find_nearest_grain(R, grain_radii, grain_library)
            
            # Propagate to both ears
            drop_pos = (x, y, z)
            propagate_to_ear(grain, frequency, drop_pos, ear_L, 
                            source_energy, output_L, t, sample_rate)
            propagate_to_ear(grain, frequency, drop_pos, ear_R, 
                            source_energy, output_R, t, sample_rate)
        
        print(file=sys.stderr)  # New line after progress
        print(f"M-H acceptance rate: {last_acceptance_rate*100:.2f}%", file=sys.stderr)
        
        # Trim to desired length
        output_L = output_L[:n_samples]
        output_R = output_R[:n_samples]
    
    # Normalize to prevent clipping
    max_amplitude = max(np.max(np.abs(output_L)), np.max(np.abs(output_R)))
    if max_amplitude > 0:
        # Leave some headroom
        scale = 0.95 / max_amplitude
        output_L *= scale
        output_R *= scale
    
    return output_L, output_R


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate binaural rainfall audio using physics-based synthesis with M-H sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d 600 --rain-rate 10 -o rain.wav
  %(prog)s -d 300 --rain-rate 25 -q 5 -o heavy_rain.wav
  %(prog)s -d 600 --rain-rate 5 | sox -t wav - rain_normalized.wav norm
        """
    )
    
    parser.add_argument('-d', '--duration', type=float, required=True,
                       help='Duration in seconds')
    parser.add_argument('--rain-rate', type=float, default=10.0,
                       help='Rainfall rate in mm/hr - controls drop size distribution only (default: 10.0)')
    parser.add_argument('--drop-rate', type=float, default=5000.0,
                       help='Number of drops per second - controls sound density (default: 5000.0, typical range: 1000-20000)')
    parser.add_argument('-q', '--quality-factor', type=float, default=10.0,
                       help='Quality factor (Q): controls resonance (2=splashy, 20=tonal, default: 10)')
    parser.add_argument('-e', '--ear-separation', type=float, default=0.17,
                       help='Ear separation in meters (default: 0.17)')
    parser.add_argument('-l', '--listener-height', type=float, default=1.7,
                       help='Listener height above raindrop plane in meters (default: 1.7m, typical standing height)')
    parser.add_argument('-r', '--sample-rate', type=int, default=44100,
                       help='Sample rate in Hz (default: 44100)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output WAV file (default: stdout)')
    parser.add_argument('-s', '--seed', type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('-w', '--workers', type=int,
                       help='Number of worker processes (default: auto-detect CPU count - 1, use 1 for single-threaded)')
    parser.add_argument('--no-hearing-threshold', action='store_true',
                       help='Disable frequency-dependent hearing threshold filtering (enabled by default)')
    parser.add_argument('--mh-burn-in', type=int, default=1000,
                       help='Metropolis-Hastings burn-in iterations (default: 1000)')
    parser.add_argument('--mh-radius-scale', type=float,
                       help='M-H proposal scale for drop radius (default: auto from Marshall-Palmer)')
    parser.add_argument('--mh-position-scale', type=float, default=5.0,
                       help='M-H proposal scale for position in meters (default: 5.0)')
    parser.add_argument('--surface-type', type=str, default='water',
                       choices=['water', 'hard', 'capillary', 'pink_noise', 'white_noise'],
                       help='Surface impact type: "water" for bubble resonance (default), "hard" for rigid surface impacts, "capillary" for surface tension resonance, "pink_noise" for Gaussian-windowed pink noise, "white_noise" for Gaussian-windowed white noise')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.duration <= 0:
        parser.error("Duration must be positive")
    if args.rain_rate <= 0:
        parser.error("Rain rate must be positive")
    if args.drop_rate <= 0:
        parser.error("Drop rate must be positive")
    if args.quality_factor <= 0:
        parser.error("Quality factor must be positive")
    if args.ear_separation <= 0:
        parser.error("Ear separation must be positive")
    if args.mh_burn_in < 0:
        parser.error("M-H burn-in must be non-negative")
    if args.mh_position_scale <= 0:
        parser.error("M-H position scale must be positive")
    
    # Warn if quality factor is explicitly set with hard surface
    if args.surface_type == 'hard' and '--quality-factor' in sys.argv or '-q' in sys.argv:
        print("Warning: --quality-factor is ignored for hard surface impacts (no resonance)", 
              file=sys.stderr)
    
    # Synthesize
    left, right = synthesize_rain(
        duration=args.duration,
        rain_rate=args.rain_rate,
        Q=args.quality_factor,
        ear_separation=args.ear_separation,
        listener_height=args.listener_height,
        sample_rate=args.sample_rate,
        seed=args.seed,
        n_workers=args.workers,
        hearing_threshold_enabled=not args.no_hearing_threshold,
        mh_burn_in=args.mh_burn_in,
        mh_radius_scale=args.mh_radius_scale,
        mh_position_scale=args.mh_position_scale,
        drop_rate=args.drop_rate,
        surface_type=args.surface_type
    )
    
    # Combine to stereo
    stereo = np.column_stack((left, right))
    
    # Convert to 16-bit PCM
    stereo_int16 = (stereo * 32767).astype(np.int16)
    
    # Write output
    if args.output:
        wavfile.write(args.output, args.sample_rate, stereo_int16)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        # Write to stdout
        wavfile.write(sys.stdout.buffer, args.sample_rate, stereo_int16)


if __name__ == '__main__':
    main()
