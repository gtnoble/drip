#!/usr/bin/env python
"""
Generate diagnostic plot showing audible region vs droplet radius and distance.
Black pixels = audible, white pixels = inaudible.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

# Import functions from drip.py
from drip import (
    minnaert_frequency,
    acoustic_amplitude,
    air_absorption_gain,
    hearing_threshold_pa,
    get_dominant_frequency
)


def is_audible_diagnostic(R, distance, sample_rate=44100, surface_type='water'):
    """
    Check if a drop at given radius and distance is audible.
    
    Args:
        R: Drop radius in meters
        distance: Distance from listener in meters
        sample_rate: Sample rate in Hz
        surface_type: 'water' or 'hard'
        
    Returns:
        Boolean indicating if drop is audible
    """
    # Calculate frequency
    frequency = get_dominant_frequency(R, surface_type)
    
    # Nyquist frequency check
    nyquist = sample_rate / 2
    if frequency > nyquist:
        return False
    
    # Hearing threshold check
    threshold_pa = hearing_threshold_pa(frequency)
    amplitude = acoustic_amplitude(R)
    
    # Calculate received pressure at distance
    if distance < 0.01:
        distance = 0.01
    
    geometric_gain = 1.0 / distance**2
    air_gain = air_absorption_gain(frequency, distance)
    received_normalized = amplitude * geometric_gain * air_gain
    received_pa = received_normalized * 0.02  # Denormalize (ref: 60 dB SPL = 0.02 Pa)
    
    return received_pa >= threshold_pa


def plot_audible_region(radius_range=(0.0001, 0.005), distance_range=(0.1, 100),
                        resolution=(1000, 1000), sample_rate=44100,
                        surface_type='water', output_file=None):
    """
    Generate diagnostic plot of audible region.
    
    Args:
        radius_range: (min, max) drop radius in meters
        distance_range: (min, max) distance in meters
        resolution: (n_radius, n_distance) grid resolution
        sample_rate: Sample rate in Hz
        surface_type: 'water' or 'hard'
        output_file: Output filename (None = display)
    """
    # Create logarithmic grids for both axes
    radii = np.logspace(np.log10(radius_range[0]), np.log10(radius_range[1]), resolution[0])
    distances = np.logspace(np.log10(distance_range[0]), np.log10(distance_range[1]), resolution[1])
    
    print(f"Generating {resolution[0]}x{resolution[1]} grid...")
    print(f"Radius range: {radius_range[0]*1000:.4f} mm to {radius_range[1]*1000:.4f} mm")
    print(f"Distance range: {distance_range[0]:.2f} m to {distance_range[1]:.2f} m")
    
    # Create meshgrid
    R_grid, D_grid = np.meshgrid(radii, distances, indexing='ij')
    
    # Calculate audibility for each point
    audible_grid = np.zeros(R_grid.shape, dtype=bool)
    
    total_points = R_grid.size
    for i in range(resolution[0]):
        if i % 50 == 0:
            progress = (i * resolution[1]) / total_points * 100
            print(f"Progress: {progress:.1f}%", end='\r')
        
        for j in range(resolution[1]):
            audible_grid[i, j] = is_audible_diagnostic(
                R_grid[i, j], D_grid[i, j], sample_rate, surface_type
            )
    
    print(f"Progress: 100.0%")
    
    # Calculate percentage audible
    pct_audible = np.sum(audible_grid) / audible_grid.size * 100
    print(f"Audible region: {pct_audible:.2f}%")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Black = audible (1), White = inaudible (0)
    cmap = ListedColormap(['white', 'black'])
    
    # Plot using pcolormesh with log scales
    im = ax.pcolormesh(radii * 1000, distances, audible_grid.T, 
                       cmap=cmap, shading='auto')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Droplet Radius (mm)', fontsize=12)
    ax.set_ylabel('Distance from Listener (m)', fontsize=12)
    ax.set_title(f'Audible Region for Raindrop Impacts ({surface_type.title()} Surface)\n'
                f'Black = Audible, White = Inaudible | Sample Rate = {sample_rate} Hz',
                fontsize=13)
    
    # Add grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['Inaudible', 'Audible'])
    
    # Add percentage text
    ax.text(0.02, 0.98, f'Audible: {pct_audible:.1f}%',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Generate diagnostic plot of audible region vs radius and distance'
    )
    
    parser.add_argument('--min-radius', type=float, default=0.0001,
                       help='Minimum drop radius in meters (default: 0.0001 = 0.1mm)')
    parser.add_argument('--max-radius', type=float, default=0.005,
                       help='Maximum drop radius in meters (default: 0.005 = 5mm)')
    parser.add_argument('--min-distance', type=float, default=0.1,
                       help='Minimum distance in meters (default: 0.1)')
    parser.add_argument('--max-distance', type=float, default=100,
                       help='Maximum distance in meters (default: 100)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1000, 1000],
                       metavar=('N_RADIUS', 'N_DISTANCE'),
                       help='Grid resolution (default: 1000 1000)')
    parser.add_argument('-r', '--sample-rate', type=int, default=44100,
                       help='Sample rate in Hz (default: 44100)')
    parser.add_argument('--surface-type', type=str, default='water',
                       choices=['water', 'hard'],
                       help='Surface type (default: water)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output image file (default: display)')
    
    args = parser.parse_args()
    
    plot_audible_region(
        radius_range=(args.min_radius, args.max_radius),
        distance_range=(args.min_distance, args.max_distance),
        resolution=tuple(args.resolution),
        sample_rate=args.sample_rate,
        surface_type=args.surface_type,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
