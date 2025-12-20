/**
 * Circular buffer rain synthesis
 * 
 * Uses uniform random offset placement within circular buffer for seamless looping.
 * Total drops determined by drop_rate × duration, placed at random positions.
 * Drop parameters generated using M-H sampling for audibility.
 * Provides seamless circular buffer generation with wrap-around grain handling.
 */
module synthesis;

import std.stdio : stderr, writefln;
import std.math : sqrt, log10;
import std.typecons : tuple;
import std.algorithm : max;
import mir.random : Random, unpredictableSeed;
import mir.random.variable : poissonVar, uniformVar;

import physics;
import grain;
import propagation;
import sampler;
import surface;

/**
 * Synthesize rain using circular buffer with uniform random offset placement
 * 
 * Params:
 *   duration = Length in seconds (also buffer size)
 *   rain_rate = Rainfall rate in mm/hr (controls drop size distribution)
 *   ear_separation = Distance between ears in meters
 *   listener_height = Height above raindrop plane in meters
 *   sample_rate = Audio sample rate in Hz
 *   seed = Random seed for reproducibility
 *   hearing_threshold_enabled = Filter drops below absolute threshold of hearing
 *   mh_burn_in = M-H burn-in iterations
 *   mh_radius_scale = M-H proposal scale for radius (0=auto)
 *   mh_position_scale = M-H proposal scale for position in meters
 *   drop_rate = Total drops per second for the entire buffer
 *   surface_type = Surface type enumerator (water, capillary, pink_noise, white_noise)
 * 
 * Returns:
 *   Tuple of (left_channel, right_channel) float arrays
 */
auto synthesize_rain(
    double duration,
    double rain_rate = 10.0,
    double ear_separation = 0.17,
    double listener_height = 1.7,
    int sample_rate = 44_100,
    uint seed = 0,
    bool hearing_threshold_enabled = true,
    int mh_burn_in = 1000,
    double mh_radius_scale = 0.0,
    double mh_position_scale = 5.0,
    double drop_rate = 5000.0,
    SurfaceType surface_type = SurfaceType.water
) {
    // If seed is 0, generate random seed
    uint base_seed = (seed == 0) ? cast(uint)unpredictableSeed : seed;
    
    // Calculate buffer size from duration
    int buffer_size = cast(int)(duration * sample_rate);
    
    // Calculate total number of drops
    int total_drops = cast(int)(drop_rate * duration);
    
    stderr.writefln("Generating rain with circular buffer and uniform random offsets");
    stderr.writefln("Duration: %.1fs (%d samples)", duration, buffer_size);
    stderr.writefln("Total drops: %d", total_drops);
    stderr.writefln("Rain rate: %.1f mm/hr (drop size distribution)", rain_rate);
    stderr.writefln("Drop rate: %.1f drops/s", drop_rate);
    stderr.writefln("Surface type: %s", surface_type);
    stderr.writefln("M-H burn-in: %d", mh_burn_in);
    stderr.writefln("Seed: %u", base_seed);
    
    // Allocate circular buffers (same size as duration)
    float[] buffer_L = new float[buffer_size];
    float[] buffer_R = new float[buffer_size];
    foreach (ref s; buffer_L) s = 0.0;
    foreach (ref s; buffer_R) s = 0.0;
    
    // Initialize drop sampler state
    auto sampler_state = initialize_drop_sampler(
        rain_rate, ear_separation, listener_height, sample_rate,
        hearing_threshold_enabled, mh_burn_in, mh_radius_scale,
        mh_position_scale, base_seed, surface_type
    );
    
    // Initialize RNG for drop placement (separate from M-H RNG)
    auto placement_rng = Random(base_seed + 1);
    
    // Pre-create uniform distribution for better performance
    auto uniform_dist = uniformVar!int(0, buffer_size);
    
    // Ear positions
    Position ear_left = Position(-ear_separation / 2.0, 0.0, listener_height);
    Position ear_right = Position(ear_separation / 2.0, 0.0, listener_height);
    
    // We'll generate drop positions on-the-fly using mir.random.variable.uniformVar
    
    stderr.writefln("Placing %d drops at uniform random positions...", total_drops);
    
    // Process each drop
    int last_progress_pct = -1;
    foreach (drop_idx; 0 .. total_drops) {
        // Sample drop parameters using M-H
        auto drop = sample_drop(sampler_state);
        double drop_R = drop.radius;
        double drop_x = drop.position.x;
        double drop_y = drop.position.y;
        double drop_z = drop.position.z;
        
        // Calculate physical properties
        double f0 = get_dominant_frequency(drop_R, surface_type);
        double acoustic_energy = acoustic_amplitude(drop_R);
        Position drop_pos = Position(drop_x, drop_y, drop_z);
        
        // Drop position in buffer (uniform random on-the-fly)
        int buffer_pos = uniform_dist(placement_rng);
        double event_time = cast(double)buffer_pos / cast(double)sample_rate;
        
        // Create grain ranges and propagate to circular buffer
        final switch (surface_type) {
            case SurfaceType.water:
            case SurfaceType.capillary:
                // Calculate physics-based Q factor
                double Q = get_quality_factor(drop_R, surface_type);
                auto grain_L = WaterGrainRange.create(drop_R, Q, sample_rate, acoustic_energy, surface_type);
                propagate_to_sliding_buffer(grain_L, f0, drop_pos, ear_left,
                                           buffer_L, buffer_pos, sample_rate);
                
                auto grain_R = WaterGrainRange.create(drop_R, Q, sample_rate, acoustic_energy, surface_type);
                propagate_to_sliding_buffer(grain_R, f0, drop_pos, ear_right,
                                           buffer_R, buffer_pos, sample_rate);
                break;
            case SurfaceType.pink_noise:
                auto grain_L = PinkNoiseGrainRange.create(drop_R, sample_rate, sampler_state.rng, acoustic_energy);
                propagate_to_sliding_buffer(grain_L, f0, drop_pos, ear_left,
                                           buffer_L, buffer_pos, sample_rate);
                
                auto grain_R = PinkNoiseGrainRange.create(drop_R, sample_rate, sampler_state.rng, acoustic_energy);
                propagate_to_sliding_buffer(grain_R, f0, drop_pos, ear_right,
                                           buffer_R, buffer_pos, sample_rate);
                break;
            case SurfaceType.white_noise:
                auto grain_L = WhiteNoiseGrainRange.create(drop_R, sample_rate, sampler_state.rng, acoustic_energy);
                propagate_to_sliding_buffer(grain_L, f0, drop_pos, ear_left,
                                           buffer_L, buffer_pos, sample_rate);
                
                auto grain_R = WhiteNoiseGrainRange.create(drop_R, sample_rate, sampler_state.rng, acoustic_energy);
                propagate_to_sliding_buffer(grain_R, f0, drop_pos, ear_right,
                                           buffer_R, buffer_pos, sample_rate);
                break;
        }
        
        // Progress reporting
        int progress_pct = cast(int)((drop_idx * 100) / total_drops);
        if (progress_pct != last_progress_pct && progress_pct % 10 == 0) {
            stderr.writefln("  Progress: %d%%...", progress_pct);
            last_progress_pct = progress_pct;
        }
    }
    
    double acceptance_rate = get_acceptance_rate(sampler_state);
    stderr.writefln("Completed: %d drops generated (M-H acceptance: %.2f%%)",
                   total_drops, acceptance_rate * 100.0);
    
    // Normalize circular buffers
    normalize_buffers(buffer_L, buffer_R);
    
    return tuple(buffer_L, buffer_R);
}

/**
 * Normalize stereo buffers to prevent clipping
 */
void normalize_buffers(ref float[] left, ref float[] right) {
    float max_amplitude = 0.0;
    
    foreach (sample; left) {
        float abs_sample = (sample < 0) ? -sample : sample;
        if (abs_sample > max_amplitude) max_amplitude = abs_sample;
    }
    
    foreach (sample; right) {
        float abs_sample = (sample < 0) ? -sample : sample;
        if (abs_sample > max_amplitude) max_amplitude = abs_sample;
    }
    
    if (max_amplitude > 0.0) {
        float scale = 0.95 / max_amplitude;  // Leave 5% headroom
        
        foreach (ref sample; left) sample *= scale;
        foreach (ref sample; right) sample *= scale;
        
        stderr.writefln("Normalized by %.2f dB", 20.0 * log10(scale));
    } else {
        stderr.writeln("Warning: All samples are zero!");
    }
}
