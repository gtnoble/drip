/**
 * Poisson process rain synthesis
 * 
 * Uses sample-by-sample Poisson process to determine drop timing.
 * At each sample, draws number of drops from Poisson(λ = drop_rate / sample_rate).
 * Drop parameters generated using M-H sampling for audibility.
 * Provides sequential buffer writes and streaming capability.
 */
module synthesis;

import std.stdio : stderr, writefln;
import std.math : sqrt, log10;
import std.typecons : tuple;
import std.algorithm : max;
import mir.random : Random, unpredictableSeed;
import mir.random.variable : poissonVar;

import physics;
import grain;
import propagation;
import sampler;

/**
 * Synthesize rain using Poisson process and physics-based granular synthesis
 * 
 * Params:
 *   duration = Length in seconds
 *   rain_rate = Rainfall rate in mm/hr (controls drop size distribution)
 *   Q = Quality factor (controls resonance: low Q=splashy, high Q=tonal)
 *   ear_separation = Distance between ears in meters
 *   listener_height = Height above raindrop plane in meters
 *   sample_rate = Audio sample rate in Hz
 *   seed = Random seed for reproducibility
 *   hearing_threshold_enabled = Filter drops below absolute threshold of hearing
 *   mh_burn_in = M-H burn-in iterations
 *   mh_radius_scale = M-H proposal scale for radius (0=auto)
 *   mh_position_scale = M-H proposal scale for position in meters
 *   drop_rate = Number of drops per second
 *   surface_type = "water", "pink_noise", or "white_noise"
 *   buffer_size = Sliding window buffer size in samples
 * 
 * Returns:
 *   Tuple of (left_channel, right_channel) float arrays
 */
auto synthesize_rain(
    double duration,
    double rain_rate = 10.0,
    double Q = 10.0,
    double ear_separation = 0.17,
    double listener_height = 1.7,
    int sample_rate = 44100,
    uint seed = 0,
    bool hearing_threshold_enabled = true,
    int mh_burn_in = 1000,
    double mh_radius_scale = 0.0,
    double mh_position_scale = 5.0,
    double drop_rate = 5000.0,
    string surface_type = "water",
    int buffer_size = 22050
) {
    // If seed is 0, generate random seed
    uint base_seed = (seed == 0) ? cast(uint)unpredictableSeed : seed;
    
    // Calculate total samples and Poisson lambda
    int n_samples = cast(int)(duration * sample_rate);
    double lambda = drop_rate / cast(double)sample_rate;  // Expected drops per sample
    
    // Calculate maximum grain duration for buffer size validation
    double max_drop_radius = 0.005;  // 5mm
    double max_grain_duration = calculate_impulse_duration(max_drop_radius, surface_type, Q);
    int max_grain_samples = cast(int)(max_grain_duration * sample_rate);
    
    if (buffer_size < max_grain_samples) {
        stderr.writefln("WARNING: buffer_size (%d samples = %.3fs) < max_grain_duration (%d samples = %.3fs)",
                       buffer_size, buffer_size / cast(double)sample_rate,
                       max_grain_samples, max_grain_duration);
        stderr.writefln("         Grains may be truncated! Consider increasing --buffer-size");
    }
    
    stderr.writefln("Generating rain with Poisson process (λ = %.6f drops/sample)", lambda);
    stderr.writefln("Duration: %.1fs (%d samples)", duration, n_samples);
    stderr.writefln("Expected total drops: ~%.0f", drop_rate * duration);
    stderr.writefln("Rain rate: %.1f mm/hr (drop size distribution)", rain_rate);
    stderr.writefln("Drop rate: %.1f drops/s", drop_rate);
    stderr.writefln("Q factor: %.1f", Q);
    stderr.writefln("Surface type: %s", surface_type);
    stderr.writefln("Buffer size: %d samples (%.3fs)", buffer_size, buffer_size / cast(double)sample_rate);
    stderr.writefln("M-H burn-in: %d", mh_burn_in);
    
    // Allocate sliding window buffers
    float[] buffer_L = new float[buffer_size];
    float[] buffer_R = new float[buffer_size];
    foreach (ref s; buffer_L) s = 0.0;
    foreach (ref s; buffer_R) s = 0.0;
    
    // Allocate output buffers
    float[] output_L = new float[n_samples];
    float[] output_R = new float[n_samples];
    foreach (ref s; output_L) s = 0.0;
    foreach (ref s; output_R) s = 0.0;
    
    // Initialize drop sampler state
    auto sampler_state = initialize_drop_sampler(
        rain_rate, ear_separation, listener_height, sample_rate,
        hearing_threshold_enabled, mh_burn_in, mh_radius_scale,
        mh_position_scale, base_seed, surface_type, Q
    );
    
    // Initialize Poisson RNG (separate from M-H RNG)
    auto poisson_rng = Random(base_seed + 1);
    auto poisson_dist = poissonVar!double(lambda);
    
    // Ear positions
    Position ear_left = Position(-ear_separation / 2.0, 0.0, listener_height);
    Position ear_right = Position(ear_separation / 2.0, 0.0, listener_height);
    
    int total_drops_generated = 0;
    int buffer_write_pos = 0;  // Track position in sliding buffer
    
    // Progress reporting
    int last_progress_pct = -1;
    
    // Process each sample in time order (Poisson process)
    foreach (sample_idx; 0 .. n_samples) {
        // Sample number of drops at this time instant
        int n_drops = cast(int)poisson_dist(poisson_rng);
        
        // Generate each drop for this sample time
        foreach (drop_num; 0 .. n_drops) {
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
            
            // Calculate event time for this sample
            double event_time = cast(double)sample_idx / cast(double)sample_rate;
            
            // Create grain ranges and propagate to sliding buffer
            if (surface_type == "water") {
                auto grain_L = WaterGrainRange.create(drop_R, Q, sample_rate, acoustic_energy);
                propagate_to_sliding_buffer(grain_L, f0, drop_pos, ear_left,
                                           buffer_L, buffer_write_pos, sample_rate);
                
                auto grain_R = WaterGrainRange.create(drop_R, Q, sample_rate, acoustic_energy);
                propagate_to_sliding_buffer(grain_R, f0, drop_pos, ear_right,
                                           buffer_R, buffer_write_pos, sample_rate);
            } else if (surface_type == "pink_noise") {
                auto grain_L = PinkNoiseGrainRange.create(drop_R, sample_rate, sampler_state.rng, acoustic_energy);
                propagate_to_sliding_buffer(grain_L, f0, drop_pos, ear_left,
                                           buffer_L, buffer_write_pos, sample_rate);
                
                auto grain_R = PinkNoiseGrainRange.create(drop_R, sample_rate, sampler_state.rng, acoustic_energy);
                propagate_to_sliding_buffer(grain_R, f0, drop_pos, ear_right,
                                           buffer_R, buffer_write_pos, sample_rate);
            } else if (surface_type == "white_noise") {
                auto grain_L = WhiteNoiseGrainRange.create(drop_R, sample_rate, sampler_state.rng, acoustic_energy);
                propagate_to_sliding_buffer(grain_L, f0, drop_pos, ear_left,
                                           buffer_L, buffer_write_pos, sample_rate);
                
                auto grain_R = WhiteNoiseGrainRange.create(drop_R, sample_rate, sampler_state.rng, acoustic_energy);
                propagate_to_sliding_buffer(grain_R, f0, drop_pos, ear_right,
                                           buffer_R, buffer_write_pos, sample_rate);
            }
            
            total_drops_generated++;
        }
        
        // Write completed sample from buffer to output
        // The current sample is at buffer_write_pos in the circular buffer
        output_L[sample_idx] = buffer_L[buffer_write_pos];
        output_R[sample_idx] = buffer_R[buffer_write_pos];
        
        // Clear buffer position for future use (circular buffer)
        buffer_L[buffer_write_pos] = 0.0;
        buffer_R[buffer_write_pos] = 0.0;
        
        // Advance circular buffer write position
        buffer_write_pos = (buffer_write_pos + 1) % buffer_size;
        
        // Progress reporting
        int progress_pct = cast(int)((sample_idx * 100) / n_samples);
        if (progress_pct != last_progress_pct && progress_pct % 5 == 0) {
            stderr.writefln("  Progress: %d%% (%d drops generated)...", 
                          progress_pct, total_drops_generated);
            last_progress_pct = progress_pct;
        }
    }
    
    double acceptance_rate = get_acceptance_rate(sampler_state);
    stderr.writefln("Completed: %d drops generated (M-H acceptance: %.2f%%)",
                   total_drops_generated, acceptance_rate * 100.0);
    
    // Normalize
    normalize_buffers(output_L, output_R);
    
    return tuple(output_L, output_R);
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
