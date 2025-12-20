/**
 * Wave propagation and binaural mixing
 * 
 * Implements geometric attenuation, frequency-dependent air absorption,
 * and propagation delay for binaural audio rendering.
 */
module propagation;

import std.math : sqrt, pow;
import std.algorithm : min;

import physics;

/**
 * 3D position (in meters)
 */
struct Position {
    double x, y, z;
}

/**
 * Calculate frequency-dependent air absorption gain
 * 
 * Model: α(f) ≈ k × f^1.5 (classical + molecular absorption)
 * Empirical fit for typical atmospheric conditions (20°C, 50% humidity)
 * 
 * Params:
 *   frequency = Signal frequency in Hz
 *   distance = Propagation distance in meters
 * 
 * Returns:
 *   Gain factor (0 to 1)
 */
double air_absorption_gain(double frequency, double distance) {
    if (distance < 0.1) return 1.0;
    
    // Absorption coefficient in dB/100m (empirical)
    // Roughly: 0.1 dB/100m at 1kHz, 1 dB/100m at 4kHz, 5 dB/100m at 10kHz
    double f_khz = frequency / 1000.0;
    double alpha_db_per_100m = 0.02 * pow(f_khz, 1.5);  // Empirical power law
    
    // Total attenuation
    double attenuation_db = alpha_db_per_100m * (distance / 100.0);
    
    // Convert to linear gain
    double gain = pow(10.0, -attenuation_db / 20.0);
    
    return gain;
}

/**
 * Propagate wave from drop to ear and mix into sliding window buffer
 * 
 * Applies geometric attenuation (1/r²), air absorption, and propagation delay.
 * Writes to circular buffer at positions relative to current write position.
 * 
 * Params:
 *   GrainRange = Input range type that yields double samples
 *   grain_range = Source emission signal with acoustic energy already scaled
 *   grain_frequency = Dominant frequency of grain (Hz)
 *   drop_position = Drop position in meters
 *   ear_position = Ear position in meters
 *   buffer = Circular buffer to write into
 *   current_pos = Current write position in circular buffer (for t=0 of grain)
 *   sample_rate = Sample rate in Hz
 */
void propagate_to_sliding_buffer(GrainRange)(ref GrainRange grain_range, double grain_frequency,
                     Position drop_position, Position ear_position,
                     ref float[] buffer, int current_pos, int sample_rate) {
    // Calculate distance (3D Euclidean)
    double dx = drop_position.x - ear_position.x;
    double dy = drop_position.y - ear_position.y;
    double dz = drop_position.z - ear_position.z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    if (distance < 0.01) distance = 0.01;  // Avoid singularity
    
    // Amplitude attenuation from geometric spreading and air absorption
    double geometric_attenuation = 1.0 / distance;
    double air_gain = air_absorption_gain(grain_frequency, distance);
    double amplitude_scale = geometric_attenuation * air_gain;
    
    // Time delay (speed of sound = 343 m/s)
    enum double speed_of_sound = 343.0;
    double delay = distance / speed_of_sound;
    int delay_samples = cast(int)(delay * sample_rate);
    
    // Write grain samples into circular buffer
    int i = 0;
    int buffer_size = cast(int)buffer.length;
    
    while (!grain_range.empty()) {
        int write_offset = delay_samples + i;
        int write_idx = (current_pos + write_offset) % buffer_size;
        
        double grain_sample = grain_range.front();
        buffer[write_idx] += cast(float)(grain_sample * amplitude_scale);
        
        grain_range.popFront();
        i++;
    }
}

/**
 * Propagate wave from drop to ear and mix into output buffer (legacy)
 * 
 * Applies geometric attenuation (1/r²), air absorption, and propagation delay.
 * Consumes grain range on-the-fly without buffering.
 * Grain already has acoustic energy scaled into it.
 * 
 * Params:
 *   GrainRange = Input range type that yields double samples
 *   grain_range = Source emission signal with acoustic energy already scaled
 *   grain_frequency = Dominant frequency of grain (Hz)
 *   drop_position = Drop position in meters
 *   ear_position = Ear position in meters
 *   output_buffer = Output audio buffer to mix into
 *   event_time = Drop event time in seconds
 *   sample_rate = Sample rate in Hz
 */
void propagate_to_ear(GrainRange)(ref GrainRange grain_range, double grain_frequency,
                     Position drop_position, Position ear_position,
                     ref float[] output_buffer,
                     double event_time, int sample_rate) {
    // Calculate distance (3D Euclidean)
    double dx = drop_position.x - ear_position.x;
    double dy = drop_position.y - ear_position.y;
    double dz = drop_position.z - ear_position.z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    if (distance < 0.01) distance = 0.01;  // Avoid singularity
    
    // Amplitude attenuation from geometric spreading (1/r for hemispherical amplitude)
    // and air absorption
    double geometric_attenuation = 1.0 / distance;
    double air_gain = air_absorption_gain(grain_frequency, distance);
    
    // Combined attenuation (grain already has acoustic energy scaled in)
    double amplitude_scale = geometric_attenuation * air_gain;
    
    // Time delay (speed of sound = 343 m/s)
    enum double speed_of_sound = 343.0;
    double delay = distance / speed_of_sound;
    int sample_idx = cast(int)((event_time + delay) * sample_rate);
    
    // Stream grain samples directly into output buffer
    int i = 0;
    while (!grain_range.empty()) {
        int write_idx = sample_idx + i;
        
        if (write_idx >= 0 && write_idx < output_buffer.length) {
            double grain_sample = grain_range.front();
            output_buffer[write_idx] += cast(float)(grain_sample * amplitude_scale);
        }
        
        grain_range.popFront();
        i++;
    }
}

// Unit tests
unittest {
    import std.math : approxEqual, abs, sqrt;
    import std.stdio : writeln;
    import mir.random : Random;
    import grain : WaterGrainRange;
    
    writeln("Testing propagation module...");
    
    // Test air_absorption_gain - basic properties
    {
        double freq = 1000.0;
        double distance = 10.0;
        
        double gain = air_absorption_gain(freq, distance);
        
        // Gain should be between 0 and 1
        assert(gain >= 0.0 && gain <= 1.0, "Air absorption gain should be in [0, 1]");
        
        // Very short distance should have minimal absorption
        double gain_short = air_absorption_gain(freq, 0.05);
        assert(gain_short > 0.99, "Short distance should have minimal absorption");
        
        // Higher frequency should have more absorption at same distance
        double gain_low_freq = air_absorption_gain(500.0, distance);
        double gain_high_freq = air_absorption_gain(10000.0, distance);
        assert(gain_high_freq < gain_low_freq, "High frequency should absorb more");
    }
    
    // Test air_absorption_gain - frequency dependence
    {
        double distance = 100.0;  // meters
        
        double gain_1k = air_absorption_gain(1000.0, distance);
        double gain_4k = air_absorption_gain(4000.0, distance);
        double gain_10k = air_absorption_gain(10000.0, distance);
        
        // Should have increasing absorption with frequency
        assert(gain_4k < gain_1k, "4 kHz should absorb more than 1 kHz");
        assert(gain_10k < gain_4k, "10 kHz should absorb more than 4 kHz");
    }
    
    // Test propagate_to_ear - basic functionality
    {
        // Create a simple grain range for testing
        auto rng = Random(777);
        double drop_radius = 0.001;
        int sample_rate = 44100;
        double acoustic_energy = 1e-6;  // Joules
        
        auto grain = WaterGrainRange.create(drop_radius, 10.0, sample_rate, acoustic_energy);
        
        // Setup positions
        Position drop_pos = Position(1.0, 0.0, 0.0);  // 1m away on x-axis
        Position ear_pos = Position(0.0, 0.0, 0.0);   // At origin
        
        // Create output buffer
        float[] output = new float[sample_rate];  // 1 second
        foreach (ref s; output) s = 0.0;
        
        double event_time = 0.1;  // seconds
        double grain_freq = minnaert_frequency(drop_radius);
        
        // Propagate
        propagate_to_ear(grain, grain_freq, drop_pos, ear_pos, 
                        output, event_time, sample_rate);
        
        // Check that some samples are non-zero
        bool has_nonzero = false;
        foreach (s; output) {
            if (abs(s) > 1e-10) {
                has_nonzero = true;
                break;
            }
        }
        assert(has_nonzero, "Propagated signal should have non-zero samples");
        
        // Check that signal appears after delay
        enum double speed_of_sound = 343.0;
        double distance = 1.0;
        double delay = distance / speed_of_sound;
        int delay_samples = cast(int)((event_time + delay) * sample_rate);
        
        // Samples before delay should be zero or very small
        if (delay_samples > 10) {
            double max_before = 0.0;
            foreach (i; 0..delay_samples-10) {
                if (abs(output[i]) > max_before) max_before = abs(output[i]);
            }
            assert(max_before < 1e-8, "Signal should not appear before propagation delay");
        }
    }
    
    // Test geometric attenuation (1/r²)
    {
        auto rng = Random(888);
        double drop_radius = 0.001;
        int sample_rate = 44100;
        double acoustic_energy = 1e-6;
        double event_time = 0.1;
        double grain_freq = minnaert_frequency(drop_radius);
        
        // Test at distance 1m
        auto grain1 = WaterGrainRange.create(drop_radius, 10.0, sample_rate, acoustic_energy);
        Position drop_pos1 = Position(1.0, 0.0, 0.0);
        Position ear_pos = Position(0.0, 0.0, 0.0);
        float[] output1 = new float[sample_rate];
        foreach (ref s; output1) s = 0.0;
        propagate_to_ear(grain1, grain_freq, drop_pos1, ear_pos,
                        output1, event_time, sample_rate);
        
        // Test at distance 2m
        auto grain2 = WaterGrainRange.create(drop_radius, 10.0, sample_rate, acoustic_energy);
        Position drop_pos2 = Position(2.0, 0.0, 0.0);
        float[] output2 = new float[sample_rate];
        foreach (ref s; output2) s = 0.0;
        propagate_to_ear(grain2, grain_freq, drop_pos2, ear_pos,
                        output2, event_time, sample_rate);
        
        // Calculate max amplitude in each case
        double max1 = 0.0;
        foreach (s; output1) if (abs(s) > max1) max1 = abs(s);
        
        double max2 = 0.0;
        foreach (s; output2) if (abs(s) > max2) max2 = abs(s);
        
        // Amplitude should scale as 1/r (so 2m should be ~1/2 of 1m)
        // Note: also includes air absorption, so won't be exactly 1/2
        import std.format : format;
        assert(max2 < max1, 
               format("Greater distance should have lower amplitude: max1=%.6e, max2=%.6e", max1, max2));
        double ratio = max2 / max1;
        assert(ratio > 0.4 && ratio < 0.6, 
               format("Should show 1/r attenuation, ratio=%.4f (expected ~0.5)", ratio));
    }
    
    writeln("  ✓ All propagation tests passed");
}
