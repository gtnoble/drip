/**
 * Grain synthesis for raindrop sounds using streaming input ranges
 * 
 * Implements three synthesis methods as D input ranges (zero allocations):
 * - Water surface: Unit impulse → biquad bandpass (WaterGrainRange)
 * - Pink noise: Voss-McCartney cascade → Gaussian window (PinkNoiseGrainRange)
 * - White noise: Gaussian noise → Gaussian window (WhiteNoiseGrainRange)
 * 
 * All grains yield samples on-demand with analytical energy normalization.
 */
module grain;

import std.math : sqrt, exp, PI;
import std.algorithm : min;
import mir.random : Random;
import mir.random.variable : normalVar;

import filter;
import physics;
import surface;

/**
 * Infinite pink noise generator using Voss-McCartney algorithm
 * 
 * Cascades three first-order lowpass filters at octave-spaced cutoffs
 * to approximate 1/f power spectrum. Implements D input range interface.
 */
struct VossMcCartneyRange {
    double[3] states;   // Filter states
    double[3] alphas;   // Filter coefficients (one-pole lowpass)
    Random* rng;
    
    /**
     * Create pink noise generator
     * 
     * Params:
     *   sample_rate = Sample rate in Hz
     *   rng = Random number generator (must outlive this range)
     */
    static VossMcCartneyRange create(int sample_rate, ref Random rng) {
        VossMcCartneyRange range;
        range.rng = &rng;
        double nyquist = sample_rate / 2.0;
        
        // Three poles at octave intervals (Nyquist/2, Nyquist/4, Nyquist/8)
        foreach (k; 1..4) {
            double f_cutoff = nyquist / (1 << k);  // 2^k
            double rc = 1.0 / (2.0 * PI * f_cutoff);
            range.alphas[k-1] = 1.0 / (1.0 + rc * sample_rate);
            range.states[k-1] = 0.0;
        }
        return range;
    }
    
    enum bool empty = false;  // Infinite range
    
    @property double front() {
        auto normal_dist = normalVar!double(0.0, 1.0);
        double output = normal_dist(*rng);
        
        // Cascade three first-order lowpass filters
        foreach (i; 0..3) {
            states[i] = (1.0 - alphas[i]) * states[i] + alphas[i] * output;
            output = states[i];
        }
        
        return output;
    }
    
    void popFront() { /* State updated in front() */ }
}

/**
 * Water grain generator (streaming range)
 * 
 * Generates unit impulse → biquad filter in a single pass with no allocations.
 * Unit impulse energy is physically realistic (impulse = drop impact).
 */
struct WaterGrainRange {
    BiquadFilter filter;
    int n_samples;
    int samples_emitted;
    double impulse_amplitude;
    bool first_sample;
    
    /**
     * Create water grain range
     * 
     * Params:
     *   drop_radius = Drop radius in meters
     *   Q = Quality factor (2=splashy, 20=tonal)
     *   sample_rate = Sample rate in Hz
     *   acoustic_energy = Acoustic energy in Joules
     *   surface_type = Surface type enumerator (water, capillary) for different resonance types (default: water)
     */
    static WaterGrainRange create(double drop_radius, double Q, int sample_rate, double acoustic_energy, SurfaceType surface_type = SurfaceType.water) {
        WaterGrainRange range;
        
        // Calculate frequency based on surface type
        double freq;
        final switch (surface_type) {
            case SurfaceType.capillary:
                freq = capillary_frequency(drop_radius);
                break;
            case SurfaceType.water:
            case SurfaceType.pink_noise:
            case SurfaceType.white_noise:
                freq = minnaert_frequency(drop_radius);
                break;
        }
        
        // Duration based on decay time (Q/f)
        double decay_time = Q / freq;
        double duration = min(4.0 * decay_time, 0.5);  // Cap at 0.5 seconds
        range.n_samples = cast(int)(duration * sample_rate);
        if (range.n_samples < 1) range.n_samples = 1;
        
        // Design biquad filter
        auto coeffs = design_bandpass_biquad(freq, Q, sample_rate);
        range.filter = BiquadFilter(coeffs);
        
        // Set impulse amplitude based on acoustic energy
        // Impulse energy = amplitude^2, so amplitude = sqrt(energy)
        // Filter will shape this impulse but not normalize it
        range.impulse_amplitude = sqrt(acoustic_energy);
        range.samples_emitted = 0;
        range.first_sample = true;
        
        return range;
    }
    
    bool empty() const {
        return samples_emitted >= n_samples;
    }
    
    @property double front() {
        if (first_sample) {
            // Impulse scaled by acoustic energy on first sample
            return filter.process(impulse_amplitude);
        } else {
            // Zero input for remaining samples (just filter ringing)
            return filter.process(0.0);
        }
    }
    
    void popFront() {
        first_sample = false;
        samples_emitted++;
    }
}

/**
 * Pink noise grain generator (streaming range)
 * 
 * Generates pink noise → Gaussian window with analytical energy normalization.
 * Single-pass streaming, no buffer allocation.
 */
struct PinkNoiseGrainRange {
    VossMcCartneyRange pink_source;
    int n_samples;
    int samples_emitted;
    double t_center, sigma;
    double sample_rate;
    double amplitude_scale;
    
    /**
     * Create pink noise grain range
     * 
     * Params:
     *   drop_radius = Drop radius in meters
     *   sample_rate = Sample rate in Hz
     *   rng = Random number generator
     *   acoustic_energy = Acoustic energy in Joules
     */
    static PinkNoiseGrainRange create(double drop_radius, int sample_rate, ref Random rng, double acoustic_energy) {
        PinkNoiseGrainRange range;
        
        // Calculate contact time from drop physics
        double diameter = 2.0 * drop_radius;
        double velocity = terminal_velocity(drop_radius);
        double contact_time = diameter / velocity;
        
        // Duration = 4 × contact_time, capped at 0.1 seconds
        double duration = min(4.0 * contact_time, 0.1);
        range.n_samples = cast(int)(duration * sample_rate);
        if (range.n_samples < 1) range.n_samples = 1;
        
        // Gaussian window parameters
        range.t_center = duration / 2.0;
        range.sigma = duration / 6.0;
        range.sample_rate = sample_rate;
        range.samples_emitted = 0;
        
        // Create pink noise source
        range.pink_source = VossMcCartneyRange.create(sample_rate, rng);
        
        // Calculate analytical window energy: Σw[n]²
        // For Gaussian window: w[n] = exp(-((n/fs - tc)²)/(2σ²))
        double window_energy = 0.0;
        foreach (i; 0 .. range.n_samples) {
            double t = cast(double)i / sample_rate;
            double w = exp(-((t - range.t_center) * (t - range.t_center)) / (2.0 * range.sigma * range.sigma));
            window_energy += w * w;
        }
        
        // Amplitude scale: acoustic energy divided by window energy
        // Pink noise variance ≈ 1.0, total energy = variance × window_energy
        // To get acoustic_energy: scale = sqrt(acoustic_energy / window_energy)
        range.amplitude_scale = sqrt(acoustic_energy / window_energy);
        
        return range;
    }
    
    bool empty() const {
        return samples_emitted >= n_samples;
    }
    
    @property double front() {
        // Get pink noise sample
        double pink_sample = pink_source.front();
        pink_source.popFront();
        
        // Apply Gaussian window
        double t = cast(double)samples_emitted / sample_rate;
        double window = exp(-((t - t_center) * (t - t_center)) / (2.0 * sigma * sigma));
        
        // Return windowed and scaled sample
        return pink_sample * window * amplitude_scale;
    }
    
    void popFront() {
        samples_emitted++;
    }
}

/**
 * White noise grain generator (streaming range)
 * 
 * Generates Gaussian noise → Gaussian window with analytical energy normalization.
 * Single-pass streaming, no buffer allocation.
 */
struct WhiteNoiseGrainRange {
    Random* rng;
    int n_samples;
    int samples_emitted;
    double t_center, sigma;
    double sample_rate;
    double amplitude_scale;
    
    /**
     * Create white noise grain range
     * 
     * Params:
     *   drop_radius = Drop radius in meters
     *   sample_rate = Sample rate in Hz
     *   rng = Random number generator
     *   acoustic_energy = Acoustic energy in Joules
     */
    static WhiteNoiseGrainRange create(double drop_radius, int sample_rate, ref Random rng, double acoustic_energy) {
        WhiteNoiseGrainRange range;
        range.rng = &rng;
        
        // Calculate contact time from drop physics
        double diameter = 2.0 * drop_radius;
        double velocity = terminal_velocity(drop_radius);
        double contact_time = diameter / velocity;
        
        // Duration = 4 × contact_time, capped at 0.1 seconds
        double duration = min(4.0 * contact_time, 0.1);
        range.n_samples = cast(int)(duration * sample_rate);
        if (range.n_samples < 1) range.n_samples = 1;
        
        // Gaussian window parameters
        range.t_center = duration / 2.0;
        range.sigma = duration / 6.0;
        range.sample_rate = sample_rate;
        range.samples_emitted = 0;
        
        // Calculate analytical window energy
        double window_energy = 0.0;
        foreach (i; 0 .. range.n_samples) {
            double t = cast(double)i / sample_rate;
            double w = exp(-((t - range.t_center) * (t - range.t_center)) / (2.0 * range.sigma * range.sigma));
            window_energy += w * w;
        }
        
        // Amplitude scale: acoustic energy divided by window energy
        // White noise variance = 1.0, total energy = window_energy
        // To get acoustic_energy: scale = sqrt(acoustic_energy / window_energy)
        range.amplitude_scale = sqrt(acoustic_energy / window_energy);
        
        return range;
    }
    
    bool empty() const {
        return samples_emitted >= n_samples;
    }
    
    @property double front() {
        // Generate white noise sample
        auto normal_dist = normalVar!double(0.0, 1.0);
        double white_sample = normal_dist(*rng);
        
        // Apply Gaussian window
        double t = cast(double)samples_emitted / sample_rate;
        double window = exp(-((t - t_center) * (t - t_center)) / (2.0 * sigma * sigma));
        
        // Return windowed and scaled sample
        return white_sample * window * amplitude_scale;
    }
    
    void popFront() {
        samples_emitted++;
    }
}

// Unit tests
unittest {
    import std.math : approxEqual, abs, sqrt;
    import std.stdio : writeln;
    import mir.random : Random;
    
    writeln("Testing grain module...");
    
    // Test VossMcCartneyRange - infinite pink noise
    {
        auto rng = Random(12345);
        auto pink = VossMcCartneyRange.create(44100, rng);
        
        // Range should never be empty
        assert(!pink.empty, "Pink noise range should be infinite");
        
        // Generate some samples
        double[] samples;
        foreach (_; 0..1000) {
            assert(!pink.empty, "Pink noise should never be empty");
            samples ~= pink.front;
            pink.popFront();
        }
        
        // Samples should have reasonable variance (not all zeros)
        double mean = 0.0;
        foreach (s; samples) mean += s;
        mean /= samples.length;
        
        double variance = 0.0;
        foreach (s; samples) variance += (s - mean) * (s - mean);
        variance /= samples.length;
        
        import std.format : format;
        assert(variance > 0.01, 
               format("Pink noise should have non-trivial variance, got %.6f", variance));
        assert(variance < 10.0, 
               format("Pink noise variance should be reasonable, got %.6f", variance));
    }
    
    // Test WaterGrainRange - basic properties
    {
        double drop_radius = 0.001;  // 1mm
        double Q = 10.0;
        int sample_rate = 44100;
        double acoustic_energy = 1e-6;  // 1 microjoule
        
        auto grain = WaterGrainRange.create(drop_radius, Q, sample_rate, acoustic_energy);
        
        // Should have finite length
        assert(!grain.empty, "New grain should not be empty");
        assert(grain.n_samples > 0, "Grain should have positive length");
        
        // Collect all samples
        double[] samples;
        while (!grain.empty) {
            samples ~= grain.front;
            grain.popFront();
        }
        
        assert(grain.empty, "Grain should be empty after consuming all samples");
        assert(samples.length == grain.n_samples, "Should emit expected number of samples");
        
        // First sample should be non-zero (impulse response)
        assert(abs(samples[0]) > 0.0, "First sample should be non-zero");
        
        // Samples should eventually decay
        if (samples.length > 10) {
            double max_early = 0.0;
            foreach (i; 0..10) {
                if (abs(samples[i]) > max_early) max_early = abs(samples[i]);
            }
            double max_late = 0.0;
            size_t late_start = samples.length / 2;
            foreach (i; late_start..samples.length) {
                if (abs(samples[i]) > max_late) max_late = abs(samples[i]);
            }
            assert(max_late < max_early, "Water grain should decay");
        }
    }
    
    // Test PinkNoiseGrainRange - energy normalization
    {
        auto rng = Random(54321);
        double drop_radius = 0.002;  // 2mm
        int sample_rate = 44100;
        double acoustic_energy = 1e-6;  // 1 microjoule
        
        auto grain = PinkNoiseGrainRange.create(drop_radius, sample_rate, rng, acoustic_energy);
        
        // Should have finite length
        assert(!grain.empty, "New pink grain should not be empty");
        
        // Collect samples and measure energy
        double[] samples;
        while (!grain.empty) {
            samples ~= grain.front;
            grain.popFront();
        }
        
        assert(samples.length > 0, "Pink grain should produce samples");
        
        // Calculate energy
        double energy = 0.0;
        foreach (s; samples) energy += s * s;
        
        // Energy should be close to acoustic_energy (1e-6 J)
        // Allow some variance due to random nature of pink noise
        import std.format : format;
        assert(energy > 0.0, 
               format("Pink grain should have non-zero energy, got %.6e", energy));
        assert(energy > acoustic_energy * 0.1 && energy < acoustic_energy * 10.0, 
               format("Pink grain energy should be ~%.6e J, got %.6e", acoustic_energy, energy));
    }
    
    // Test WhiteNoiseGrainRange - energy normalization
    {
        auto rng = Random(99999);
        double drop_radius = 0.0015;  // 1.5mm
        int sample_rate = 44100;
        double acoustic_energy = 1e-6;  // 1 microjoule
        
        auto grain = WhiteNoiseGrainRange.create(drop_radius, sample_rate, rng, acoustic_energy);
        
        // Should have finite length
        assert(!grain.empty, "New white grain should not be empty");
        
        // Collect samples and measure energy
        double[] samples;
        while (!grain.empty) {
            samples ~= grain.front;
            grain.popFront();
        }
        
        assert(samples.length > 0, "White grain should produce samples");
        
        // Calculate energy
        double energy = 0.0;
        foreach (s; samples) energy += s * s;
        
        // Energy should be close to acoustic_energy (1e-6 J)
        import std.format : format;
        assert(energy > 0.0, 
               format("White grain should have non-zero energy, got %.6e", energy));
        assert(energy > acoustic_energy * 0.1 && energy < acoustic_energy * 10.0,
               format("White grain energy should be ~%.6e J, got %.6e", acoustic_energy, energy));
    }
    
    // Test grain duration consistency
    {
        auto rng = Random(11111);
        double radius = 0.001;
        int sample_rate = 44100;
        double acoustic_energy = 1e-6;  // 1 microjoule
        
        // Water grain
        auto water_grain = WaterGrainRange.create(radius, 10.0, sample_rate, acoustic_energy);
        double water_duration = calculate_impulse_duration(radius, SurfaceType.water, 10.0);
        int expected_water_samples = cast(int)(water_duration * sample_rate);
        assert(water_grain.n_samples == expected_water_samples, 
               "Water grain duration should match physics calculation");
        
        // Pink noise grain
        auto pink_grain = PinkNoiseGrainRange.create(radius, sample_rate, rng, acoustic_energy);
        double pink_duration = calculate_impulse_duration(radius, SurfaceType.pink_noise);
        int expected_pink_samples = cast(int)(pink_duration * sample_rate);
        assert(pink_grain.n_samples == expected_pink_samples,
               "Pink grain duration should match physics calculation");
        
        // White noise grain
        auto white_grain = WhiteNoiseGrainRange.create(radius, sample_rate, rng, acoustic_energy);
        double white_duration = calculate_impulse_duration(radius, SurfaceType.white_noise);
        int expected_white_samples = cast(int)(white_duration * sample_rate);
        assert(white_grain.n_samples == expected_white_samples,
               "White grain duration should match physics calculation");
    }
    
    writeln("  ✓ All grain tests passed");
}
