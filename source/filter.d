/**
 * Biquad filter design and implementation for grain synthesis
 * 
 * Uses Robert Bristow-Johnson's Audio EQ Cookbook formulas to design
 * bandpass filters directly from center frequency and Q factor (RLC circuit analogy).
 */
module filter;

import std.math : PI, sin, cos, isNaN, isFinite;

/**
 * Biquad filter coefficients (normalized, a0 = 1.0)
 */
struct BiquadCoeffs {
    double b0, b1, b2;  // Numerator coefficients
    double a1, a2;      // Denominator coefficients (a0 normalized to 1.0)
}

/**
 * Design a bandpass biquad filter using RBJ Audio EQ Cookbook formulas
 * 
 * Direct design from center frequency f0 and Q factor (quality factor).
 * RLC circuit analogy: Q = 1/R * sqrt(L/C), f0 = 1/(2π√LC)
 * Bandwidth = f0 / Q
 * 
 * Params:
 *   f0 = Center frequency in Hz
 *   Q = Quality factor (2 = wide/splashy, 20 = narrow/tonal)
 *   sample_rate = Sample rate in Hz
 * 
 * Returns:
 *   Normalized biquad coefficients
 */
BiquadCoeffs design_bandpass_biquad(double f0, double Q, double sample_rate) {
    // Clamp frequency to valid range (avoid Nyquist issues)
    double nyquist = sample_rate / 2.0;
    double f0_clamped = f0;
    
    if (f0_clamped < 20.0) f0_clamped = 20.0;
    if (f0_clamped > nyquist * 0.99) f0_clamped = nyquist * 0.99;
    
    // Clamp Q to reasonable range
    double Q_clamped = Q;
    if (Q_clamped < 0.5) Q_clamped = 0.5;
    if (Q_clamped > 100.0) Q_clamped = 100.0;
    
    // Calculate intermediate values
    double omega = 2.0 * PI * f0_clamped / sample_rate;
    double alpha = sin(omega) / (2.0 * Q_clamped);
    
    // RBJ Bandpass filter (constant 0 dB peak gain)
    double b0 = alpha;
    double b1 = 0.0;
    double b2 = -alpha;
    
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cos(omega);
    double a2 = 1.0 - alpha;
    
    // Normalize by a0
    BiquadCoeffs coeffs;
    coeffs.b0 = b0 / a0;
    coeffs.b1 = b1 / a0;
    coeffs.b2 = b2 / a0;
    coeffs.a1 = a1 / a0;
    coeffs.a2 = a2 / a0;
    
    return coeffs;
}

/**
 * Direct-Form II biquad filter (canonical form with minimal state)
 * 
 * Implements the difference equation:
 *   w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
 *   y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
 * 
 * State variables s1 and s2 store w[n-1] and w[n-2] respectively.
 */
struct BiquadFilter {
    BiquadCoeffs coeffs;
    double s1 = 0.0;  // w[n-1]
    double s2 = 0.0;  // w[n-2]
    
    /**
     * Process a single sample through the filter
     */
    double process(double input) {
        // Direct-Form II
        double w = input - coeffs.a1 * s1 - coeffs.a2 * s2;
        double output = coeffs.b0 * w + coeffs.b1 * s1 + coeffs.b2 * s2;
        
        // Update state
        s2 = s1;
        s1 = w;
        
        return output;
    }
    
    /**
     * Reset filter state to zero
     */
    void reset() {
        s1 = 0.0;
        s2 = 0.0;
    }
}

/**
 * Filter an entire array of samples
 * 
 * Params:
 *   input = Input signal
 *   coeffs = Biquad filter coefficients
 * 
 * Returns:
 *   Filtered signal
 */
double[] filter_signal(const double[] input, BiquadCoeffs coeffs) {
    auto filter = BiquadFilter(coeffs);
    double[] output = new double[input.length];
    
    foreach (i, sample; input) {
        output[i] = filter.process(sample);
    }
    
    return output;
}

// Unit tests
unittest {
    import std.math : approxEqual, abs, sin, PI, isFinite, isNaN;
    import std.stdio : writeln;
    import std.algorithm : maxElement;
    
    writeln("Testing filter module...");
    
    // Test design_bandpass_biquad - basic coefficient properties
    {
        auto coeffs = design_bandpass_biquad(1000.0, 5.0, 44100.0);
        
        // Coefficients should be finite and non-NaN
        assert(isFinite(coeffs.b0) && !isNaN(coeffs.b0), "b0 should be finite");
        assert(isFinite(coeffs.b1) && !isNaN(coeffs.b1), "b1 should be finite");
        assert(isFinite(coeffs.b2) && !isNaN(coeffs.b2), "b2 should be finite");
        assert(isFinite(coeffs.a1) && !isNaN(coeffs.a1), "a1 should be finite");
        assert(isFinite(coeffs.a2) && !isNaN(coeffs.a2), "a2 should be finite");
        
        // For bandpass: b1 should be 0
        assert(approxEqual(coeffs.b1, 0.0, 1e-10), "Bandpass b1 coefficient should be 0");
        
        // b0 and b2 should have opposite signs for bandpass
        assert(coeffs.b0 * coeffs.b2 < 0, "b0 and b2 should have opposite signs");
    }
    
    // Test frequency clamping
    {
        double sample_rate = 44100.0;
        
        // Very low frequency should be clamped to 20 Hz
        auto coeffs_low = design_bandpass_biquad(1.0, 5.0, sample_rate);
        assert(isFinite(coeffs_low.b0), "Low frequency clamped coefficients should be valid");
        
        // Very high frequency should be clamped below Nyquist
        auto coeffs_high = design_bandpass_biquad(30000.0, 5.0, sample_rate);
        assert(isFinite(coeffs_high.b0), "High frequency clamped coefficients should be valid");
    }
    
    // Test Q factor clamping
    {
        // Very low Q should be clamped to 0.5
        auto coeffs_low_q = design_bandpass_biquad(1000.0, 0.1, 44100.0);
        assert(isFinite(coeffs_low_q.b0), "Low Q clamped coefficients should be valid");
        
        // Very high Q should be clamped to 100
        auto coeffs_high_q = design_bandpass_biquad(1000.0, 200.0, 44100.0);
        assert(isFinite(coeffs_high_q.b0), "High Q clamped coefficients should be valid");
    }
    
    // Test BiquadFilter - impulse response
    {
        auto coeffs = design_bandpass_biquad(1000.0, 10.0, 44100.0);
        auto filter = BiquadFilter(coeffs);
        
        // Feed impulse (1.0 followed by zeros)
        double[] impulse_response;
        impulse_response ~= filter.process(1.0);
        foreach (_; 0..100) {
            impulse_response ~= filter.process(0.0);
        }
        
        // First sample should be non-zero
        assert(abs(impulse_response[0]) > 0.0, "Impulse response should have non-zero first sample");
        
        // Response should eventually decay
        double max_early = 0.0;
        foreach (i; 0..10) {
            if (abs(impulse_response[i]) > max_early) max_early = abs(impulse_response[i]);
        }
        double max_late = 0.0;
        foreach (i; 50..100) {
            if (abs(impulse_response[i]) > max_late) max_late = abs(impulse_response[i]);
        }
        assert(max_late < max_early, "Filter response should decay");
    }
    
    // Test BiquadFilter - DC and high frequency rejection
    {
        auto coeffs = design_bandpass_biquad(1000.0, 5.0, 44100.0);
        auto filter = BiquadFilter(coeffs);
        
        // Feed DC signal (all ones) - should be rejected
        filter.reset();
        double dc_output = 0.0;
        foreach (_; 0..1000) {
            dc_output = filter.process(1.0);
        }
        import std.format : format;
        assert(abs(dc_output) < 0.1, 
               format("DC should be rejected by bandpass filter, got %.6f", dc_output));
        
        // Reset for next test
        filter.reset();
        assert(approxEqual(filter.s1, 0.0) && approxEqual(filter.s2, 0.0), 
               "Filter reset should clear state");
    }
    
    // Test filter_signal - batch processing
    {
        auto coeffs = design_bandpass_biquad(1000.0, 5.0, 44100.0);
        
        // Create test signal: impulse
        double[] input = new double[100];
        input[0] = 1.0;
        
        double[] output = filter_signal(input, coeffs);
        
        // Output should have same length as input
        assert(output.length == input.length, "Output length should match input");
        
        // First sample should be non-zero
        assert(abs(output[0]) > 0.0, "Filtered impulse should have non-zero first sample");
    }
    
    // Test filter at center frequency (should pass signal)
    {
        double f0 = 1000.0;
        double sample_rate = 44100.0;
        auto coeffs = design_bandpass_biquad(f0, 5.0, sample_rate);
        auto filter = BiquadFilter(coeffs);
        
        // Generate sine wave at center frequency
        double[] sine_wave;
        foreach (i; 0..441) {  // 10ms at 44.1kHz
            double t = i / sample_rate;
            sine_wave ~= sin(2.0 * PI * f0 * t);
        }
        
        // Filter the sine wave
        double[] filtered;
        filter.reset();
        foreach (sample; sine_wave) {
            filtered ~= filter.process(sample);
        }
        
        // After settling (skip first 50 samples), output should be non-zero
        double max_output = 0.0;
        foreach (i; 50..filtered.length) {
            if (abs(filtered[i]) > max_output) max_output = abs(filtered[i]);
        }
        assert(max_output > 0.1, "Filter should pass signal at center frequency");
    }
    
    writeln("  ✓ All filter tests passed");
}
