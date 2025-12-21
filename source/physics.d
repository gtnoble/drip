/**
 * Physics calculations for raindrop acoustics
 * 
 * Implements Minnaert resonance, terminal velocity, acoustic energy,
 * psychoacoustic weighting (A-weighting), and hearing thresholds.
 */
module physics;

import std.math : sqrt, exp, log10, PI, pow;
import mir.random : Random;
import mir.random.variable : exponentialVar;
import surface;

enum rho_water = 1000.0;  // kg/m³
enum rho_air = 1.225;     // kg/m³
enum water_surface_tension = 0.072;  // N/m
enum c_air = 343.0;          // Speed of sound in air (m/s)
enum c_water = 1482.0;       // Speed of sound in water (m/s) at 20°C
enum air_polytopic_index = 1.4; // For adiabatic processes in air
enum p_atm = 101325.0;      // Atmospheric pressure in Pascals
// enum water_viscosity = 0.0008891; // Pa·s (dynamic viscosity of water at 25°C)
enum water_viscosity = 1.0; // Pa·s (dynamic viscosity of water at 25°C)

/**
 * Calculate Minnaert resonance frequency for air bubble
 * 
 * Simplified formula: f = 3.26 / R_bubble (Hz)
 * 
 * Params:
 *   drop_radius = Drop radius in meters
 * 
 * Returns:
 *   Frequency in Hz
 */
double minnaert_frequency(double drop_radius) {
    // Bubble radius ≈ 0.7 × drop radius (Newton's impact depth approximation)
    double bubble_radius = 0.7 * drop_radius;
    return 3.26 / bubble_radius;
}

double sphere_surface_area(double radius) {
    return 4.0 * PI * radius ^^ 2;
}

double sphere_volume(double radius) {
    return (4.0/3.0) * PI * radius ^^ 3;
}

double minnaert_quality_factor(double drop_radius) {
    double acoustic_radiation_resistance = rho_water * c_water;
    double viscous_resistance = water_viscosity / sphere_volume(drop_radius);
    double compression_mechanical_capacitance = sphere_volume(drop_radius) / (air_polytopic_index * p_atm);
    double displacement_mechanical_inductance = rho_water / (4 * PI * drop_radius);
    double q_minnaert = 
        sqrt(displacement_mechanical_inductance / compression_mechanical_capacitance) / 
        (acoustic_radiation_resistance + viscous_resistance);
    return q_minnaert;
}

/**
 * Calculate capillary wave resonance frequency from surface tension
 * 
 * f = sqrt(σ / m) where σ is surface tension and m is drop mass
 * 
 * Params:
 *   drop_radius = Drop radius in meters
 * 
 * Returns:
 *   Frequency in Hz
 */
double capillary_frequency(double drop_radius) {
    enum double surface_tension = 0.072;  // N/m (water-air interface at 20°C)
    
    // Drop mass
    double volume = (4.0/3.0) * PI * drop_radius^^3;
    double mass = volume * rho_water;
    
    // Capillary frequency
    return sqrt(surface_tension / mass);
}

double capillary_quality_factor(double drop_radius) {
    double mass_mechanical_inductance = (4.0/3.0) * PI * drop_radius^^3 * rho_water; // kg
    double tension_mechanical_capacitance = 1.0 / water_surface_tension;
    double acoustic_radiation_resistance = rho_air * c_air * 2.0 * PI * drop_radius^^2;
    double Q_capillary = 
        sqrt(mass_mechanical_inductance / tension_mechanical_capacitance) / 
        acoustic_radiation_resistance;
    return Q_capillary;
}

/**
 * Calculate terminal velocity of raindrop in air
 * 
 * Simplified model assuming spherical drops with constant drag coefficient.
 * Based on balance between gravitational and drag forces:
 * v_terminal = sqrt((8/3) * (ρ_water/ρ_air) * g * R / C_d)
 * 
 * Params:
 *   drop_radius = Drop radius in meters
 * 
 * Returns:
 *   Velocity in m/s
 */
double terminal_velocity(double drop_radius) {
    enum double rho_water = 1000.0;  // kg/m³
    enum double rho_air = 1.225;     // kg/m³
    enum double g = 9.81;            // m/s²
    enum double C_d = 0.47;          // Drag coefficient for sphere
    
    double v_terminal = sqrt((8.0/3.0) * (rho_water/rho_air) * g * drop_radius / C_d);
    return v_terminal;
}

/**
 * Calculate kinetic energy of falling raindrop
 * 
 * E_kinetic = 0.5 * m * v²
 * 
 * Params:
 *   drop_radius = Drop radius in meters
 * 
 * Returns:
 *   Kinetic energy in Joules
 */
double calculate_kinetic_energy(double drop_radius) {
    enum double rho_water = 1000.0;  // kg/m³
    
    // Drop mass
    double volume = (4.0/3.0) * PI * drop_radius^^3;
    double mass = volume * rho_water;
    
    // Terminal velocity
    double velocity = terminal_velocity(drop_radius);
    
    // Kinetic energy
    double E_kinetic = 0.5 * mass * velocity^^2;
    
    return E_kinetic;
}

/**
 * Calculate acoustic energy from drop kinetic energy
 * 
 * Params:
 *   drop_radius = Drop radius in meters
 *   acoustic_efficiency = Fraction of kinetic energy converted to sound (default: 0.001)
 * 
 * Returns:
 *   Acoustic energy in Joules
 */
double acoustic_amplitude(double drop_radius, double acoustic_efficiency = 0.001) {
    double E_kinetic = calculate_kinetic_energy(drop_radius);
    double E_acoustic = acoustic_efficiency * E_kinetic;
    return E_acoustic;
}

/**
 * Calculate A-weighting in dB (IEC 61672-1:2013)
 * 
 * Standard frequency weighting curve used in sound level measurement.
 * Normalized to 0 dB at 1 kHz.
 * 
 * Params:
 *   frequency = Frequency in Hz
 * 
 * Returns:
 *   A-weighting value in dB
 *   
 * Reference values for validation:
 *   100 Hz: -19.1 dB
 *   1000 Hz: 0.0 dB
 *   3000 Hz: +1.2 dB
 *   10000 Hz: -0.8 dB
 */
double a_weighting_db(double frequency) {
    double f = frequency;
    
    // IEC 61672-1:2013 reference frequencies
    enum double f1 = 20.598997;
    enum double f2 = 107.65265;
    enum double f3 = 737.86223;
    enum double f4 = 12194.217;
    
    double numerator = f4^^2 * f^^4;
    double denominator = ((f^^2 + f1^^2) * 
                         sqrt((f^^2 + f2^^2) * (f^^2 + f3^^2)) * 
                         (f^^2 + f4^^2));
    
    double A = numerator / denominator;
    return 20.0 * log10(A) + 2.0;
}

/**
 * Calculate approximate audibility threshold using A-weighting
 * 
 * Uses A-weighting curve anchored to typical absolute threshold at 1 kHz.
 * Anchor point: ~4 dB SPL at 1 kHz (typical absolute threshold)
 * Formula: threshold(f) = 4 dB - A_weighting(f)
 * 
 * Params:
 *   frequency = Frequency in Hz
 * 
 * Returns:
 *   Threshold in dB SPL (re: 20 µPa)
 */
double hearing_threshold_db_spl(double frequency) {
    if (frequency < 20.0 || frequency > 20000.0) {
        return 1000.0;  // Effectively inaudible outside human range
    }
    
    // Reference threshold at 1 kHz is approximately 4 dB SPL
    // Apply inverse A-weighting to get frequency-dependent threshold
    return 4.0 - a_weighting_db(frequency);
}

/**
 * Calculate absolute threshold of hearing in Pascals
 * 
 * Params:
 *   frequency = Frequency in Hz
 * 
 * Returns:
 *   Threshold pressure in Pascals (Pa)
 */
double hearing_threshold_pa(double frequency) {
    double threshold_db = hearing_threshold_db_spl(frequency);
    enum double p_ref = 20e-6;  // Reference pressure: 20 micropascals
    return p_ref * pow(10.0, threshold_db / 20.0);
}

/**
 * Calculate Marshall-Palmer slope parameter
 * 
 * Used for tuning M-H sampler proposal scales.
 * Λ = 4.1 * R^(-0.21) where R is rain rate in mm/hr
 * 
 * Params:
 *   rain_rate_mm_hr = Rainfall rate in mm/hour
 * 
 * Returns:
 *   Slope parameter Λ (units: 1/mm)
 */
double marshall_palmer_lambda(double rain_rate_mm_hr) {
    return 4.1 * pow(rain_rate_mm_hr, -0.21);
}

/**
 * Get dominant frequency for a drop based on surface type
 * 
 * For noise surface types (pink_noise, white_noise), returns the frequency
 * of maximum human hearing sensitivity (~3.5 kHz) since the energy content
 * is broadband. This ensures conservative audibility filtering.
 * 
 * Params:
 *   drop_radius = Drop radius in meters
 *   surface_type = Surface type enumerator (water, capillary, pink_noise, white_noise)
 * 
 * Returns:
 *   Dominant frequency in Hz
 */
double get_dominant_frequency(double drop_radius, SurfaceType surface_type) {
    final switch (surface_type) {
        case SurfaceType.pink_noise:
        case SurfaceType.white_noise:
            // Use frequency of peak hearing sensitivity for conservative audibility check
            return 3_500.0;
        case SurfaceType.capillary:
            // Capillary wave resonance from surface tension
            return capillary_frequency(drop_radius);
        case SurfaceType.water:
            // Water surface: use Minnaert frequency
            return minnaert_frequency(drop_radius);
    }
}

/**
 * Calculate acoustic impulse duration based on surface type
 * 
 * Params:
 *   drop_radius = Drop radius in meters
 *   surface_type = "water", "pink_noise", or "white_noise"
 *   Q = Quality factor (only used for water surfaces)
 * 
 * Returns:
 *   Duration in seconds
 */
double calculate_impulse_duration(double drop_radius, SurfaceType surface_type, double Q = 10.0) {
    final switch (surface_type) {
        case SurfaceType.pink_noise:
        case SurfaceType.white_noise:
            // Contact time: t_contact = diameter / velocity
            double diameter = 2.0 * drop_radius;
            double velocity = terminal_velocity(drop_radius);
            double contact_time = diameter / velocity;
            
            // Duration = 4 × contact_time, capped at 0.1 seconds
            double duration = 4.0 * contact_time;
            if (duration > 0.1) duration = 0.1;
            return duration;
        case SurfaceType.water:
        case SurfaceType.capillary:
            // Water surface or capillary: decay time from Q factor
            double frequency;
            if (surface_type == SurfaceType.capillary) {
                frequency = capillary_frequency(drop_radius);
            } else {
                frequency = minnaert_frequency(drop_radius);
            }
            double decay_time = Q / frequency;
            
            // Duration = 4 × decay_time, capped at 0.5 seconds
            double water_duration = 4.0 * decay_time;
            if (water_duration > 0.5) water_duration = 0.5;
            return water_duration;
    }
}

/**
 * Get quality factor for a drop based on surface type
 * 
 * Returns physics-based quality factor for water and capillary surfaces.
 * For noise surfaces, returns 0 as Q is not used in noise generation.
 * 
 * Params:
 *   drop_radius = Drop radius in meters
 *   surface_type = Surface type enumerator
 * 
 * Returns:
 *   Quality factor calculated from physical principles
 */
double get_quality_factor(double drop_radius, SurfaceType surface_type) {
    final switch (surface_type) {
        case SurfaceType.water:
            return minnaert_quality_factor(drop_radius);
        case SurfaceType.capillary:
            return capillary_quality_factor(drop_radius);
        case SurfaceType.pink_noise:
        case SurfaceType.white_noise:
            return 0.0;  // Q not used for noise sources
    }
}

// Unit tests
unittest {
    import std.math : approxEqual, abs;
    import std.stdio : writeln;
    
    writeln("Testing physics module...");
    
    // Test minnaert_frequency
    {
        // 1mm radius drop (2mm diameter) should produce ~2326 Hz
        double freq_1mm = minnaert_frequency(0.001);
        assert(approxEqual(freq_1mm, 4657.0, 0.01), "1mm drop Minnaert frequency incorrect");
        
        // Larger drop (2mm radius) should have lower frequency
        double freq_2mm = minnaert_frequency(0.002);
        assert(freq_2mm < freq_1mm, "Larger drops should have lower frequency");
        assert(approxEqual(freq_2mm, 2328.5, 0.01), "2mm drop Minnaert frequency incorrect");
    }
    
    // Test terminal_velocity
    {
        // Small drop (1mm radius) should fall slower than large drop
        double v_small = terminal_velocity(0.001);
        double v_large = terminal_velocity(0.003);
        assert(v_large > v_small, "Larger drops should fall faster");
        
        // Velocity should be positive and realistic (< 20 m/s for raindrops)
        import std.format : format;
        assert(v_small > 0.0 && v_small < 20.0, 
               format("Small drop velocity unrealistic: %.2f m/s", v_small));
        assert(v_large > 0.0 && v_large < 20.0, 
               format("Large drop velocity unrealistic: %.2f m/s", v_large));
    }
    
    // Test acoustic_amplitude
    {
        // Energy should increase with drop size
        double E_small = acoustic_amplitude(0.001);
        double E_large = acoustic_amplitude(0.003);
        assert(E_large > E_small, "Larger drops should have more acoustic energy");
        
        // Energy should be positive and finite
        assert(E_small > 0.0, "Acoustic energy should be positive");
        assert(E_large > 0.0, "Acoustic energy should be positive");
    }
    
    // Test a_weighting_db (IEC 61672-1:2013)
    {
        // A-weighting should be 0 dB at 1 kHz
        import std.format : format;
        double weight_1k = a_weighting_db(1000.0);
        assert(abs(weight_1k) < 0.01, 
               format("A-weighting at 1 kHz should be ~0 dB, got %.6f", weight_1k));
        
        // A-weighting at 100 Hz should be approximately -19.1 dB
        double weight_100 = a_weighting_db(100.0);
        assert(abs(weight_100 - (-19.1)) < 1.0, "A-weighting at 100 Hz incorrect");
        
        // A-weighting should have peak near 3-4 kHz
        double weight_3k = a_weighting_db(3000.0);
        assert(weight_3k > weight_1k, "A-weighting should peak above 1 kHz");
    }
    
    // Test hearing_threshold_db_spl
    {
        // Threshold should be lowest (most sensitive) around 3-4 kHz
        double thresh_1k = hearing_threshold_db_spl(1000.0);
        double thresh_4k = hearing_threshold_db_spl(4000.0);
        assert(thresh_4k < thresh_1k, "Hearing should be most sensitive ~4 kHz");
        
        // Out of range frequencies should return high threshold
        double thresh_low = hearing_threshold_db_spl(10.0);
        double thresh_high = hearing_threshold_db_spl(25000.0);
        assert(thresh_low > 100.0, "Sub-audible frequencies should have high threshold");
        assert(thresh_high > 100.0, "Ultrasonic frequencies should have high threshold");
    }
    
    // Test marshall_palmer_lambda
    {
        // Higher rain rate should give smaller lambda (larger drops)
        double lambda_light = marshall_palmer_lambda(5.0);
        double lambda_heavy = marshall_palmer_lambda(50.0);
        assert(lambda_heavy < lambda_light, "Heavy rain should have smaller lambda");
        
        // Lambda should be positive
        assert(lambda_light > 0.0, "Lambda should be positive");
    }
    
    // Test get_dominant_frequency
    {
        double radius = 0.001;  // 1mm
        
        // Water surface should use Minnaert frequency
        double freq_water = get_dominant_frequency(radius, SurfaceType.water);
        assert(approxEqual(freq_water, minnaert_frequency(radius), 0.01), 
               "Water surface should use Minnaert frequency");
        
        // Noise surfaces should use peak hearing sensitivity (~3.5 kHz)
        double freq_pink = get_dominant_frequency(radius, SurfaceType.pink_noise);
        double freq_white = get_dominant_frequency(radius, SurfaceType.white_noise);
        assert(approxEqual(freq_pink, 3500.0, 0.01), "Pink noise should use 3.5 kHz");
        assert(approxEqual(freq_white, 3500.0, 0.01), "White noise should use 3.5 kHz");
    }
    
    // Test calculate_impulse_duration
    {
        double radius = 0.001;
        
        // Water surface duration should be based on Q factor
        double dur_water = calculate_impulse_duration(radius, SurfaceType.water, 10.0);
        assert(dur_water > 0.0 && dur_water < 1.0, "Water duration should be reasonable");
        
        // Noise surface duration should be based on contact time
        double dur_pink = calculate_impulse_duration(radius, SurfaceType.pink_noise);
        double dur_white = calculate_impulse_duration(radius, SurfaceType.white_noise);
        assert(dur_pink > 0.0 && dur_pink <= 0.1, "Pink noise duration in valid range");
        assert(dur_white > 0.0 && dur_white <= 0.1, "White noise duration in valid range");
    }
    
    writeln("  ✓ All physics tests passed");
}
