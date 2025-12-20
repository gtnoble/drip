/**
 * Drop parameter sampling for Poisson process generation
 * 
 * Provides standalone function to generate drop parameters (radius, position)
 * using Metropolis-Hastings sampling for audibility constraints.
 * Time is handled by Poisson process at sample level.
 */
module sampler;

import std.math : sqrt, exp, log, pow;
import mir.random : Random, unpredictableSeed;
import mir.random.variable : normalVar, uniformVar, poissonVar;

import physics;
import propagation;
import surface;

/**
 * Drop parameters (without time - handled by Poisson process)
 */
struct Drop {
    double radius;       // Drop radius in meters
    Position position;   // Drop position (x, y, z)
}

/**
 * State for Metropolis-Hastings drop parameter sampler
 */
struct DropSamplerState {
    double R_current;
    double x_current;
    double y_current;
    int accepted;
    int total_proposals;
    Random rng;
    
    // Configuration
    double rain_rate;
    Position ear_L, ear_R;
    int sample_rate;
    bool hearing_threshold_enabled;
    double radius_scale;
    double position_scale;
    SurfaceType surface_type;
    double z_current = 0.0;  // All drops on z=0 plane
}

/**
 * Initialize drop sampler state
 */
DropSamplerState initialize_drop_sampler(
    double rain_rate,
    double ear_separation,
    double listener_height,
    int sample_rate,
    bool hearing_threshold_enabled,
    int burn_in,
    double radius_scale,
    double position_scale,
    uint seed,
    SurfaceType surface_type) {
    
    DropSamplerState state = DropSamplerState.init;
    state.rain_rate = rain_rate;
    state.sample_rate = sample_rate;
    state.hearing_threshold_enabled = hearing_threshold_enabled;
    state.surface_type = surface_type;
    
    // Ear positions (3D: x, y, z)
    state.ear_L = Position(-ear_separation / 2.0, 0.0, listener_height);
    state.ear_R = Position(ear_separation / 2.0, 0.0, listener_height);
    
    // Auto-tune radius scale from Marshall-Palmer distribution if not provided
    if (radius_scale <= 0.0) {
        double Lambda = marshall_palmer_lambda(rain_rate);
        state.radius_scale = (1.0 / Lambda) / 2000.0;  // Mean radius in meters
    } else {
        state.radius_scale = radius_scale;
    }
    
    state.position_scale = position_scale;
    
    // Initialize RNG
    if (seed == 0) {
        state.rng = Random(unpredictableSeed);
    } else {
        state.rng = Random(seed);
    }
    
    // Initialize chain at origin with typical drop size
    double Lambda = marshall_palmer_lambda(rain_rate);
    state.R_current = (1.0 / Lambda) / 2000.0;  // Mean radius
    state.x_current = 0.0;
    state.y_current = 0.0;
    state.accepted = 0;
    state.total_proposals = 0;
    
    // Maximum physically possible radius (10mm diameter)
    enum double max_radius = 0.005;  // 5mm radius in meters
    
    // Find initial audible state (required for M-H)
    bool found_initial = false;
    auto normal_dist = normalVar!double(0.0, 1.0);
    
    foreach (attempt; 0 .. 10000) {
        double R_proposal = state.R_current + normal_dist(state.rng) * state.radius_scale;
        double x_proposal = state.x_current + normal_dist(state.rng) * state.position_scale;
        double y_proposal = state.y_current + normal_dist(state.rng) * state.position_scale;
        
        // Clamp radius to physical range
        if (R_proposal < 0.0001 || R_proposal > max_radius) continue;
        
        if (is_audible(state, R_proposal, x_proposal, y_proposal)) {
            state.R_current = R_proposal;
            state.x_current = x_proposal;
            state.y_current = y_proposal;
            found_initial = true;
            break;
        }
    }
    
    assert(found_initial, "Could not find initial audible state in M-H sampler");
    
    // Burn-in phase
    foreach (i; 0 .. burn_in) {
        double R_proposal = state.R_current + normal_dist(state.rng) * state.radius_scale;
        double x_proposal = state.x_current + normal_dist(state.rng) * state.position_scale;
        double y_proposal = state.y_current + normal_dist(state.rng) * state.position_scale;
        
        if (R_proposal >= 0.0001 && R_proposal <= max_radius &&
            is_audible(state, R_proposal, x_proposal, y_proposal)) {
            state.R_current = R_proposal;
            state.x_current = x_proposal;
            state.y_current = y_proposal;
        }
    }
    
    return state;
}

double acoustic_energy_to_pressure(double energy, double impulse_duration) {
    // Convert energy to pressure: E -> Power -> Intensity -> Pressure
    double power = energy / impulse_duration;
    double intensity = power / (2.0 * 3.14159265358979323846);  // Hemispherical spreading
    enum double rho_c = 415.0;  // Acoustic impedance of air (ρ × c)
    double pressure = sqrt(intensity * rho_c);
    return pressure;
}

/**
 * Check if a drop at (x, y, z) with radius R is audible
 */
bool is_audible(ref DropSamplerState state, double R, double x, double y) {
        // Calculate frequency
        double frequency = get_dominant_frequency(R, state.surface_type);
        
        // Nyquist frequency check
        double nyquist = state.sample_rate / 2.0;
        if (frequency > nyquist) return false;
        
        // Hearing threshold check (if enabled)
        if (state.hearing_threshold_enabled) {
            // Calculate acoustic energy
            double source_energy = acoustic_amplitude(R);
            
            // Calculate audibility from center point between ears (0, 0, listener_height)
            // This makes audibility radially symmetric
            Position drop_pos = Position(x, y, state.z_current);
            Position center_pos = Position(0.0, 0.0, (state.ear_L.z + state.ear_R.z) / 2.0);
            
            // Calculate distance from center point
            double dx = drop_pos.x - center_pos.x;
            double dy = drop_pos.y - center_pos.y;
            double dz = drop_pos.z - center_pos.z;
            double distance = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (distance < 0.01) distance = 0.01;
            
            // Calculate received energy
            double geometric_attenuation = 1.0 / (distance * distance);
            double impulse_duration = calculate_impulse_duration(R, state.surface_type);
            double air_gain = air_absorption_gain(frequency, distance);
            double attenuated_energy = source_energy * geometric_attenuation * air_gain;

            // Check against hearing threshold
            double attenuated_pa = acoustic_energy_to_pressure(attenuated_energy, impulse_duration);
            double threshold = hearing_threshold_pa(frequency);
            return attenuated_pa >= threshold;
        }
        
        return true;  // Passed Nyquist check
    }

/**
 * Sample one drop parameter set using M-H sampling
 * Updates state in-place and returns new drop parameters
 */
Drop sample_drop(ref DropSamplerState state) {
    enum double max_radius = 0.005;
    auto normal_dist = normalVar!double(0.0, 1.0);
    auto uniform_dist = uniformVar!double(0.0, 1.0);
    
    // Propose new state with Gaussian random walk
    double R_proposal = state.R_current + normal_dist(state.rng) * state.radius_scale;
    double x_proposal = state.x_current + normal_dist(state.rng) * state.position_scale;
    double y_proposal = state.y_current + normal_dist(state.rng) * state.position_scale;
    
    state.total_proposals++;
    
    // Metropolis-Hastings acceptance logic
    // Reject negative radii or physically impossible large drops
    if (R_proposal >= 0.0001 && R_proposal <= max_radius) {
        // Marshall-Palmer probability ratio
        // Marshall-Palmer uses diameter D in mm: N(D) = N₀ exp(-Λ D)
        // Convert radius (m) to diameter (mm): D_mm = 2 * R_m * 1000
        double Lambda = marshall_palmer_lambda(state.rain_rate);
        double exponent = -Lambda * (R_proposal - state.R_current) * 2000.0;  // 2*1000: radius→diameter, m→mm
        
        // Clamp exponent to prevent overflow/underflow
        if (exponent > 100.0) exponent = 100.0;
        if (exponent < -100.0) exponent = -100.0;
        
        double p_ratio = exp(exponent);
        
        // Audibility check
        bool audible_proposed = is_audible(state, R_proposal, x_proposal, y_proposal);
        
        // Accept if audible AND passes M-H test
        if (audible_proposed) {
            if (uniform_dist(state.rng) < (p_ratio < 1.0 ? p_ratio : 1.0)) {
                state.R_current = R_proposal;
                state.x_current = x_proposal;
                state.y_current = y_proposal;
                state.accepted++;
            }
        }
    }
    
    return Drop(state.R_current, Position(state.x_current, state.y_current, state.z_current));
}

/**
 * Get current M-H acceptance rate
 */
double get_acceptance_rate(ref DropSamplerState state) {
    return (state.total_proposals > 0) ? 
        cast(double)state.accepted / cast(double)state.total_proposals : 0.0;
}
