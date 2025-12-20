/**
 * Surface type enumeration for raindrop acoustics
 * 
 * Defines different surface types that affect the acoustic properties
 * of raindrop impacts.
 */
module surface;

/**
 * Enum defining different surface types for raindrop impacts
 */
enum SurfaceType {
    water,       // Water surface with Minnaert resonance
    capillary,   // Surface tension dominated (capillary waves)
    pink_noise,  // Pink noise generation
    white_noise  // White noise generation
}

/**
 * Convert string to SurfaceType enum
 * 
 * Params:
 *   surface_str = String representation of surface type
 * 
 * Returns:
 *   Corresponding SurfaceType enum value
 * 
 * Throws:
 *   Exception if surface string is not recognized
 */
SurfaceType parseSurfaceType(string surface_str) {
    final switch (surface_str) {
        case "water":
            return SurfaceType.water;
        case "capillary":
            return SurfaceType.capillary;
        case "pink_noise":
            return SurfaceType.pink_noise;
        case "white_noise":
            return SurfaceType.white_noise;
    }
}

/**
 * Convert SurfaceType enum to string
 * 
 * Params:
 *   surface_type = SurfaceType enum value
 * 
 * Returns:
 *   String representation of the surface type
 */
string surfaceTypeToString(SurfaceType surface_type) {
    final switch (surface_type) {
        case SurfaceType.water:
            return "water";
        case SurfaceType.capillary:
            return "capillary";
        case SurfaceType.pink_noise:
            return "pink_noise";
        case SurfaceType.white_noise:
            return "white_noise";
    }
}