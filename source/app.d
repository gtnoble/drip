/**
 * Drip: Physics-based rainfall sound generator
 * 
 * Command-line interface for generating binaural rainfall audio.
 */
import std.stdio;
import std.conv : to;
import std.typecons : tuple;
import darg;

import synthesis;

/**
 * Command-line arguments
 */
struct Options {
    @Option("help", "h")
    @Help("Show this help message")
    OptionFlag help;
    
    @Option("duration", "d")
    @Help("Duration in seconds (required)")
    double duration = 0.0;
    
    @Option("rain-rate")
    @Help("Rainfall rate in mm/hr - controls drop size distribution (default: 10.0)")
    double rain_rate = 10.0;
    
    @Option("drop-rate")
    @Help("Total drops per second distributed across the buffer (default: 5000.0)")
    double drop_rate = 5000.0;
    
    @Option("quality-factor", "q")
    @Help("Quality factor Q: controls resonance (2=splashy, 20=tonal, default: 10.0)")
    double quality_factor = 10.0;
    
    @Option("ear-separation", "e")
    @Help("Ear separation in meters (default: 0.17)")
    double ear_separation = 0.17;
    
    @Option("listener-height", "l")
    @Help("Listener height above raindrop plane in meters (default: 1.7)")
    double listener_height = 1.7;
    
    @Option("sample-rate", "r")
    @Help("Sample rate in Hz (default: 44100)")
    int sample_rate = 44100;
    
    @Option("output", "o")
    @Help("Output WAV file (default: stdout)")
    string output = "";
    
    @Option("surface-type", "s")
    @Help("Surface type: water, pink_noise, white_noise (default: water)")
    string surface_type = "water";
    
    @Option("seed")
    @Help("Random seed for reproducibility (default: random)")
    uint seed = 0;
    
    @Option("no-hearing-threshold")
    @Help("Disable hearing threshold filtering (generate all drops)")
    OptionFlag no_hearing_threshold;
    
    @Option("mh-burn-in")
    @Help("Metropolis-Hastings burn-in iterations (default: 1000)")
    int mh_burn_in = 1000;
    
    @Option("mh-position-scale")
    @Help("M-H position proposal scale in meters (default: 5.0)")
    double mh_position_scale = 5.0;
    
    
}

/**
 * Write WAV file
 */
void write_wav(string filename, float[] left, float[] right, int sample_rate) {
    import std.file : write;
    import std.outbuffer : OutBuffer;
    
    int n_samples = cast(int)left.length;
    int n_channels = 2;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * n_channels * bits_per_sample / 8;
    int block_align = n_channels * bits_per_sample / 8;
    int data_size = n_samples * n_channels * bits_per_sample / 8;
    
    auto buf = new OutBuffer();
    
    // RIFF header
    buf.write("RIFF");
    buf.write(cast(uint)(36 + data_size));
    buf.write("WAVE");
    
    // fmt chunk
    buf.write("fmt ");
    buf.write(cast(uint)16);  // chunk size
    buf.write(cast(ushort)1); // audio format (PCM)
    buf.write(cast(ushort)n_channels);
    buf.write(cast(uint)sample_rate);
    buf.write(cast(uint)byte_rate);
    buf.write(cast(ushort)block_align);
    buf.write(cast(ushort)bits_per_sample);
    
    // data chunk
    buf.write("data");
    buf.write(cast(uint)data_size);
    
    // Convert float to int16 and interleave
    foreach (i; 0 .. n_samples) {
        short sample_L = cast(short)(left[i] * 32767.0);
        short sample_R = cast(short)(right[i] * 32767.0);
        buf.write(sample_L);
        buf.write(sample_R);
    }
    
    // Write to file or stdout
    if (filename == "" || filename == "-") {
        import core.stdc.stdio : fwrite, stdout, fflush;
        fwrite(buf.toBytes().ptr, 1, buf.offset, stdout);
        fflush(stdout);
    } else {
        import std.file : write;
        write(filename, buf.toBytes());
    }
}

void main(string[] args) {
    Options options;
    
    try {
        options = parseArgs!Options(args[1..$]);
    } catch (ArgParseError e) {
        stderr.writeln("Error: ", e.msg);
        stderr.writeln();
        stderr.writeln(usageString!Options("drip"));
        return;
    } catch (ArgParseHelp e) {
        writeln(e.msg);
        return;
    }
    
    // Validate required arguments
    if (options.duration <= 0.0) {
        stderr.writeln("Error: --duration is required and must be positive");
        stderr.writeln();
        stderr.writeln(usageString!Options("drip"));
        return;
    }
    
    // Validate surface type
    if (options.surface_type != "water" && 
        options.surface_type != "pink_noise" && 
        options.surface_type != "white_noise") {
        stderr.writeln("Error: --surface-type must be water, pink_noise, or white_noise");
        return;
    }
    
    // Synthesize rain
    try {
        auto result = synthesize_rain(
            options.duration,
            options.rain_rate,
            options.quality_factor,
            options.ear_separation,
            options.listener_height,
            options.sample_rate,
            options.seed,
            !options.no_hearing_threshold,
            options.mh_burn_in,
            0.0,  // auto radius scale
            options.mh_position_scale,
            options.drop_rate,
            options.surface_type
        );
        
        // Write output
        write_wav(options.output, result[0], result[1], options.sample_rate);
        
        if (options.output != "" && options.output != "-") {
            stderr.writefln("Wrote %s", options.output);
        }
        
    } catch (Exception e) {
        stderr.writeln("Error during synthesis: ", e.msg);
        import core.stdc.stdlib : exit;
        exit(1);
    }
}
