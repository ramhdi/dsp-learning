use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use hound::WavReader;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// FIR Filter using ring buffer for block processing (from original example)
struct Filter {
    coefficients: Vec<f32>,
    delay_line: Vec<f32>,    // Ring buffer for input history
    input_buffer: Vec<f32>,  // Current input block
    output_buffer: Vec<f32>, // Current output block
    write_index: usize,      // Ring buffer write position
}

impl Filter {
    fn new(coefficients: Vec<f32>, block_size: usize) -> Self {
        let filter_length = coefficients.len();
        Self {
            coefficients,
            delay_line: vec![0.0; filter_length],
            input_buffer: vec![0.0; block_size],
            output_buffer: vec![0.0; block_size],
            write_index: 0,
        }
    }

    /// Copy input block to internal buffer
    fn take_input(&mut self, input: &[f32]) {
        // Handle case where input size differs from expected block size
        let copy_len = input.len().min(self.input_buffer.len());
        self.input_buffer[..copy_len].copy_from_slice(&input[..copy_len]);

        // Fill remaining with zeros if input is smaller
        if copy_len < self.input_buffer.len() {
            self.input_buffer[copy_len..].fill(0.0);
        }
    }

    /// Perform time-domain convolution on the current input block
    fn convolve(&mut self) {
        // Process each sample in the input block
        for (i, &sample) in self.input_buffer.iter().enumerate() {
            // Store new sample in ring buffer delay line
            self.delay_line[self.write_index] = sample;

            // Compute convolution: y[n] = Σ h[k] * x[n-k]
            let mut output_sample = 0.0;
            for k in 0..self.coefficients.len() {
                // Ring buffer index for x[n-k]: wrap around using modulo
                let delay_index =
                    (self.write_index + self.delay_line.len() - k) % self.delay_line.len();
                output_sample += self.coefficients[k] * self.delay_line[delay_index];
            }

            // Store output sample
            self.output_buffer[i] = output_sample;

            // Advance ring buffer write pointer
            self.write_index = (self.write_index + 1) % self.delay_line.len();
        }
    }

    /// Get reference to the current output block
    fn yield_output(&self) -> &[f32] {
        &self.output_buffer
    }
}

/// Audio stream that provides blocks of samples
struct AudioStream {
    samples: Vec<f32>,
    position: usize,
    sample_rate: u32,
    loop_playback: bool,
    finished: bool,
    block_size: usize,
}

impl AudioStream {
    fn new(
        wav_path: &str,
        loop_playback: bool,
        block_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = WavReader::open(wav_path)?;
        let spec = reader.spec();

        println!("WAV file info:");
        println!("  Sample rate: {} Hz", spec.sample_rate);
        println!("  Channels: {}", spec.channels);
        println!("  Bits per sample: {}", spec.bits_per_sample);
        println!("  Block size: {} samples", block_size);

        // Read all samples and convert to f32
        let samples: Result<Vec<f32>, _> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect(),
            hound::SampleFormat::Int => {
                reader
                    .samples::<i32>()
                    .map(|s| {
                        // Convert to f32 range [-1.0, 1.0]
                        s.map(|sample| sample as f32 / (1i32 << (spec.bits_per_sample - 1)) as f32)
                    })
                    .collect()
            }
        };

        let samples = samples?;
        println!("  Total samples: {}", samples.len());
        println!(
            "  Duration: {:.2} seconds",
            samples.len() as f32 / spec.sample_rate as f32
        );
        println!(
            "  Total blocks: {}",
            (samples.len() + block_size - 1) / block_size
        );
        println!("  Looping: {}", if loop_playback { "Yes" } else { "No" });

        Ok(Self {
            samples,
            position: 0,
            sample_rate: spec.sample_rate,
            loop_playback,
            finished: false,
            block_size,
        })
    }

    /// Read next block of samples (like original example)
    fn read_block(&mut self, output: &mut [f32]) -> usize {
        if self.finished {
            output.fill(0.0);
            return 0;
        }

        let mut samples_read = 0;
        let requested_samples = output.len().min(self.block_size);

        for i in 0..requested_samples {
            if self.position >= self.samples.len() {
                if self.loop_playback {
                    // Loop the audio
                    self.position = 0;
                    println!("🔄 Looping audio...");
                } else {
                    // Mark as finished and fill remaining with silence
                    self.finished = true;
                    for j in i..requested_samples {
                        output[j] = 0.0;
                    }
                    return samples_read;
                }
            }

            output[i] = self.samples[self.position];
            self.position += 1;
            samples_read += 1;
        }

        // Fill remaining buffer with silence if needed
        for i in samples_read..output.len() {
            output[i] = 0.0;
        }

        samples_read
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn get_progress(&self) -> f32 {
        if self.samples.is_empty() {
            return 1.0;
        }
        (self.position as f32 / self.samples.len() as f32).min(1.0)
    }

    fn get_time_remaining(&self) -> f32 {
        if self.loop_playback || self.samples.is_empty() {
            return f32::INFINITY;
        }
        let remaining_samples = self.samples.len().saturating_sub(self.position);
        remaining_samples as f32 / self.sample_rate as f32
    }
}

/// Playback control and status
struct PlaybackControl {
    should_stop: AtomicBool,
    is_playing: AtomicBool,
    playback_finished: AtomicBool,
}

impl PlaybackControl {
    fn new() -> Self {
        Self {
            should_stop: AtomicBool::new(false),
            is_playing: AtomicBool::new(false),
            playback_finished: AtomicBool::new(false),
        }
    }

    fn stop(&self) {
        self.should_stop.store(true, Ordering::Relaxed);
        self.is_playing.store(false, Ordering::Relaxed);
    }

    fn start(&self) {
        self.is_playing.store(true, Ordering::Relaxed);
        self.should_stop.store(false, Ordering::Relaxed);
        self.playback_finished.store(false, Ordering::Relaxed);
    }

    fn should_stop(&self) -> bool {
        self.should_stop.load(Ordering::Relaxed)
    }

    fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::Relaxed)
    }

    fn set_finished(&self) {
        self.playback_finished.store(true, Ordering::Relaxed);
        self.is_playing.store(false, Ordering::Relaxed);
    }

    fn is_finished(&self) -> bool {
        self.playback_finished.load(Ordering::Relaxed)
    }
}

/// Filter design functions
fn create_lowpass_filter(cutoff_hz: f32, sample_rate: f32, length: usize) -> Vec<f32> {
    let mut coeffs = vec![0.0; length];
    let fc = cutoff_hz / sample_rate; // Normalized frequency
    let center = (length - 1) as f32 / 2.0;

    for i in 0..length {
        let n = i as f32 - center;

        if n == 0.0 {
            coeffs[i] = 2.0 * fc;
        } else {
            // Sinc function
            coeffs[i] = (2.0 * PI * fc * n).sin() / (PI * n);
        }

        // Apply Hamming window
        let window = 0.54 - 0.46 * (2.0 * PI * i as f32 / (length - 1) as f32).cos();
        coeffs[i] *= window;
    }

    // Normalize
    let sum: f32 = coeffs.iter().sum();
    for coeff in &mut coeffs {
        *coeff /= sum;
    }

    coeffs
}

fn create_highpass_filter(cutoff_hz: f32, sample_rate: f32, length: usize) -> Vec<f32> {
    let mut lowpass = create_lowpass_filter(cutoff_hz, sample_rate, length);

    // Create highpass by spectral inversion
    for coeff in &mut lowpass {
        *coeff = -*coeff;
    }

    // Add delta at center
    let center = (length - 1) / 2;
    lowpass[center] += 1.0;

    lowpass
}

fn create_moving_average(length: usize) -> Vec<f32> {
    vec![1.0 / length as f32; length]
}

/// Main application with proper block processing
struct WavFilterApp {
    audio_stream: Arc<Mutex<AudioStream>>,
    filter: Arc<Mutex<Filter>>,
    control: Arc<PlaybackControl>,
    block_size: usize,
    _stream: Stream,
}

impl WavFilterApp {
    fn new(
        wav_path: &str,
        filter_type: &str,
        cutoff_hz: f32,
        loop_playback: bool,
        block_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let audio_stream = Arc::new(Mutex::new(AudioStream::new(
            wav_path,
            loop_playback,
            block_size,
        )?));

        // Get sample rate from audio stream
        let sample_rate = {
            let stream = audio_stream.lock().unwrap();
            stream.sample_rate as f32
        };

        // Create filter coefficients
        let coeffs = match filter_type {
            "lowpass" => create_lowpass_filter(cutoff_hz, sample_rate, 64),
            "highpass" => create_highpass_filter(cutoff_hz, sample_rate, 64),
            "moving_average" => create_moving_average(64),
            _ => create_moving_average(1), // default no filtering
        };

        println!(
            "Filter: {} (cutoff: {} Hz, {} taps)",
            filter_type,
            cutoff_hz,
            coeffs.len()
        );

        let filter = Arc::new(Mutex::new(Filter::new(coeffs, block_size)));
        let control = Arc::new(PlaybackControl::new());

        // Setup audio output
        let stream = Self::setup_audio_output(
            audio_stream.clone(),
            filter.clone(),
            control.clone(),
            block_size,
        )?;

        Ok(Self {
            audio_stream,
            filter,
            control,
            block_size,
            _stream: stream,
        })
    }

    fn setup_audio_output(
        audio_stream: Arc<Mutex<AudioStream>>,
        filter: Arc<Mutex<Filter>>,
        control: Arc<PlaybackControl>,
        block_size: usize,
    ) -> Result<Stream, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("No output device available")?;

        let config = device.default_output_config()?;
        let sample_format = config.sample_format();
        let config: StreamConfig = config.into();

        println!("Audio output:");
        println!("  Device sample rate: {} Hz", config.sample_rate.0);
        println!("  Channels: {}", config.channels);
        println!("  Processing block size: {} samples", block_size);

        let stream = match sample_format {
            cpal::SampleFormat::F32 => {
                device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _| {
                        // Check if we should stop
                        if control.should_stop() {
                            data.fill(0.0);
                            return;
                        }

                        // Block processing (like original example)
                        let mut audio = audio_stream.lock().unwrap();
                        let mut filt = filter.lock().unwrap();

                        // Create input block buffer
                        let mut input_block = vec![0.0f32; block_size];

                        // Process in blocks
                        let mut output_pos = 0;
                        while output_pos < data.len() {
                            if control.should_stop() {
                                // Fill remaining with silence
                                for i in output_pos..data.len() {
                                    data[i] = 0.0;
                                }
                                break;
                            }

                            // Read input block
                            let samples_read = audio.read_block(&mut input_block);

                            if samples_read == 0 || audio.is_finished() {
                                // Audio finished, set control and fill with silence
                                control.set_finished();
                                for i in output_pos..data.len() {
                                    data[i] = 0.0;
                                }
                                break;
                            }

                            // Process block through filter (like original example)
                            filt.take_input(&input_block);
                            filt.convolve();
                            let output_block = filt.yield_output();

                            // Copy output block to audio device buffer
                            let copy_len = (data.len() - output_pos).min(output_block.len());
                            for i in 0..copy_len {
                                data[output_pos + i] = output_block[i];
                            }
                            output_pos += copy_len;
                        }
                    },
                    move |err| {
                        eprintln!("Audio stream error: {}", err);
                    },
                    None,
                )?
            }
            _ => return Err("Unsupported sample format".into()),
        };

        Ok(stream)
    }

    fn play(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Set up Ctrl+C handler
        let control_for_ctrlc = self.control.clone();
        ctrlc::set_handler(move || {
            println!("\n🛑 Received Ctrl+C, stopping playback...");
            control_for_ctrlc.stop();
        })?;

        self._stream.play()?;
        self.control.start();

        println!("\n🎵 Playing filtered audio with block processing...");
        println!("Processing {} samples per block", self.block_size);
        println!("Press Enter to stop, or Ctrl+C for immediate stop");

        // Monitor playback in a separate thread
        let control_for_monitor = self.control.clone();
        let audio_for_monitor = self.audio_stream.clone();
        thread::spawn(move || {
            let mut last_progress = 0.0;
            let mut block_count = 0u64;

            while control_for_monitor.is_playing() && !control_for_monitor.is_finished() {
                thread::sleep(Duration::from_millis(1000));

                let (progress, time_remaining) = {
                    let audio = audio_for_monitor.lock().unwrap();
                    (audio.get_progress(), audio.get_time_remaining())
                };

                block_count += 1;

                // Show progress every 10% or every 10 seconds
                if (progress - last_progress) >= 0.1 || time_remaining.is_finite() {
                    if time_remaining.is_finite() {
                        println!(
                            "🎵 Progress: {:.1}% - Time remaining: {:.1}s - Blocks processed: {}",
                            progress * 100.0,
                            time_remaining,
                            block_count * 1000 / 1000 // Approximate blocks per second
                        );
                    }
                    last_progress = progress;
                }
            }
        });

        // Wait for user input or automatic finish
        let mut input = String::new();
        let stdin_result = std::io::stdin().read_line(&mut input);

        // Check if playback finished naturally
        if self.control.is_finished() {
            println!("✅ Playback completed.");
        } else if stdin_result.is_ok() {
            println!("🛑 Stopping playback...");
            self.control.stop();
        } else {
            eprintln!("❌ Error reading input, stopping...");
            self.control.stop();
        }

        // Give audio stream time to stop gracefully
        thread::sleep(Duration::from_millis(100));

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎛️  Block Processing WAV Audio Filter");
    println!("====================================");

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!(
            "Usage: {} <wav_file> [filter_type] [cutoff_hz] [block_size] [--loop]",
            args[0]
        );
        println!();
        println!("Filter types:");
        println!("  lowpass      - Remove frequencies above cutoff");
        println!("  highpass     - Remove frequencies below cutoff");
        println!("  moving_average - Simple smoothing filter");
        println!();
        println!("Parameters:");
        println!("  cutoff_hz    - Filter cutoff frequency (default: 1000)");
        println!("  block_size   - Processing block size (default: 64)");
        println!("  --loop       - Loop audio playback indefinitely");
        println!();
        println!("Examples:");
        println!("  {} audio.wav lowpass 2000 64", args[0]);
        println!("  {} music.wav highpass 100 32 --loop", args[0]);
        println!("  {} voice.wav moving_average", args[0]);
        return Ok(());
    }

    let wav_path = &args[1];
    let filter_type = args.get(2).map(|s| s.as_str()).unwrap_or("lowpass");
    let cutoff_hz: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000.0);
    let block_size: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let loop_playback = args.contains(&"--loop".to_string());

    // Validate block size
    if block_size == 0 || block_size > 4096 {
        eprintln!("❌ Error: Block size must be between 1 and 4096");
        return Ok(());
    }

    // Check if file exists
    if !std::path::Path::new(wav_path).exists() {
        eprintln!("❌ Error: WAV file '{}' not found", wav_path);
        return Ok(());
    }

    println!("📁 File: {}", wav_path);

    // Create and run the application
    match WavFilterApp::new(wav_path, filter_type, cutoff_hz, loop_playback, block_size) {
        Ok(app) => {
            app.play()?;
            println!("✅ Application finished.");
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    Ok(())
}

// Cargo.toml dependencies:
/*
[dependencies]
cpal = "0.15"
hound = "3.5"
ctrlc = "3.4"
*/
