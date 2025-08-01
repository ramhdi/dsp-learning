use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;
use wide::f32x8;

/// Signal source that generates sinusoidal signals in blocks
struct SignalSource {
    frequency: f32,
    sampling_rate: f32,
    block_size: usize,
    phase: f32,
    output_buffer: Vec<f32>,
}

impl SignalSource {
    fn new(frequency: f32, sampling_rate: f32, block_size: usize) -> Self {
        Self {
            frequency,
            sampling_rate,
            block_size,
            phase: 0.0,
            output_buffer: vec![0.0; block_size],
        }
    }

    /// Generate the next block of sinusoidal samples
    fn read(&mut self) -> &[f32] {
        let omega = 2.0 * PI * self.frequency / self.sampling_rate;

        // Generate block of samples, maintaining phase continuity
        for i in 0..self.block_size {
            self.output_buffer[i] = (self.phase + i as f32 * omega).sin();
        }

        // Update phase for next block (keep bounded to avoid precision loss)
        self.phase += self.block_size as f32 * omega;
        self.phase = self.phase % (2.0 * PI);

        &self.output_buffer
    }
}

/// Filter with both scalar and SIMD convolution methods
struct Filter {
    coefficients: Vec<f32>,
    delay_line: Vec<f32>,           // Ring buffer for input history
    input_buffer: Vec<f32>,         // Current input block
    output_buffer_scalar: Vec<f32>, // Scalar convolution output
    output_buffer_simd: Vec<f32>,   // SIMD convolution output
    write_index: usize,             // Ring buffer write position
}

impl Filter {
    fn new(coefficients: Vec<f32>, block_size: usize) -> Self {
        let filter_length = coefficients.len();
        println!("Filter created with {} coefficients", filter_length);
        println!("SIMD will process in chunks of 8 (f32x8)");

        Self {
            coefficients,
            delay_line: vec![0.0; filter_length],
            input_buffer: vec![0.0; block_size],
            output_buffer_scalar: vec![0.0; block_size],
            output_buffer_simd: vec![0.0; block_size],
            write_index: 0,
        }
    }

    /// Copy input block to internal buffer
    fn take_input(&mut self, input: &[f32]) {
        self.input_buffer.copy_from_slice(input);
    }

    /// Original scalar convolution (unchanged from main.rs)
    fn convolve_scalar(&mut self) {
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
            self.output_buffer_scalar[i] = output_sample;

            // Advance ring buffer write pointer
            self.write_index = (self.write_index + 1) % self.delay_line.len();
        }
    }

    /// SIMD convolution using wide::f32x8
    fn convolve_simd(&mut self) {
        // Reset write index to same starting position as scalar
        let mut simd_write_index = self.write_index;

        // We need to reprocess the same input, so restore the delay line state
        // For learning purposes, we'll create a separate delay line copy
        let mut simd_delay_line = self.delay_line.clone();

        // Process each sample in the input block
        for (i, &sample) in self.input_buffer.iter().enumerate() {
            // Store new sample in SIMD delay line
            simd_delay_line[simd_write_index] = sample;

            // SIMD-accelerated dot product computation
            let mut acc = f32x8::ZERO;

            // Process coefficients in chunks of 8
            let simd_chunks = self.coefficients.len() / 8;

            for chunk in 0..simd_chunks {
                let coeff_start = chunk * 8;

                // Load 8 coefficients into SIMD vector
                let coeffs = f32x8::from([
                    self.coefficients[coeff_start],
                    self.coefficients[coeff_start + 1],
                    self.coefficients[coeff_start + 2],
                    self.coefficients[coeff_start + 3],
                    self.coefficients[coeff_start + 4],
                    self.coefficients[coeff_start + 5],
                    self.coefficients[coeff_start + 6],
                    self.coefficients[coeff_start + 7],
                ]);

                // Load 8 corresponding delayed samples
                let mut samples = [0.0f32; 8];
                for j in 0..8 {
                    let k = coeff_start + j;
                    let delay_index =
                        (simd_write_index + simd_delay_line.len() - k) % simd_delay_line.len();
                    samples[j] = simd_delay_line[delay_index];
                }
                let delayed = f32x8::from(samples);

                // SIMD multiply and accumulate: acc += coeffs * delayed
                acc += coeffs * delayed;
            }

            // Sum all 8 elements in the accumulator (horizontal sum)
            let sum_array = acc.to_array();
            let mut output_sample = sum_array.iter().sum::<f32>();

            // Handle remaining coefficients that don't fit in SIMD chunks (scalar fallback)
            let remainder_start = simd_chunks * 8;
            for k in remainder_start..self.coefficients.len() {
                let delay_index =
                    (simd_write_index + simd_delay_line.len() - k) % simd_delay_line.len();
                output_sample += self.coefficients[k] * simd_delay_line[delay_index];
            }

            // Store SIMD output sample
            self.output_buffer_simd[i] = output_sample;

            // Advance SIMD ring buffer write pointer
            simd_write_index = (simd_write_index + 1) % simd_delay_line.len();
        }

        // Update the main delay line to match SIMD processing
        self.delay_line = simd_delay_line;
        self.write_index = simd_write_index;
    }

    /// Get reference to scalar output
    fn yield_scalar_output(&self) -> &[f32] {
        &self.output_buffer_scalar
    }

    /// Get reference to SIMD output  
    fn yield_simd_output(&self) -> &[f32] {
        &self.output_buffer_simd
    }
}

/// Generate simple lowpass FIR filter coefficients (moving average)
fn create_lowpass_fir(length: usize) -> Vec<f32> {
    // Simple moving average filter: h[k] = 1/N for k = 0..N-1
    vec![1.0 / length as f32; length]
}

/// Save comparison data to CSV file
fn save_comparison_to_csv(
    filename: &str,
    input_samples: &[f32],
    scalar_output: &[f32],
    simd_output: &[f32],
    sampling_rate: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(filename)?;

    // Write CSV header
    writeln!(
        file,
        "index,time,input,scalar_output,simd_output,difference"
    )?;

    // Write data rows
    for (i, ((&input, &scalar), &simd)) in input_samples
        .iter()
        .zip(scalar_output.iter())
        .zip(simd_output.iter())
        .enumerate()
    {
        let time = i as f32 / sampling_rate;
        let difference = (scalar - simd).abs();
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.9}",
            i, time, input, scalar, simd, difference
        )?;
    }

    println!("Comparison data saved to: {}", filename);
    Ok(())
}

fn main() {
    println!("SIMD Learning: Scalar vs SIMD Convolution Comparison");
    println!("====================================================");

    // Simulation parameters
    let sampling_rate = 1000.0;
    let frequency = 50.0;
    let block_size = 32;
    let num_blocks = 5;

    // Create filter with length that demonstrates SIMD benefits
    let filter_length = 64;
    let coefficients = create_lowpass_fir(filter_length);

    // Create TWO INDEPENDENT filters - much cleaner!
    let mut signal_source = SignalSource::new(frequency, sampling_rate, block_size);
    let mut scalar_filter = Filter::new(coefficients.clone(), block_size);
    let mut simd_filter = Filter::new(coefficients, block_size);

    // Storage for accumulated results
    let mut input_accumulated = Vec::new();
    let mut scalar_output_accumulated = Vec::new();
    let mut simd_output_accumulated = Vec::new();

    println!("Processing parameters:");
    println!("  Sampling Rate: {} Hz", sampling_rate);
    println!("  Input Frequency: {} Hz", frequency);
    println!("  Block Size: {} samples", block_size);
    println!("  Filter Length: {} coefficients", filter_length);
    println!(
        "  SIMD Chunks: {} (remainder: {})",
        filter_length / 8,
        filter_length % 8
    );
    println!();

    // Process blocks with independent filters
    for block_num in 0..num_blocks {
        println!("Processing block {}...", block_num);

        // Get input signal block
        let input_block = signal_source.read();

        // Process with scalar filter
        scalar_filter.take_input(input_block);
        let start_time = std::time::Instant::now();
        scalar_filter.convolve_scalar();
        let scalar_duration = start_time.elapsed();
        let scalar_output = scalar_filter.yield_scalar_output();

        // Process with SIMD filter (independent state)
        simd_filter.take_input(input_block);
        let start_time = std::time::Instant::now();
        simd_filter.convolve_simd();
        let simd_duration = start_time.elapsed();
        let simd_output = simd_filter.yield_simd_output();

        // Collect results
        input_accumulated.extend_from_slice(input_block);
        scalar_output_accumulated.extend_from_slice(scalar_output);
        simd_output_accumulated.extend_from_slice(simd_output);

        // Print timing comparison
        println!("  Scalar time: {:?}", scalar_duration);
        println!("  SIMD time: {:?}", simd_duration);

        // Check for differences
        let max_diff = scalar_output
            .iter()
            .zip(simd_output.iter())
            .map(|(s, v)| (s - v).abs())
            .fold(0.0f32, f32::max);

        println!("  Maximum difference: {:.2e}", max_diff);

        // Print first few samples for manual verification
        if block_num < 2 {
            println!("  First 4 samples:");
            for i in 0..4.min(input_block.len()) {
                println!(
                    "    [{}] Input: {:.4}, Scalar: {:.4}, SIMD: {:.4}, Diff: {:.2e}",
                    i,
                    input_block[i],
                    scalar_output[i],
                    simd_output[i],
                    (scalar_output[i] - simd_output[i]).abs()
                );
            }
        }
        println!();
    }

    // Display final statistics
    println!("Processing complete!");
    println!("Total samples processed: {}", input_accumulated.len());

    // Calculate overall error statistics
    let differences: Vec<f32> = scalar_output_accumulated
        .iter()
        .zip(simd_output_accumulated.iter())
        .map(|(s, v)| (s - v).abs())
        .collect();

    let max_error = differences.iter().fold(0.0f32, |a, &b| a.max(b));
    let mean_error = differences.iter().sum::<f32>() / differences.len() as f32;
    let rms_error =
        (differences.iter().map(|&x| x * x).sum::<f32>() / differences.len() as f32).sqrt();

    println!("\nError Analysis:");
    println!("  Maximum error: {:.2e}", max_error);
    println!("  Mean error: {:.2e}", mean_error);
    println!("  RMS error: {:.2e}", rms_error);

    if max_error < 1e-6 {
        println!(
            "  ✅ SIMD implementation appears correct (errors within floating-point precision)"
        );
    } else {
        println!("  ⚠️ SIMD implementation may have issues (errors exceed expected precision)");
    }

    // Save comparison data to CSV for Python analysis
    save_comparison_to_csv(
        "simd_comparison.csv",
        &input_accumulated,
        &scalar_output_accumulated,
        &simd_output_accumulated,
        sampling_rate,
    )
    .unwrap();

    println!("\n🔍 Run the Python verification script to analyze results:");
    println!("python verify_simd.py");
}
