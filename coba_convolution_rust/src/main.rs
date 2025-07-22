use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;

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

/// FIR Filter using ring buffer for zero-allocation block processing
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
        self.input_buffer.copy_from_slice(input);
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

/// Generate simple lowpass FIR filter coefficients (moving average)
fn create_lowpass_fir(length: usize) -> Vec<f32> {
    // Simple moving average filter: h[k] = 1/N for k = 0..N-1
    vec![1.0 / length as f32; length]
}

/// Save signal data to CSV file
fn save_to_csv(
    filename: &str,
    input_samples: &[f32],
    output_samples: &[f32],
    sampling_rate: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(filename)?;

    // Write CSV header
    writeln!(file, "index,time,input,output")?;

    // Write data rows
    for (i, (&input, &output)) in input_samples.iter().zip(output_samples.iter()).enumerate() {
        let time = i as f32 / sampling_rate;
        writeln!(file, "{},{:.6},{:.6},{:.6}", i, time, input, output)?;
    }

    println!("Data saved to: {}", filename);
    Ok(())
}

fn main() {
    // Simulation parameters
    let sampling_rate = 1000.0; // Hz
    let frequency = 50.0; // Hz - input sine wave frequency
    let block_size = 32; // Samples per block
    let num_blocks = 10; // Number of blocks to process

    // Create signal source and filter
    let mut signal_source = SignalSource::new(frequency, sampling_rate, block_size);
    let mut filter = Filter::new(create_lowpass_fir(32), block_size);

    // Storage for accumulated input and output
    let mut input_accumulated = Vec::new();
    let mut output_accumulated = Vec::new();

    println!("Time-Domain Block Convolution Demo");
    println!("Sampling Rate: {} Hz", sampling_rate);
    println!("Input Frequency: {} Hz", frequency);
    println!("Block Size: {} samples", block_size);
    println!("Filter: 8-tap moving average lowpass");
    println!();

    // Process blocks
    for block_num in 0..num_blocks {
        // Get input signal block
        let input_block = signal_source.read();

        // Process through filter
        filter.take_input(input_block);
        filter.convolve();

        // Collect input and output
        let output_block = filter.yield_output();
        input_accumulated.extend_from_slice(input_block);
        output_accumulated.extend_from_slice(output_block);

        // Print some debug info for first few blocks
        if block_num < 3 {
            println!(
                "Block {}: Input[0-3] = [{:.3}, {:.3}, {:.3}, {:.3}]",
                block_num, input_block[0], input_block[1], input_block[2], input_block[3]
            );
            println!(
                "         Output[0-3] = [{:.3}, {:.3}, {:.3}, {:.3}]",
                output_block[0], output_block[1], output_block[2], output_block[3]
            );
            println!();
        }
    }

    // Display summary statistics
    println!("Processing complete!");
    println!("Total samples processed: {}", input_accumulated.len());

    // Save data to CSV
    save_to_csv(
        "signal_data.csv",
        &input_accumulated,
        &output_accumulated,
        sampling_rate,
    )
    .unwrap();

    // Optional: print first few and last few samples
    println!("\nFirst 10 samples:");
    for i in 0..10.min(input_accumulated.len()) {
        println!(
            "  Sample[{}]: Input = {:.4}, Output = {:.4}",
            i, input_accumulated[i], output_accumulated[i]
        );
    }

    println!("\nLast 10 samples:");
    let start_idx = input_accumulated.len().saturating_sub(10);
    for i in start_idx..input_accumulated.len() {
        println!(
            "  Sample[{}]: Input = {:.4}, Output = {:.4}",
            i, input_accumulated[i], output_accumulated[i]
        );
    }
}
