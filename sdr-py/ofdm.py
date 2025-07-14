#!/usr/bin/env python3
"""
Complete OFDM Simulation
Implements text-to-text transmission through OFDM modulation/demodulation
with constellation diagrams, BER analysis, and character error counting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os


class OFDMSimulator:
    """Complete OFDM transmission simulation with comprehensive analysis."""

    def __init__(
        self,
        N: int = 64,  # Number of subcarriers
        N_cp: int = 16,  # Cyclic prefix length
        modulation: str = "QPSK",  # Modulation scheme
        snr_db: float = 20.0,  # Signal-to-noise ratio in dB
        channel_taps: Optional[List[float]] = None,
    ):
        """
        Initialize OFDM simulator parameters.

        Args:
            N: Number of subcarriers
            N_cp: Cyclic prefix length
            modulation: Modulation scheme ('QPSK', '16QAM')
            snr_db: Signal-to-noise ratio in dB
            channel_taps: Channel impulse response (None for AWGN only)
        """
        self.N = N
        self.N_cp = N_cp
        self.modulation = modulation
        self.snr_db = snr_db
        self.channel_taps = channel_taps or [1.0]  # AWGN channel by default

        # Modulation parameters
        if modulation == "BPSK":
            self.bits_per_symbol = 1
            self.constellation = self._generate_bpsk_constellation()
        elif modulation == "QPSK":
            self.bits_per_symbol = 2
            self.constellation = self._generate_qpsk_constellation()
        elif modulation == "16QAM":
            self.bits_per_symbol = 4
            self.constellation = self._generate_16qam_constellation()
        else:
            raise ValueError(f"Unsupported modulation: {modulation}")

        # Statistics tracking
        self.tx_bits = np.ndarray([])
        self.rx_bits = np.ndarray([])
        self.tx_symbols = np.ndarray([])
        self.rx_symbols_before_eq = np.ndarray([])
        self.rx_symbols_after_eq = np.ndarray([])

        # Comb-type pilot sequence for channel equalization
        self.pilot_spacing = 8  # Space between pilots
        self.pilot_symbol = 1.0  # Base pilot symbol (BPSK)
        self.pilot_indices = np.arange(0, self.N, self.pilot_spacing)
        self.num_pilots = len(self.pilot_indices)
        self.data_indices = np.array(
            [i for i in range(self.N) if i not in self.pilot_indices]
        )

        # Generate alternating pilot sequence
        self.pilot_sequence = np.array(
            [
                self.pilot_symbol * (1 if i % 2 == 0 else -1)
                for i in range(self.num_pilots)
            ],
            dtype=np.complex64,
        )

    def _generate_bpsk_constellation(self) -> np.ndarray:
        """Generate BPSK constellation points."""
        return np.array([1, -1])  # 0 or 1

    def _generate_qpsk_constellation(self) -> np.ndarray:
        """Generate QPSK constellation points."""
        return np.array(
            [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]  # 00  # 01  # 10  # 11
        ) / np.sqrt(2)

    def _generate_16qam_constellation(self) -> np.ndarray:
        """Generate 16-QAM constellation points."""
        points = []
        for i in range(4):
            for q in range(4):
                # Gray coding for 16-QAM
                real = 2 * i - 3
                imag = 2 * q - 3
                points.append(real + 1j * imag)
        return np.array(points) / np.sqrt(10)  # Normalize for unit average power

    def text_to_bits(self, text: str) -> np.ndarray:
        """Convert text to binary representation."""
        # Convert to UTF-8 bytes then to bits
        text_bytes = text.encode("utf-8")
        bits = []
        for byte in text_bytes:
            bits.extend([int(b) for b in format(byte, "08b")])
        return np.array(bits, dtype=np.uint8)

    def bits_to_text(self, bits: np.ndarray) -> str:
        """Convert binary representation back to text."""
        # Pad to multiple of 8 if necessary
        remainder = len(bits) % 8
        if remainder:
            bits = np.append(bits, np.zeros(8 - remainder, dtype=np.uint8))

        # Convert bits to bytes
        text_bytes = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i : i + 8]
            byte_value = int("".join(map(str, byte_bits)), 2)
            text_bytes.append(byte_value)

        try:
            return bytes(text_bytes).decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            return bytes(text_bytes).decode("utf-8", errors="replace")

    def bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """Map bits to complex constellation symbols."""
        # Pad bits to multiple of bits_per_symbol
        remainder = len(bits) % self.bits_per_symbol
        if remainder:
            bits = np.append(
                bits, np.zeros(self.bits_per_symbol - remainder, dtype=np.uint8)
            )

        symbols = []
        for i in range(0, len(bits), self.bits_per_symbol):
            symbol_bits = bits[i : i + self.bits_per_symbol]
            # Convert to decimal index
            symbol_index = int("".join(map(str, symbol_bits)), 2)
            symbols.append(self.constellation[symbol_index])

        return np.array(symbols, dtype=np.complex64)

    # New method to insert pilots into OFDM symbols
    def _insert_pilots(self, data_symbols: np.ndarray) -> np.ndarray:
        """Insert pilots into OFDM symbols in frequency domain."""
        # Pad data to multiple of (N - num_pilots)
        symbols_per_ofdm = self.N - self.num_pilots
        remainder = len(data_symbols) % symbols_per_ofdm
        if remainder:
            padding = np.zeros(symbols_per_ofdm - remainder, dtype=np.complex64)
            data_symbols = np.append(data_symbols, padding)

        num_ofdm_symbols = len(data_symbols) // symbols_per_ofdm
        data_matrix = data_symbols.reshape(num_ofdm_symbols, symbols_per_ofdm)

        ofdm_symbols_freq = []
        for i in range(num_ofdm_symbols):
            # Create empty frequency domain symbol
            freq_symbol = np.zeros(self.N, dtype=np.complex64)

            # Insert data
            freq_symbol[self.data_indices] = data_matrix[i]

            # Insert pilots
            freq_symbol[self.pilot_indices] = self.pilot_sequence

            ofdm_symbols_freq.append(freq_symbol)

        return np.array(ofdm_symbols_freq, dtype=np.complex64)

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Demap symbols to bits using minimum distance detection."""
        bits = []
        for symbol in symbols:
            # Find closest constellation point
            distances = np.abs(symbol - self.constellation)
            closest_idx = np.argmin(distances)

            # Convert index to bits
            bit_string = format(closest_idx, f"0{self.bits_per_symbol}b")
            bits.extend([int(b) for b in bit_string])

        return np.array(bits, dtype=np.uint8)

    # Modified OFDM modulation
    def ofdm_modulate(self, symbols: np.ndarray) -> np.ndarray:
        """OFDM modulation with pilot insertion."""
        # Insert pilots into frequency domain symbols
        ofdm_symbols_freq = self._insert_pilots(symbols)
        num_ofdm_symbols = len(ofdm_symbols_freq)

        ofdm_signal = []
        for ofdm_sym in ofdm_symbols_freq:
            # IFFT to convert to time domain
            time_domain = np.fft.ifft(ofdm_sym)

            # Add cyclic prefix
            cp = time_domain[-self.N_cp :]
            time_domain_with_cp = np.concatenate([cp, time_domain])

            ofdm_signal.extend(time_domain_with_cp)

        return np.array(ofdm_signal, dtype=np.complex64)

    def channel_simulation(self, signal: np.ndarray) -> np.ndarray:
        """
        Simulate channel effects: multipath + AWGN noise

        Args:
            signal: Input signal

        Returns:
            Signal after channel effects
        """
        # Apply multipath channel
        # if len(self.channel_taps) > 1 or self.channel_taps[0] != 1.0:
        # Use 'full' convolution then truncate to avoid shifting artifacts
        convolved = np.convolve(signal, self.channel_taps, mode="full")
        # Take first len(signal) samples to maintain original length
        signal = convolved[: len(signal)]

        # Add AWGN noise
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (self.snr_db / 10))

        # Generate complex Gaussian noise
        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise = noise_real + 1j * noise_imag

        return signal + noise

    # Modified OFDM demodulation with channel estimation
    def ofdm_demodulate(
        self, received_signal: np.ndarray, original_num_symbols: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """OFDM demodulation with pilot-based channel estimation."""
        symbol_length = self.N + self.N_cp
        num_ofdm_symbols = len(received_signal) // symbol_length

        symbols_before_eq = []
        symbols_after_eq = []

        for i in range(num_ofdm_symbols):
            start_idx = i * symbol_length
            end_idx = start_idx + symbol_length
            ofdm_symbol = received_signal[start_idx:end_idx]

            # Remove cyclic prefix
            ofdm_symbol_no_cp = ofdm_symbol[self.N_cp : self.N_cp + self.N]

            # FFT to frequency domain
            freq_domain = np.fft.fft(ofdm_symbol_no_cp)
            symbols_before_eq.extend(freq_domain[self.data_indices])

            # Channel estimation using pilots
            received_pilots = freq_domain[self.pilot_indices]
            H_est_pilots = received_pilots / self.pilot_sequence

            # Linear interpolation for full channel response
            H_est = np.interp(
                np.arange(self.N), self.pilot_indices, H_est_pilots, period=self.N
            )

            # Zero-forcing equalization
            equalized = freq_domain / (H_est + 1e-10)
            symbols_after_eq.extend(equalized[self.data_indices])

        # Truncate to original number of symbols
        symbols_before_eq = np.array(
            symbols_before_eq[:original_num_symbols], dtype=np.complex64
        )
        symbols_after_eq = np.array(
            symbols_after_eq[:original_num_symbols], dtype=np.complex64
        )

        return symbols_before_eq, symbols_after_eq

    def simulate(self, input_text: str) -> Tuple[str, dict]:
        """
        Complete OFDM simulation from text input to text output.

        Args:
            input_text: Input text to transmit

        Returns:
            Tuple of (decoded_text, statistics_dict)
        """
        print(f"=== OFDM Simulation Started ===")
        print(f"Input text length: {len(input_text)} characters")

        # Step 1: Text to bits
        self.tx_bits = self.text_to_bits(input_text)
        print(f"Generated {len(self.tx_bits)} bits")

        # Step 2: Bits to symbols
        self.tx_symbols = self.bits_to_symbols(self.tx_bits)
        print(f"Generated {len(self.tx_symbols)} symbols")

        # Step 3: OFDM modulation
        tx_signal = self.ofdm_modulate(self.tx_symbols)
        print(f"OFDM signal length: {len(tx_signal)} samples")

        # Step 4: Channel simulation
        rx_signal = self.channel_simulation(tx_signal)

        # Step 5: OFDM demodulation
        self.rx_symbols_before_eq, self.rx_symbols_after_eq = self.ofdm_demodulate(
            rx_signal, len(self.tx_symbols)
        )

        # Step 6: Symbols to bits
        self.rx_bits = self.symbols_to_bits(self.rx_symbols_after_eq)

        # Step 7: Bits to text (truncate to original length)
        original_bit_length = len(self.tx_bits)
        decoded_text = self.bits_to_text(self.rx_bits[:original_bit_length])

        # Calculate statistics
        stats = self._calculate_statistics(input_text, decoded_text)

        print(f"=== Simulation Complete ===")
        print(f"Output text length: {len(decoded_text)} characters")
        print(f"BER: {stats['ber']:.6f}")
        print(f"Character errors: {stats['char_errors']}/{len(input_text)}")

        return decoded_text, stats

    def _calculate_statistics(self, original_text: str, decoded_text: str) -> dict:
        """Calculate BER and character error statistics."""
        # Bit Error Rate
        original_bits = self.tx_bits
        received_bits = self.rx_bits[: len(original_bits)]

        bit_errors = np.sum(original_bits != received_bits)
        ber = bit_errors / len(original_bits) if len(original_bits) > 0 else 0

        # Character errors
        min_length = min(len(original_text), len(decoded_text))
        char_errors = sum(
            1 for i in range(min_length) if original_text[i] != decoded_text[i]
        )

        # Add extra characters as errors
        char_errors += abs(len(original_text) - len(decoded_text))

        return {
            "ber": ber,
            "bit_errors": int(bit_errors),
            "total_bits": len(original_bits),
            "char_errors": char_errors,
            "total_chars": len(original_text),
            "char_error_rate": (
                char_errors / len(original_text) if len(original_text) > 0 else 0
            ),
        }

    def plot_constellation(self, save_path: Optional[str] = None):
        """Plot constellation diagrams before and after equalization."""
        if self.rx_symbols_before_eq is None or self.rx_symbols_after_eq is None:
            print("No simulation data available for constellation plot.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Original constellation
        ax1.scatter(
            self.constellation.real,
            self.constellation.imag,
            c="red",
            s=100,
            marker="x",
            linewidth=3,
            label="Ideal",
        )
        ax1.set_title("Ideal Constellation")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("In-phase")
        ax1.set_ylabel("Quadrature")
        ax1.legend()
        ax1.axis("equal")

        # Before equalization
        sample_size = min(1000, len(self.rx_symbols_before_eq))
        sample_indices = np.random.choice(
            len(self.rx_symbols_before_eq), sample_size, replace=False
        )
        sample_symbols = self.rx_symbols_before_eq[sample_indices]

        ax2.scatter(
            sample_symbols.real,
            sample_symbols.imag,
            c="blue",
            alpha=0.6,
            s=20,
            label="Received",
        )
        ax2.scatter(
            self.constellation.real,
            self.constellation.imag,
            c="red",
            s=100,
            marker="x",
            linewidth=3,
            label="Ideal",
        )
        ax2.set_title("Before Equalization")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("In-phase")
        ax2.set_ylabel("Quadrature")
        ax2.legend()
        ax2.axis("equal")

        # After equalization
        sample_symbols_eq = self.rx_symbols_after_eq[sample_indices]

        ax3.scatter(
            sample_symbols_eq.real,
            sample_symbols_eq.imag,
            c="green",
            alpha=0.6,
            s=20,
            label="Equalized",
        )
        ax3.scatter(
            self.constellation.real,
            self.constellation.imag,
            c="red",
            s=100,
            marker="x",
            linewidth=3,
            label="Ideal",
        )
        ax3.set_title("After Equalization")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("In-phase")
        ax3.set_ylabel("Quadrature")
        ax3.legend()
        ax3.axis("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Constellation diagram saved to: {save_path}")

        plt.show()  # Display popup window


def main():
    """Main simulation function with file I/O."""

    # Configuration
    input_file = "input_text.txt"
    output_file = "decoded_text.txt"
    constellation_plot = "constellation_diagram.png"

    # Create sample input file if it doesn't exist
    if not os.path.exists(input_file):
        sample_text = """Hello, World! This is a test message for OFDM simulation.
The quick brown fox jumps over the lazy dog.
OFDM (Orthogonal Frequency Division Multiplexing) is a digital modulation technique.
It splits data across multiple orthogonal subcarriers to improve spectral efficiency.
Special characters: áéíóú ñ ç 中文 русский العربية 🌟⭐🚀"""

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"Created sample input file: {input_file}")

    # Read input text
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Channel Configuration - Uncomment one of the following:
    # channel_taps = [1.0]  # AWGN only (ideal channel)
    # channel_taps = [1.0, 0.0, 0.0, 0.0]  # Identity with padding
    # channel_taps = [0, 0, 0, 1.0]  # Identity with delay
    # channel_taps = [0.8, 0.0, 0.0, 0.3]  # Multipath channel (default)
    # channel_taps = [1.0, 0.5, 0.25]  # Short multipath
    # channel_taps = [0.9, 0.3, 0.1]  # Moderate multipath
    channel_taps = [1.0, -0.2, 0.1, -0.05]  # Multipath with negative taps
    # channel_taps = [1.0, 1.0]

    # OFDM Simulation Parameters
    simulator = OFDMSimulator(
        N=64,  # Number of subcarriers
        N_cp=16,  # Cyclic prefix length
        modulation="QPSK",  # Modulation scheme, supported: BPSK, QPSK, 16QAM
        snr_db=10.0,  # SNR in dB
        channel_taps=channel_taps,
    )

    # Run simulation
    try:
        decoded_text, stats = simulator.simulate(input_text)

        # Save decoded text
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(decoded_text)
        print(f"Decoded text saved to: {output_file}")

        # Display results
        print("\n" + "=" * 50)
        print("SIMULATION RESULTS")
        print("=" * 50)
        print(f"Modulation: {simulator.modulation}")
        print(f"Subcarriers: {simulator.N}")
        print(f"Cyclic Prefix: {simulator.N_cp}")
        print(f"SNR: {simulator.snr_db} dB")
        print(f"Channel Taps: {simulator.channel_taps}")
        print("-" * 50)
        print(f"Bit Error Rate (BER): {stats['ber']:.6f}")
        print(f"Bit Errors: {stats['bit_errors']} / {stats['total_bits']}")
        print(f"Character Errors: {stats['char_errors']} / {stats['total_chars']}")
        print(f"Character Error Rate: {stats['char_error_rate']:.4f}")
        print("-" * 50)

        # Show text comparison (first 200 characters)
        print("ORIGINAL TEXT (first 200 chars):")
        print(repr(input_text[:200]))
        print("\nDECODED TEXT (first 200 chars):")
        print(repr(decoded_text[:200]))

        # Generate constellation diagram (popup window)
        simulator.plot_constellation()

        # Optionally save to file as well
        # simulator.plot_constellation(constellation_plot)

    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
