#!/usr/bin/env python3
"""
Complete Digital Modulation Simulation
Implements text-to-text transmission through single-carrier digital modulation
with pulse shaping, multipath channels, MMSE equalization, and comprehensive analysis.

Based on: Pilot Sequence → PSK/QAM → RRC Pulse Shaping → Channel → MMSE Equalizer → Demod
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List, Optional
import os


class DigitalModulationSimulator:
    """Complete single-carrier digital modulation simulation with comprehensive analysis."""

    def __init__(
        self,
        modulation: str = "QPSK",  # Modulation scheme
        oversampling_factor: int = 8,  # Upsampling factor (L)
        rrc_rolloff: float = 0.35,  # RRC roll-off factor (β)
        rrc_span: int = 6,  # RRC filter span in symbols
        pilot_length: int = 16,  # Number of pilot symbols
        snr_db: float = 20.0,  # Signal-to-noise ratio in dB
        channel_taps: Optional[List[complex]] = None,  # Channel impulse response
        equalizer_taps: int = 11,  # MMSE equalizer length
    ):
        """
        Initialize digital modulation simulator parameters.

        Args:
            modulation: Modulation scheme ('BPSK', 'QPSK', '16QAM')
            oversampling_factor: Upsampling factor for pulse shaping
            rrc_rolloff: Root Raised Cosine roll-off factor (0 < β ≤ 1)
            rrc_span: RRC filter span in symbols
            pilot_length: Number of pilot symbols for channel estimation
            snr_db: Signal-to-noise ratio in dB
            channel_taps: Channel impulse response (None for AWGN only)
            equalizer_taps: Number of taps in MMSE equalizer (odd number)
        """
        # Parameter validation
        if not (0 < rrc_rolloff <= 1):
            raise ValueError(
                f"RRC roll-off factor must be in (0, 1], got {rrc_rolloff}"
            )
        if oversampling_factor < 2:
            raise ValueError(
                f"Oversampling factor must be ≥ 2, got {oversampling_factor}"
            )
        if pilot_length < 1:
            raise ValueError(f"Pilot length must be ≥ 1, got {pilot_length}")

        self.modulation = modulation
        self.L = oversampling_factor  # Upsampling factor
        self.beta = rrc_rolloff  # RRC roll-off
        self.rrc_span = rrc_span
        self.pilot_length = pilot_length
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

        # Normalize channel taps to unit power for fair comparison
        if channel_taps is not None:
            channel_array = np.array(channel_taps, dtype=np.complex64)
            channel_power = np.sum(np.abs(channel_array) ** 2)
            self.channel_taps = channel_array / np.sqrt(channel_power)
            print(
                f"Channel normalized: original power = {channel_power:.4f}, normalized power = {np.sum(np.abs(self.channel_taps)**2):.4f}"
            )
        else:
            self.channel_taps = np.array([1.0], dtype=np.complex64)

        self.equalizer_taps = (
            equalizer_taps if equalizer_taps % 2 == 1 else equalizer_taps + 1
        )

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

        # Generate RRC filter
        self.rrc_filter = self._generate_rrc_filter()

        # Generate pilot sequence (BPSK Barker-like sequence for good autocorrelation)
        self.pilot_symbols = self._generate_pilot_sequence()

        # Statistics tracking
        self.tx_bits = np.array([])
        self.rx_bits = np.array([])
        self.tx_symbols = np.array([])
        self.tx_signal = np.array([])
        self.rx_symbols_before_eq = np.array([])
        self.rx_symbols_after_eq = np.array([])
        self.channel_estimate = np.array([])

    def _generate_bpsk_constellation(self) -> np.ndarray:
        """Generate BPSK constellation points."""
        return np.array([1.0, -1.0], dtype=np.complex64)

    def _generate_qpsk_constellation(self) -> np.ndarray:
        """Generate QPSK constellation points with Gray coding."""
        return np.array(
            [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j],  # 00  # 01  # 10  # 11
            dtype=np.complex64,
        ) / np.sqrt(2)

    def _generate_16qam_constellation(self) -> np.ndarray:
        """Generate 16-QAM constellation points with proper Gray coding."""
        # Standard 16-QAM Gray coding constellation
        # Bit mapping: [b3 b2 b1 b0] where b3,b2 = I-channel, b1,b0 = Q-channel
        gray_constellation = [
            -3 - 3j,  # 0000
            -3 - 1j,  # 0001
            -3 + 3j,  # 0010
            -3 + 1j,  # 0011
            -1 - 3j,  # 0100
            -1 - 1j,  # 0101
            -1 + 3j,  # 0110
            -1 + 1j,  # 0111
            +3 - 3j,  # 1000
            +3 - 1j,  # 1001
            +3 + 3j,  # 1010
            +3 + 1j,  # 1011
            +1 - 3j,  # 1100
            +1 - 1j,  # 1101
            +1 + 3j,  # 1110
            +1 + 1j,  # 1111
        ]

        return np.array(gray_constellation, dtype=np.complex64) / np.sqrt(
            10
        )  # Normalize

    def _generate_rrc_filter(self) -> np.ndarray:
        """Generate Root Raised Cosine filter impulse response."""
        # Create time vector
        n_taps = self.rrc_span * self.L + 1
        t = np.arange(-self.rrc_span // 2, self.rrc_span // 2 + 1 / self.L, 1 / self.L)

        # RRC formula with numerical stability
        h = np.zeros_like(t, dtype=np.float64)
        tol = 1e-7  # Numerical tolerance for float comparisons

        for i, time in enumerate(t):
            if abs(time) < tol:  # t ≈ 0
                h[i] = 1 + self.beta * (4 / np.pi - 1)
            elif abs(abs(time) - 1 / (4 * self.beta)) < tol:  # t ≈ ±1/(4β)
                h[i] = (self.beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * self.beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * self.beta))
                )
            else:  # General case
                numerator = np.sin(
                    np.pi * time * (1 - self.beta)
                ) + 4 * self.beta * time * np.cos(np.pi * time * (1 + self.beta))
                denominator = np.pi * time * (1 - (4 * self.beta * time) ** 2)
                h[i] = numerator / denominator

        # Normalize for unit energy
        h = h / np.sqrt(np.sum(h**2))
        return h.astype(np.complex64)

    def _generate_pilot_sequence(self) -> np.ndarray:
        """Generate pilot sequence with good autocorrelation properties."""
        # Use a modified Barker sequence or Zadoff-Chu-like sequence
        if self.pilot_length <= 13:
            # Known Barker sequences
            barker_sequences = {
                2: [1, -1],
                3: [1, 1, -1],
                4: [1, 1, -1, 1],
                5: [1, 1, 1, -1, 1],
                7: [1, 1, 1, -1, -1, 1, -1],
                11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
                13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
            }

            # Find closest Barker sequence
            available_lengths = list(barker_sequences.keys())
            closest_length = min(
                available_lengths, key=lambda x: abs(x - self.pilot_length)
            )
            base_sequence = barker_sequences[closest_length]

            # Extend or truncate to desired length
            if len(base_sequence) < self.pilot_length:
                # Repeat and truncate
                repetitions = (self.pilot_length // len(base_sequence)) + 1
                extended = (base_sequence * repetitions)[: self.pilot_length]
                pilots = np.array(extended, dtype=np.complex64)
            else:
                pilots = np.array(
                    base_sequence[: self.pilot_length], dtype=np.complex64
                )
        else:
            # For longer sequences, use BPSK with good autocorrelation
            np.random.seed(42)  # Fixed seed for reproducible pilots
            pilots = np.random.choice([1.0, -1.0], self.pilot_length).astype(
                np.complex64
            )

        return pilots

    def text_to_bits(self, text: str) -> np.ndarray:
        """Convert text to binary representation."""
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

    def pulse_shape_and_upsample(self, symbols: np.ndarray) -> np.ndarray:
        """Apply pulse shaping with upsampling."""
        # Upsample: insert L-1 zeros between symbols
        upsampled = np.zeros(len(symbols) * self.L, dtype=np.complex64)
        upsampled[:: self.L] = symbols

        # Apply RRC filter using 'full' convolution
        shaped_signal = np.convolve(upsampled, self.rrc_filter, mode="full")

        return shaped_signal

    def channel_simulation(self, signal: np.ndarray) -> np.ndarray:
        """Simulate multipath channel and AWGN noise."""
        # Apply multipath channel using 'full' convolution
        if len(self.channel_taps) > 1 or abs(self.channel_taps[0] - 1.0) > 1e-6:
            signal = np.convolve(signal, self.channel_taps, mode="full")

        # Add AWGN noise
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / self.snr_linear

        # Generate complex Gaussian noise
        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise = noise_real + 1j * noise_imag

        return signal + noise

    def matched_filter_and_downsample(
        self, received_signal: np.ndarray, num_symbols: int
    ) -> np.ndarray:
        """Apply matched filter and downsample with exact delay calculation."""
        # Apply matched filter using 'full' convolution
        filtered = np.convolve(
            received_signal, np.conj(self.rrc_filter[::-1]), mode="full"
        )

        # Calculate exact delays:
        # - TX RRC filter delay: (filter_length - 1) / 2 samples
        # - Channel delay: (channel_length - 1) / 2 samples
        # - RX matched filter delay: (filter_length - 1) / 2 samples
        rrc_delay = (len(self.rrc_filter) - 1) // 2
        channel_delay = (len(self.channel_taps) - 1) // 2

        # Total delay in samples
        total_delay = rrc_delay + channel_delay + rrc_delay

        print(
            f"Calculated delays - RRC: {rrc_delay}, Channel: {channel_delay}, Total: {total_delay}"
        )

        # Start sampling at the calculated delay point
        if total_delay >= len(filtered):
            print(
                f"Warning: Total delay {total_delay} >= signal length {len(filtered)}"
            )
            # Emergency fallback - start from the middle
            start_sample = len(filtered) // 2
        else:
            start_sample = total_delay

        # Downsample: take every L-th sample starting from the calculated point
        downsampled = filtered[start_sample :: self.L]

        print(f"Downsampled signal length: {len(downsampled)}, expected: {num_symbols}")

        # Ensure we have exactly the expected number of symbols
        if len(downsampled) < num_symbols:
            # Pad with zeros if we don't have enough samples
            padding = np.zeros(num_symbols - len(downsampled), dtype=np.complex64)
            downsampled = np.concatenate([downsampled, padding])
            print(f"Padded {len(padding)} symbols")
        elif len(downsampled) > num_symbols:
            # Truncate to expected number of symbols
            downsampled = downsampled[:num_symbols]
            print(f"Truncated to {num_symbols} symbols")

        return downsampled

    def estimate_channel(self, received_pilots: np.ndarray) -> np.ndarray:
        """Estimate channel using pilot symbols via least squares."""
        # Simple least squares channel estimation
        # For a single-tap equivalent channel: h_est = r_pilot / s_pilot
        # For multi-tap channel, this gives an approximation

        if len(received_pilots) != len(self.pilot_symbols):
            min_len = min(len(received_pilots), len(self.pilot_symbols))
            received_pilots = received_pilots[:min_len]
            pilot_ref = self.pilot_symbols[:min_len]
        else:
            pilot_ref = self.pilot_symbols

        # Least squares estimate
        h_est = np.sum(received_pilots * np.conj(pilot_ref)) / np.sum(
            np.abs(pilot_ref) ** 2
        )

        return h_est

    def mmse_equalizer(
        self, received_symbols: np.ndarray, channel_estimate: complex
    ) -> np.ndarray:
        """Apply MMSE equalization."""
        # For simplicity, use a single-tap MMSE equalizer
        # MMSE weight: w = h* / (|h|^2 + σ²)
        noise_variance = 1.0 / self.snr_linear

        # Single-tap MMSE equalizer
        h_conj = np.conj(channel_estimate)
        h_power = np.abs(channel_estimate) ** 2
        mmse_weight = h_conj / (h_power + noise_variance)

        # Apply equalizer
        equalized = mmse_weight * received_symbols

        return equalized

    def simulate(self, input_text: str) -> Tuple[str, dict]:
        """
        Complete digital modulation simulation from text input to text output.

        Args:
            input_text: Input text to transmit

        Returns:
            Tuple of (decoded_text, statistics_dict)
        """
        print(f"=== Digital Modulation Simulation Started ===")
        print(f"Input text length: {len(input_text)} characters")

        # Step 1: Text to bits
        self.tx_bits = self.text_to_bits(input_text)
        print(f"Generated {len(self.tx_bits)} bits")

        # Step 2: Bits to symbols
        data_symbols = self.bits_to_symbols(self.tx_bits)
        print(f"Generated {len(data_symbols)} data symbols")

        # Step 3: Prepend pilot sequence
        self.tx_symbols = np.concatenate([self.pilot_symbols, data_symbols])
        print(f"Total symbols (pilots + data): {len(self.tx_symbols)}")

        # Step 4: Pulse shaping and upsampling
        self.tx_signal = self.pulse_shape_and_upsample(self.tx_symbols)
        print(
            f"Pulse-shaped signal length: {len(self.tx_signal)} samples (after full convolution)"
        )

        # Step 5: Channel simulation
        rx_signal = self.channel_simulation(self.tx_signal)
        print(f"After channel: {len(rx_signal)} samples")

        # Step 6: Matched filter and downsampling
        rx_symbols = self.matched_filter_and_downsample(rx_signal, len(self.tx_symbols))
        print(f"Received symbols after matched filtering: {len(rx_symbols)}")

        # Step 7: Split pilots and data
        rx_pilots = rx_symbols[: self.pilot_length]
        rx_data_before_eq = rx_symbols[self.pilot_length :]
        self.rx_symbols_before_eq = rx_data_before_eq
        print(f"Data symbols before equalization: {len(rx_data_before_eq)}")

        # Step 8: Channel estimation
        self.channel_estimate = self.estimate_channel(rx_pilots)
        print(f"Channel estimate: {self.channel_estimate:.4f}")

        # Step 9: MMSE equalization
        self.rx_symbols_after_eq = self.mmse_equalizer(
            rx_data_before_eq, self.channel_estimate
        )
        print(f"Data symbols after equalization: {len(self.rx_symbols_after_eq)}")

        # Step 10: Symbols to bits
        self.rx_bits = self.symbols_to_bits(self.rx_symbols_after_eq)
        print(f"Received bits: {len(self.rx_bits)} (expected: {len(self.tx_bits)})")

        # Step 11: Bits to text (truncate to original length)
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
        # Bit Error Rate - handle length mismatches gracefully
        original_bits = self.tx_bits
        received_bits = self.rx_bits

        # Compare only the overlapping portion
        min_bit_length = min(len(original_bits), len(received_bits))

        if min_bit_length > 0:
            bit_errors = np.sum(
                original_bits[:min_bit_length] != received_bits[:min_bit_length]
            )
            # Add missing bits as errors
            bit_errors += abs(len(original_bits) - len(received_bits))
            ber = bit_errors / len(original_bits)
        else:
            bit_errors = len(original_bits)
            ber = 1.0

        # Character errors
        min_length = min(len(original_text), len(decoded_text))
        char_errors = sum(
            1 for i in range(min_length) if original_text[i] != decoded_text[i]
        )
        char_errors += abs(len(original_text) - len(decoded_text))

        return {
            "ber": ber,
            "bit_errors": int(bit_errors),
            "total_bits": len(original_bits),
            "received_bits": len(received_bits),
            "char_errors": char_errors,
            "total_chars": len(original_text),
            "char_error_rate": (
                char_errors / len(original_text) if len(original_text) > 0 else 0
            ),
        }

    def plot_results(self, save_path: Optional[str] = None):
        """Plot constellation diagrams, channel response, and signal analysis."""
        if len(self.rx_symbols_before_eq) == 0 or len(self.rx_symbols_after_eq) == 0:
            print("No simulation data available for plotting.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Ideal constellation (top-left)
        ax1.scatter(
            self.constellation.real,
            self.constellation.imag,
            c="red",
            s=150,
            marker="x",
            linewidth=4,
            label="Ideal",
        )
        ax1.set_title(f"Ideal {self.modulation} Constellation")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("In-phase")
        ax1.set_ylabel("Quadrature")
        ax1.legend()
        ax1.axis("equal")

        # 2. Before equalization (top-right)
        sample_size = min(1000, len(self.rx_symbols_before_eq))
        if sample_size > 0:
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
                s=150,
                marker="x",
                linewidth=4,
                label="Ideal",
            )
            ax2.set_title("Before Equalization")
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel("In-phase")
            ax2.set_ylabel("Quadrature")
            ax2.legend()
            ax2.axis("equal")

        # 3. After equalization (bottom-left)
        if sample_size > 0:
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
                s=150,
                marker="x",
                linewidth=4,
                label="Ideal",
            )
            ax3.set_title("After MMSE Equalization")
            ax3.grid(True, alpha=0.3)
            ax3.set_xlabel("In-phase")
            ax3.set_ylabel("Quadrature")
            ax3.legend()
            ax3.axis("equal")

        # 4. Channel and RRC filter response (bottom-right)
        # Frequency response of channel
        w_channel, h_channel = signal.freqz(self.channel_taps, worN=512)
        # Frequency response of RRC filter
        w_rrc, h_rrc = signal.freqz(self.rrc_filter, worN=512)

        ax4_mag = ax4
        ax4_mag.plot(
            w_channel / np.pi,
            20 * np.log10(np.abs(h_channel)),
            "b-",
            linewidth=2,
            label="Channel Response",
        )
        ax4_mag.plot(
            w_rrc / np.pi,
            20 * np.log10(np.abs(h_rrc)),
            "g-",
            linewidth=2,
            label="RRC Filter",
        )
        ax4_mag.set_xlabel("Normalized Frequency (×π rad/sample)")
        ax4_mag.set_ylabel("Magnitude (dB)", color="b")
        ax4_mag.tick_params(axis="y", labelcolor="b")
        ax4_mag.grid(True, alpha=0.3)
        ax4_mag.legend()

        # Plot channel phase on secondary y-axis
        ax4_phase = ax4_mag.twinx()
        phase_deg = np.angle(h_channel) * 180 / np.pi
        ax4_phase.plot(
            w_channel / np.pi,
            phase_deg,
            "r--",
            linewidth=2,
            alpha=0.8,
            label="Channel Phase",
        )
        ax4_phase.set_ylabel("Phase (degrees)", color="r")
        ax4_phase.tick_params(axis="y", labelcolor="r")

        ax4_mag.set_title("Frequency Response Analysis")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Analysis plots saved to: {save_path}")

        plt.show()


def main():
    """Main simulation function with file I/O."""

    # Configuration
    input_file = "input_text.txt"
    output_file = "decoded_text.txt"
    analysis_plot = "digital_modulation_analysis.png"

    # Create sample input file if it doesn't exist
    if not os.path.exists(input_file):
        sample_text = """Hello, World! This is a test message for digital modulation simulation.
The quick brown fox jumps over the lazy dog.
Digital modulation uses pulse shaping and equalization for robust communication.
It employs pilot sequences for channel estimation and MMSE equalizers.
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

    # Channel Configuration - All channels will be automatically normalized to unit power
    # Original unnormalized examples (will be normalized automatically):

    # channel_taps = [1.0]  # AWGN only (ideal channel)
    # channel_taps = [0.8, 0.6j]  # Simple two-path channel
    # channel_taps = [0.7, 0.5, -0.3j]  # Light frequency-selective fading
    # channel_taps = [0.6, 0.4, 0.3, 0.2]  # Moderate multipath (exponential decay)
    # channel_taps = [1.0, 0, 0, 0.9]  # Delayed strong reflection
    channel_taps = [0.5, -0.3, 0.4j, -0.2j]  # Complex fading with phase variations

    # Note: All channels above will be automatically normalized so that Σ|h[n]|² = 1
    # This ensures fair comparison across different channel types

    # Digital Modulation Simulation Parameters
    simulator = DigitalModulationSimulator(
        modulation="QPSK",  # Modulation scheme: BPSK, QPSK, 16QAM
        oversampling_factor=8,  # Upsampling factor for pulse shaping
        rrc_rolloff=0.35,  # RRC roll-off factor
        rrc_span=6,  # RRC filter span in symbols
        pilot_length=16,  # Number of pilot symbols
        snr_db=20.0,  # SNR in dB
        channel_taps=channel_taps,  # Channel impulse response
        equalizer_taps=11,  # MMSE equalizer length
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
        print("DIGITAL MODULATION SIMULATION RESULTS")
        print("=" * 50)
        print(f"Modulation: {simulator.modulation}")
        print(f"Oversampling Factor: {simulator.L}")
        print(f"RRC Roll-off: {simulator.beta}")
        print(f"Pilot Length: {simulator.pilot_length}")
        print(f"SNR: {simulator.snr_db} dB")
        print(f"Channel Taps: {simulator.channel_taps}")
        print(f"Channel Estimate: {simulator.channel_estimate:.4f}")
        print("-" * 50)
        print(f"Bit Error Rate (BER): {stats['ber']:.6f}")
        print(f"Bit Errors: {stats['bit_errors']} / {stats['total_bits']}")
        print(
            f"Received Bits: {stats['received_bits']} (vs {stats['total_bits']} transmitted)"
        )
        print(f"Character Errors: {stats['char_errors']} / {stats['total_chars']}")
        print(f"Character Error Rate: {stats['char_error_rate']:.4f}")
        print("-" * 50)

        # Show text comparison (first 200 characters)
        print("ORIGINAL TEXT (first 200 chars):")
        print(repr(input_text[:200]))
        print("\nDECODED TEXT (first 200 chars):")
        print(repr(decoded_text[:200]))

        # Generate analysis plots (popup window)
        simulator.plot_results()

        # Optionally save to file as well
        # simulator.plot_results(analysis_plot)

    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
