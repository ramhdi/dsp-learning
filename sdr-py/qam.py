#!/usr/bin/env python3
"""
Object-Oriented Digital Modulation Simulation Pipeline
Implements text-to-text transmission through modular processing blocks.

Pipeline: TextProcessor -> Modulator -> PulseShaper -> Channel -> MatchedFilter ->
          ChannelEstimator -> Equalizer -> Demodulator -> TextProcessor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List, Optional
import os


# =============================================================================
# TEXT PROCESSING BLOCK
# =============================================================================
class TextProcessor:
    """Handles text <-> bits conversion with UTF-8 encoding."""

    def __init__(self):
        self.last_text_length = 0

    def text_to_bits(self, text: str) -> np.ndarray:
        """Convert text to binary representation."""
        self.last_text_length = len(text)
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


# =============================================================================
# MODULATION BLOCK
# =============================================================================
class Modulator:
    """Digital modulator supporting BPSK, QPSK, and 16-QAM."""

    def __init__(self, modulation: str = "QPSK"):
        """
        Initialize modulator.

        Args:
            modulation: Modulation scheme ('BPSK', 'QPSK', '16QAM')
        """
        self.modulation = modulation
        self.bits_per_symbol = self._get_bits_per_symbol()
        self.constellation = self._generate_constellation()

    def _get_bits_per_symbol(self) -> int:
        """Get bits per symbol for the modulation scheme."""
        modulation_map = {"BPSK": 1, "QPSK": 2, "16QAM": 4}
        if self.modulation not in modulation_map:
            raise ValueError(f"Unsupported modulation: {self.modulation}")
        return modulation_map[self.modulation]

    def _generate_constellation(self) -> np.ndarray:
        """Generate constellation points for the modulation scheme."""
        if self.modulation == "BPSK":
            return np.array([1.0, -1.0], dtype=np.complex64)
        elif self.modulation == "QPSK":
            return np.array(
                [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex64
            ) / np.sqrt(2)
        elif self.modulation == "16QAM":
            # Standard 16-QAM Gray coding constellation
            gray_constellation = [
                -3 - 3j,
                -3 - 1j,
                -3 + 3j,
                -3 + 1j,  # 0000, 0001, 0010, 0011
                -1 - 3j,
                -1 - 1j,
                -1 + 3j,
                -1 + 1j,  # 0100, 0101, 0110, 0111
                +3 - 3j,
                +3 - 1j,
                +3 + 3j,
                +3 + 1j,  # 1000, 1001, 1010, 1011
                +1 - 3j,
                +1 - 1j,
                +1 + 3j,
                +1 + 1j,  # 1100, 1101, 1110, 1111
            ]
            return np.array(gray_constellation, dtype=np.complex64) / np.sqrt(10)

        return np.array([])

    def modulate(self, bits: np.ndarray) -> np.ndarray:
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
            symbol_index = int("".join(map(str, symbol_bits)), 2)
            symbols.append(self.constellation[symbol_index])

        return np.array(symbols, dtype=np.complex64)

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Demap symbols to bits using minimum distance detection."""
        bits = []
        for symbol in symbols:
            distances = np.abs(symbol - self.constellation)
            closest_idx = np.argmin(distances)
            bit_string = format(closest_idx, f"0{self.bits_per_symbol}b")
            bits.extend([int(b) for b in bit_string])

        return np.array(bits, dtype=np.uint8)


# =============================================================================
# PILOT SEQUENCE GENERATOR
# =============================================================================
class PilotGenerator:
    """Generates pilot sequences with good autocorrelation properties."""

    def __init__(self, pilot_length: int = 16):
        """
        Initialize pilot generator.

        Args:
            pilot_length: Number of pilot symbols
        """
        self.pilot_length = pilot_length
        self.pilot_symbols = self._generate_pilot_sequence()

    def _generate_pilot_sequence(self) -> np.ndarray:
        """Generate pilot sequence with good autocorrelation properties."""
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

            available_lengths = list(barker_sequences.keys())
            closest_length = min(
                available_lengths, key=lambda x: abs(x - self.pilot_length)
            )
            base_sequence = barker_sequences[closest_length]

            if len(base_sequence) < self.pilot_length:
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

    def prepend_pilots(self, data_symbols: np.ndarray) -> np.ndarray:
        """Prepend pilot symbols to data symbols."""
        return np.concatenate([self.pilot_symbols, data_symbols])

    def extract_pilots_and_data(
        self, symbols: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pilot and data symbols from received symbols."""
        pilots = symbols[: self.pilot_length]
        data = symbols[self.pilot_length :]
        return pilots, data


# =============================================================================
# PULSE SHAPING BLOCK
# =============================================================================
class PulseShaper:
    """Root Raised Cosine pulse shaping filter with upsampling."""

    def __init__(
        self, oversampling_factor: int = 8, rolloff: float = 0.35, span: int = 6
    ):
        """
        Initialize pulse shaper.

        Args:
            oversampling_factor: Upsampling factor (L)
            rolloff: RRC roll-off factor (β)
            span: RRC filter span in symbols
        """
        self.L = oversampling_factor
        self.beta = rolloff
        self.span = span
        self.rrc_filter = self._generate_rrc_filter()

    def _generate_rrc_filter(self) -> np.ndarray:
        """Generate Root Raised Cosine filter impulse response."""
        n_taps = self.span * self.L + 1
        t = np.arange(-self.span // 2, self.span // 2 + 1 / self.L, 1 / self.L)

        h = np.zeros_like(t, dtype=np.float64)
        tol = 1e-7

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

    def upsample_and_shape(self, symbols: np.ndarray) -> np.ndarray:
        """Apply pulse shaping with upsampling."""
        # Upsample: insert L-1 zeros between symbols
        upsampled = np.zeros(len(symbols) * self.L, dtype=np.complex64)
        upsampled[:: self.L] = symbols

        # Apply RRC filter
        shaped_signal = np.convolve(upsampled, self.rrc_filter, mode="full")
        return shaped_signal


# =============================================================================
# CHANNEL SIMULATION BLOCK
# =============================================================================
class Channel:
    """Multipath channel with AWGN noise simulation."""

    def __init__(
        self, channel_taps: Optional[List[complex]] = None, snr_db: float = 20.0
    ):
        """
        Initialize channel.

        Args:
            channel_taps: Channel impulse response (None for AWGN only)
            snr_db: Signal-to-noise ratio in dB
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

        # Normalize channel taps to unit power
        if channel_taps is not None:
            channel_array = np.array(channel_taps, dtype=np.complex64)
            channel_power = np.sum(np.abs(channel_array) ** 2)
            self.channel_taps = channel_array / np.sqrt(channel_power)
        else:
            self.channel_taps = np.array([1.0], dtype=np.complex64)

    def apply_channel(self, signal: np.ndarray) -> np.ndarray:
        """Apply multipath channel and AWGN noise."""
        # Apply multipath channel
        if len(self.channel_taps) > 1 or abs(self.channel_taps[0] - 1.0) > 1e-6:
            signal = np.convolve(signal, self.channel_taps, mode="full")

        # Add AWGN noise
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / self.snr_linear

        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise = noise_real + 1j * noise_imag

        return signal + noise


# =============================================================================
# MATCHED FILTER BLOCK
# =============================================================================
class MatchedFilter:
    """Matched filter with downsampling for symbol recovery."""

    def __init__(self, pulse_shaper: PulseShaper, channel: Channel):
        """
        Initialize matched filter.

        Args:
            pulse_shaper: Reference to pulse shaper for filter coefficients
            channel: Reference to channel for delay calculation
        """
        self.L = pulse_shaper.L
        self.rrc_filter = pulse_shaper.rrc_filter
        self.channel_taps = channel.channel_taps

    def filter_and_downsample(
        self, received_signal: np.ndarray, num_symbols: int
    ) -> np.ndarray:
        """Apply matched filter and downsample."""
        # Apply matched filter
        filtered = np.convolve(
            received_signal, np.conj(self.rrc_filter[::-1]), mode="full"
        )

        # Calculate delays
        rrc_delay = (len(self.rrc_filter) - 1) // 2
        channel_delay = (len(self.channel_taps) - 1) // 2
        total_delay = rrc_delay + channel_delay + rrc_delay

        # Start sampling at calculated delay
        if total_delay >= len(filtered):
            start_sample = len(filtered) // 2
        else:
            start_sample = total_delay

        # Downsample
        downsampled = filtered[start_sample :: self.L]

        # Ensure correct length
        if len(downsampled) < num_symbols:
            padding = np.zeros(num_symbols - len(downsampled), dtype=np.complex64)
            downsampled = np.concatenate([downsampled, padding])
        elif len(downsampled) > num_symbols:
            downsampled = downsampled[:num_symbols]

        return downsampled


# =============================================================================
# CHANNEL ESTIMATION BLOCK
# =============================================================================
class ChannelEstimator:
    """Pilot-based channel estimation using least squares."""

    def __init__(self, pilot_generator: PilotGenerator):
        """
        Initialize channel estimator.

        Args:
            pilot_generator: Reference to pilot generator for known pilots
        """
        self.pilot_symbols = pilot_generator.pilot_symbols

    def estimate_channel(self, received_pilots: np.ndarray) -> complex:
        """Estimate channel using pilot symbols via least squares."""
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


# =============================================================================
# MMSE EQUALIZER BLOCK
# =============================================================================
class MMSEEqualizer:
    """Minimum Mean Square Error equalizer."""

    def __init__(self, snr_db: float):
        """
        Initialize MMSE equalizer.

        Args:
            snr_db: Signal-to-noise ratio in dB
        """
        self.snr_linear = 10 ** (snr_db / 10.0)
        self.noise_variance = 1.0 / self.snr_linear

    def equalize(
        self, received_symbols: np.ndarray, channel_estimate: complex
    ) -> np.ndarray:
        """Apply MMSE equalization."""
        # Single-tap MMSE equalizer
        h_conj = np.conj(channel_estimate)
        h_power = np.abs(channel_estimate) ** 2
        mmse_weight = h_conj / (h_power + self.noise_variance)

        # Apply equalizer
        equalized = mmse_weight * received_symbols
        return equalized


# =============================================================================
# STATISTICS AND ANALYSIS
# =============================================================================
class StatisticsCalculator:
    """Calculate BER and character error statistics."""

    @staticmethod
    def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> dict:
        """Calculate bit error rate statistics."""
        min_bit_length = min(len(tx_bits), len(rx_bits))

        if min_bit_length > 0:
            bit_errors = np.sum(tx_bits[:min_bit_length] != rx_bits[:min_bit_length])
            bit_errors += abs(len(tx_bits) - len(rx_bits))
            ber = bit_errors / len(tx_bits)
        else:
            bit_errors = len(tx_bits)
            ber = 1.0

        return {
            "ber": ber,
            "bit_errors": int(bit_errors),
            "total_bits": len(tx_bits),
            "received_bits": len(rx_bits),
        }

    @staticmethod
    def calculate_character_errors(original_text: str, decoded_text: str) -> dict:
        """Calculate character error statistics."""
        min_length = min(len(original_text), len(decoded_text))
        char_errors = sum(
            1 for i in range(min_length) if original_text[i] != decoded_text[i]
        )
        char_errors += abs(len(original_text) - len(decoded_text))

        return {
            "char_errors": char_errors,
            "total_chars": len(original_text),
            "char_error_rate": (
                char_errors / len(original_text) if len(original_text) > 0 else 0
            ),
        }


# =============================================================================
# PLOTTING AND VISUALIZATION
# =============================================================================
class ResultsPlotter:
    """Handle all plotting and visualization."""

    def __init__(
        self, modulator: Modulator, channel: Channel, pulse_shaper: PulseShaper
    ):
        """
        Initialize plotter.

        Args:
            modulator: Reference to modulator for constellation
            channel: Reference to channel for frequency response
            pulse_shaper: Reference to pulse shaper for filter response
        """
        self.constellation = modulator.constellation
        self.modulation = modulator.modulation
        self.channel_taps = channel.channel_taps
        self.rrc_filter = pulse_shaper.rrc_filter

    def plot_results(
        self,
        rx_before_eq: np.ndarray,
        rx_after_eq: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Plot constellation diagrams and analysis."""
        if len(rx_before_eq) == 0 or len(rx_after_eq) == 0:
            print("No simulation data available for plotting.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Ideal constellation
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

        # 2. Before equalization
        sample_size = min(1000, len(rx_before_eq))
        if sample_size > 0:
            sample_indices = np.random.choice(
                len(rx_before_eq), sample_size, replace=False
            )
            sample_symbols = rx_before_eq[sample_indices]

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

        # 3. After equalization
        if sample_size > 0:
            sample_symbols_eq = rx_after_eq[sample_indices]

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

        # 4. Channel and RRC filter response
        w_channel, h_channel = signal.freqz(self.channel_taps, worN=512)
        w_rrc, h_rrc = signal.freqz(self.rrc_filter, worN=512)

        ax4.plot(
            w_channel / np.pi,
            20 * np.log10(np.abs(h_channel)),
            "b-",
            linewidth=2,
            label="Channel Response",
        )
        ax4.plot(
            w_rrc / np.pi,
            20 * np.log10(np.abs(h_rrc)),
            "g-",
            linewidth=2,
            label="RRC Filter",
        )
        ax4.set_xlabel("Normalized Frequency (×π rad/sample)")
        ax4.set_ylabel("Magnitude (dB)", color="b")
        ax4.tick_params(axis="y", labelcolor="b")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # Channel phase on secondary y-axis
        ax4_phase = ax4.twinx()
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

        ax4.set_title("Frequency Response Analysis")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Analysis plots saved to: {save_path}")

        plt.show()


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================
def load_text_file(filename: str) -> str:
    """Load text from file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        return ""
    except Exception as e:
        print(f"Error reading input file: {e}")
        return ""


def save_text_file(filename: str, text: str):
    """Save text to file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to: {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


def create_sample_input_file(filename: str):
    """Create a sample input file if it doesn't exist."""
    if not os.path.exists(filename):
        sample_text = """Hello, World! This is a test message for digital modulation simulation.
The quick brown fox jumps over the lazy dog.
Digital modulation uses pulse shaping and equalization for robust communication.
It employs pilot sequences for channel estimation and MMSE equalizers.
Special characters: áéíóú ñ ç 中文 русский العربية 🌟⭐🚀"""

        save_text_file(filename, sample_text)
        print(f"Created sample input file: {filename}")


# =============================================================================
# MAIN SIMULATION PIPELINE
# =============================================================================
def main():
    """Main simulation pipeline demonstrating OOP modular design."""

    # Configuration
    modulation = "BPSK"
    input_file = "input_text.txt"
    output_file = "decoded_text.txt"
    snr_db = 10

    # channel_taps = [1.0]  # AWGN only (ideal channel)
    # channel_taps = [0.8, 0.6j]  # Simple two-path channel
    # channel_taps = [0.7, 0.5, -0.3j]  # Light frequency-selective fading
    # channel_taps = [0.6, 0.4, 0.3, 0.2]  # Moderate multipath (exponential decay)
    # channel_taps = [1.0, 0, 0, 0.9]  # Delayed strong reflection
    channel_taps = [0.5, -0.3, 0.4j, -0.2j]  # Complex fading with phase variations

    # Create sample input if needed
    create_sample_input_file(input_file)

    # Load input text
    input_text = load_text_file(input_file)
    if not input_text:
        return

    # Initialize processing blocks
    print("=== Initializing Processing Blocks ===")

    # Text processor
    text_processor = TextProcessor()

    # Modulator
    modulator = Modulator(modulation)

    # Pilot generator
    pilot_generator = PilotGenerator(pilot_length=16)

    # Pulse shaper
    pulse_shaper = PulseShaper(oversampling_factor=8, rolloff=0.35, span=6)

    # Channel - normalized automatically
    channel = Channel(channel_taps=channel_taps, snr_db=snr_db)

    # Matched filter
    matched_filter = MatchedFilter(pulse_shaper, channel)

    # Channel estimator
    channel_estimator = ChannelEstimator(pilot_generator)

    # MMSE equalizer
    equalizer = MMSEEqualizer(snr_db=snr_db)

    # Statistics calculator
    stats_calc = StatisticsCalculator()

    # Results plotter
    plotter = ResultsPlotter(modulator, channel, pulse_shaper)

    print("=== Running Simulation Pipeline ===")

    # TRANSMITTER PIPELINE
    print(f"Input text length: {len(input_text)} characters")

    # 1. Text to bits
    tx_bits = text_processor.text_to_bits(input_text)
    print(f"Generated {len(tx_bits)} bits")

    # 2. Bits to symbols
    data_symbols = modulator.modulate(tx_bits)
    print(f"Generated {len(data_symbols)} data symbols")

    # 3. Prepend pilots
    tx_symbols = pilot_generator.prepend_pilots(data_symbols)
    print(f"Total symbols (pilots + data): {len(tx_symbols)}")

    # 4. Pulse shaping
    tx_signal = pulse_shaper.upsample_and_shape(tx_symbols)
    print(f"Pulse-shaped signal length: {len(tx_signal)} samples")

    # CHANNEL
    # 5. Channel simulation
    rx_signal = channel.apply_channel(tx_signal)
    print(f"After channel: {len(rx_signal)} samples")

    # RECEIVER PIPELINE
    # 6. Matched filtering
    rx_symbols = matched_filter.filter_and_downsample(rx_signal, len(tx_symbols))
    print(f"Received symbols after matched filtering: {len(rx_symbols)}")

    # 7. Extract pilots and data
    rx_pilots, rx_data_before_eq = pilot_generator.extract_pilots_and_data(rx_symbols)
    print(f"Data symbols before equalization: {len(rx_data_before_eq)}")

    # 8. Channel estimation
    channel_estimate = channel_estimator.estimate_channel(rx_pilots)
    print(f"Channel estimate: {channel_estimate:.4f}")

    # 9. MMSE equalization
    rx_data_after_eq = equalizer.equalize(rx_data_before_eq, channel_estimate)
    print(f"Data symbols after equalization: {len(rx_data_after_eq)}")

    # 10. Demodulation
    rx_bits = modulator.demodulate(rx_data_after_eq)
    print(f"Received bits: {len(rx_bits)} (expected: {len(tx_bits)})")

    # 11. Bits to text
    original_bit_length = len(tx_bits)
    decoded_text = text_processor.bits_to_text(rx_bits[:original_bit_length])

    # Save results
    save_text_file(output_file, decoded_text)

    # Calculate statistics
    ber_stats = stats_calc.calculate_ber(tx_bits, rx_bits)
    char_stats = stats_calc.calculate_character_errors(input_text, decoded_text)

    # Display results
    print("\n" + "=" * 50)
    print("DIGITAL MODULATION SIMULATION RESULTS")
    print("=" * 50)
    print(f"Modulation: {modulator.modulation}")
    print(f"Oversampling Factor: {pulse_shaper.L}")
    print(f"RRC Roll-off: {pulse_shaper.beta}")
    print(f"Pilot Length: {pilot_generator.pilot_length}")
    print(f"SNR: {channel.snr_db} dB")
    print(f"Channel Taps: {channel.channel_taps}")
    print(f"Channel Estimate: {channel_estimate:.4f}")
    print("-" * 50)
    print(f"Bit Error Rate (BER): {ber_stats['ber']:.6f}")
    print(f"Bit Errors: {ber_stats['bit_errors']} / {ber_stats['total_bits']}")
    print(
        f"Received Bits: {ber_stats['received_bits']} (vs {ber_stats['total_bits']} transmitted)"
    )
    print(
        f"Character Errors: {char_stats['char_errors']} / {char_stats['total_chars']}"
    )
    print(f"Character Error Rate: {char_stats['char_error_rate']:.4f}")
    print("-" * 50)

    # Show text comparison
    print("ORIGINAL TEXT (first 200 chars):")
    print(repr(input_text[:200]))
    print("\nDECODED TEXT (first 200 chars):")
    print(repr(decoded_text[:200]))

    # Generate plots
    plotter.plot_results(rx_data_before_eq, rx_data_after_eq)


if __name__ == "__main__":
    main()
