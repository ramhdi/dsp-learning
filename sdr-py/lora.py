import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# ========================
# CONFIGURABLE PARAMETERS
# ========================
"""
Independent variables with typical ranges:
- SF (Spreading Factor): Integer between 7-12 (higher = longer range, lower data rate)
- B (Bandwidth): 125e3, 250e3, or 500e3 Hz (higher = faster data rate)
- SNR (Signal-to-Noise Ratio): Any float (dB), typical range: -20 to 20 dB
- TEXT: Any string (ASCII characters work best)
- PLOT_SYMBOL_INDEX: Index of symbol to visualize (0 to len(symbols)-1)
"""
SF = 8  # Spreading Factor (7-12)
B = 125e3  # Bandwidth (Hz) - 125kHz, 250kHz, or 500kHz
SNR = 10  # Signal-to-Noise Ratio (dB)
TEXT = "LoRaTest"  # Text to transmit
PLOT_SYMBOL_INDEX = 2  # Which symbol to visualize in detail

# Derived parameters (do not change directly)
M = 2**SF  # Number of samples per symbol (fixed by SF)
T = M / B  # Symbol duration (seconds)
fs = B  # Sampling rate = Bandwidth (Hz)


def generate_base_chirp() -> np.ndarray:
    """Generate base up-chirp sweeping from -B/2 to +B/2 Hz."""
    n = np.arange(M)  # Sample indices [0, 1, ..., M-1]
    # Phase calculation: ϕ = π(n² - n*M)/M
    phase = np.pi * (n**2 - n * M) / M
    return np.exp(1j * phase)  # Complex chirp signal


def text_to_symbols(text: str) -> np.ndarray:
    """Convert text to LoRa symbols (each symbol represents SF bits)."""
    # Convert text to bytes (UTF-8 encoding)
    byte_data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)

    # Convert bytes to bits (MSB first)
    bits = np.unpackbits(byte_data)

    # Pad with zeros if not multiple of SF
    if len(bits) % SF != 0:
        padding = np.zeros(SF - (len(bits) % SF), dtype=np.uint8)
        bits = np.concatenate((bits, padding))

    # Reshape bits into SF-bit symbols
    symbols = bits.reshape(-1, SF)

    # Convert binary arrays to integer symbols
    return np.packbits(symbols, axis=1, bitorder="big").flatten() >> (8 - SF)


def modulate(symbols: np.ndarray, base_chirp: np.ndarray) -> np.ndarray:
    """Modulate symbols into LoRa chirps using cyclic shifts."""
    signal = np.array([], dtype=np.complex64)
    for symbol in symbols:
        # Apply cyclic shift to base chirp
        shifted_chirp = np.roll(base_chirp, symbol)
        signal = np.concatenate((signal, shifted_chirp))
    return signal


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add complex AWGN noise to signal."""
    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Calculate noise power based on SNR
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


def demodulate(signal: np.ndarray, base_chirp: np.ndarray) -> List[int]:
    """Demodulate LoRa signal back to symbols using FFT peak detection."""
    num_symbols = len(signal) // M
    symbols_recovered = []

    for i in range(num_symbols):
        # Extract one symbol duration
        segment = signal[i * M : (i + 1) * M]

        # Dechirping: Multiply by conjugate of base chirp
        dechirped = segment * np.conj(base_chirp)

        # FFT to find peak
        fft_result = np.fft.fft(dechirped)
        k_hat = np.argmax(np.abs(fft_result))
        symbols_recovered.append(k_hat)

    return symbols_recovered


def symbols_to_text(symbols: List[int], original_bit_length: int) -> str:
    """Convert symbols back to text (handling padding removal)."""
    # Convert symbols to bits
    bits = []
    for symbol in symbols:
        # Convert each symbol to SF bits (MSB first)
        bits.extend([(symbol >> i) & 1 for i in range(SF - 1, -1, -1)])

    # Remove padding based on original bit length
    bits = bits[:original_bit_length]

    # Convert bits to bytes
    byte_array = np.packbits(bits)

    # Convert bytes to text
    return byte_array.tobytes().decode("utf-8", errors="replace")


def calculate_errors(original: str, received: str) -> Tuple[int, int, float, float]:
    """Calculate bit error rate (BER) and character error rate (CER)."""
    # Convert to bytes for accurate bit-level comparison
    orig_bytes = original.encode("utf-8")
    recv_bytes = received.encode("utf-8")

    # Ensure equal length by padding with zeros
    max_len = max(len(orig_bytes), len(recv_bytes))
    orig_bytes_padded = orig_bytes.ljust(max_len, b"\x00")
    recv_bytes_padded = recv_bytes.ljust(max_len, b"\x00")

    # Convert to bit arrays
    orig_bits = np.unpackbits(np.frombuffer(orig_bytes_padded, dtype=np.uint8))
    recv_bits = np.unpackbits(np.frombuffer(recv_bytes_padded, dtype=np.uint8))

    # Calculate bit errors
    bit_errors = np.sum(orig_bits != recv_bits)
    total_bits = len(orig_bits)
    ber = bit_errors / total_bits if total_bits > 0 else 0.0

    # Calculate character errors
    char_errors = sum(
        1
        for a, b in zip(
            original.ljust(len(received), " "), received.ljust(len(original), " ")
        )
        if a != b
    )
    cer = char_errors / len(original) if original else 0.0

    return bit_errors, char_errors, ber, cer


def plot_results(
    base_chirp: np.ndarray,
    modulated: np.ndarray,
    noisy_signal: np.ndarray,
    symbols: np.ndarray,
    symbols_recovered: List[int],
    plot_symbol_idx: int,
) -> None:
    """Visualize key aspects of LoRa modulation/demodulation."""
    plt.figure(figsize=(15, 10))

    # 1. Base Chirp (Time Domain)
    plt.subplot(3, 2, 1)
    t = np.arange(M) / fs * 1000  # ms
    plt.plot(t, np.real(base_chirp))
    plt.title("Base Chirp (Real Part)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 2. Base Chirp (Frequency Domain)
    plt.subplot(3, 2, 2)
    freq = np.fft.fftshift(np.fft.fftfreq(M, 1 / fs)) / 1000  # kHz
    fft_base = np.fft.fftshift(np.abs(np.fft.fft(base_chirp)))
    plt.plot(freq, fft_base)
    plt.title("Base Chirp Spectrum")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    # 3. Modulated Symbol (Selected)
    plt.subplot(3, 2, 3)
    symbol_start = plot_symbol_idx * M
    symbol_end = (plot_symbol_idx + 1) * M
    symbol_wave = modulated[symbol_start:symbol_end]
    plt.plot(t, np.real(symbol_wave))
    plt.title(f"Modulated Symbol (k={symbols[plot_symbol_idx]})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 4. Noisy Symbol (Same symbol with noise)
    plt.subplot(3, 2, 4)
    noisy_symbol = noisy_signal[symbol_start:symbol_end]
    plt.plot(t, np.real(noisy_symbol))
    plt.title(f"Noisy Symbol (SNR={SNR}dB)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 5. Dechirped FFT Comparison
    plt.subplot(3, 2, 5)
    dechirped_clean = symbol_wave * np.conj(base_chirp)
    fft_clean = np.abs(np.fft.fft(dechirped_clean))

    dechirped_noisy = noisy_symbol * np.conj(base_chirp)
    fft_noisy = np.abs(np.fft.fft(dechirped_noisy))

    bins = np.arange(M)
    plt.stem(bins, fft_clean, "b-", markerfmt="bo", label="Clean")
    plt.stem(bins, fft_noisy, "r-", markerfmt="ro", label="Noisy")
    plt.title(
        f"Dechirped FFT Comparison (True k={symbols[plot_symbol_idx]}, Detected k={symbols_recovered[plot_symbol_idx]})"
    )
    plt.xlabel("FFT Bin Index")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    # 6. Constellation Diagram (First 1000 samples)
    plt.subplot(3, 2, 6)
    plt.scatter(
        np.real(noisy_signal[:1000]), np.imag(noisy_signal[:1000]), alpha=0.3, s=5
    )
    plt.title("Noisy Signal Constellation")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis("equal")

    plt.tight_layout()
    plt.show()


# ======================
# MAIN SIMULATION
# ======================
if __name__ == "__main__":
    # 1. Precompute components
    base_chirp = generate_base_chirp()

    # 2. Transmitter
    symbols = text_to_symbols(TEXT)
    original_bit_length = len(TEXT.encode("utf-8")) * 8
    modulated = modulate(symbols, base_chirp)

    # 3. Channel
    noisy_signal = add_noise(modulated, SNR)

    # 4. Receiver
    symbols_recovered = demodulate(noisy_signal, base_chirp)
    text_recovered = symbols_to_text(symbols_recovered, original_bit_length)

    # 5. Error analysis
    bit_errors, char_errors, ber, cer = calculate_errors(TEXT, text_recovered)

    # 6. Output results
    print(f"Original Text: {TEXT}")
    print(f"Decoded Text:  {text_recovered}")
    print(f"\nBit Errors: {bit_errors}/{original_bit_length} ({ber*100:.2f}%)")
    print(f"Char Errors: {char_errors}/{len(TEXT)} ({cer*100:.2f}%)")

    # 7. Visualization
    plot_results(
        base_chirp,
        modulated,
        noisy_signal,
        symbols,
        symbols_recovered,
        PLOT_SYMBOL_INDEX,
    )
