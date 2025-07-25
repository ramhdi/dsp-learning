import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def simulate_ula_signal(
    angle_deg: float,
    num_antennas: int = 8,
    antenna_spacing: float = 0.5,
    snr_db: float = 10.0,
    num_snapshots: int = 100,
    signal_freq: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate ULA signal with complex noise."""
    wavelength = 1.0
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi / wavelength

    # Steering vector
    antenna_indices = np.arange(num_antennas)
    a = np.exp(1j * k * antenna_spacing * antenna_indices * np.sin(angle_rad))
    a = a.reshape(-1, 1)

    # Complex signal
    t = np.arange(num_snapshots)
    s = np.exp(1j * 2 * np.pi * signal_freq * t)

    # Complex noise
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(num_antennas, num_snapshots)
        + 1j * np.random.randn(num_antennas, num_snapshots)
    )

    x = a @ s.reshape(1, -1) + noise
    return x, a


def bartlett_beamformer(
    x: np.ndarray,
    antenna_spacing: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Bartlett spectrum."""
    num_antennas = x.shape[0]
    angle_grid_deg = np.linspace(-90, 90, 180)
    theta_grid = np.deg2rad(angle_grid_deg)
    wavelength = 1.0
    k = 2 * np.pi / wavelength

    R = (x @ x.conj().T) / x.shape[1]
    spectrum = np.zeros_like(theta_grid, dtype=np.float64)

    for i, theta in enumerate(theta_grid):
        a = np.exp(1j * k * antenna_spacing * np.arange(num_antennas) * np.sin(theta))
        a = a.reshape(-1, 1)
        spectrum[i] = np.abs(a.conj().T @ R @ a).item()

    return angle_grid_deg, spectrum


def capon_beamformer(
    x: np.ndarray,
    antenna_spacing: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Capon (Minimum Variance) spectrum."""
    num_antennas = x.shape[0]
    angle_grid_deg = np.linspace(-90, 90, 180)
    theta_grid = np.deg2rad(angle_grid_deg)
    wavelength = 1.0
    k = 2 * np.pi / wavelength

    # Compute covariance matrix
    R = (x @ x.conj().T) / x.shape[1]
    R_inv = np.linalg.pinv(R)  # Pseudoinverse for stability

    spectrum = np.zeros_like(theta_grid, dtype=np.float64)

    for i, theta in enumerate(theta_grid):
        a = np.exp(1j * k * antenna_spacing * np.arange(num_antennas) * np.sin(theta))
        a = a.reshape(-1, 1)
        # Capon spectrum calculation
        denominator = np.abs(a.conj().T @ R_inv @ a).item()
        spectrum[i] = 1 / denominator if denominator != 0 else np.nan

    return angle_grid_deg, spectrum


def plot_results(
    x: np.ndarray,
    angle_grid_deg: np.ndarray,
    bartlett_spectrum: np.ndarray,
    capon_spectrum: np.ndarray,  # Added Capon spectrum
    true_angle_deg: float,
    num_antennas: int,
    num_snapshots: int = 100,
) -> None:
    """Plot signals and both Bartlett & Capon spectra in one figure."""
    plt.figure(figsize=(12, 10))  # Slightly taller figure

    # Plot real part of received signals (first 4 antennas for clarity)
    plt.subplot(2, 1, 1)
    t = np.arange(num_snapshots)
    for i in range(min(4, num_antennas)):
        plt.plot(t, np.real(x[i, :]), label=f"Antenna {i+1} (Real Part)")
    plt.xlabel("Time Sample")
    plt.ylabel("Amplitude")
    plt.title(f"Real Part of Received Signals (True AoA: {true_angle_deg}°)")
    plt.legend()
    plt.grid(True)

    # Plot both spectra
    plt.subplot(2, 1, 2)
    # Normalize and convert to dB
    bartlett_db = 10 * np.log10(bartlett_spectrum / np.max(bartlett_spectrum))
    capon_db = 10 * np.log10(capon_spectrum / np.max(capon_spectrum))

    plt.plot(angle_grid_deg, bartlett_db, "b-", label="Bartlett Spectrum")
    plt.plot(angle_grid_deg, capon_db, "g--", label="Capon Spectrum")
    plt.axvline(
        true_angle_deg, color="r", linestyle="--", label=f"True AoA: {true_angle_deg}°"
    )
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Normalized Power (dB)")
    plt.title("AoA Estimation: Bartlett vs Capon Beamformers")
    plt.grid(True)
    plt.legend()
    # plt.ylim([-20, 10])  # Set consistent dB range

    plt.tight_layout()
    plt.show()


def main():
    # Parameters
    true_angle_deg = 43.2
    num_antennas = 4
    antenna_spacing = 0.5
    snr_db = 20.0

    # Simulate signal
    x, a_true = simulate_ula_signal(
        angle_deg=true_angle_deg,
        num_antennas=num_antennas,
        antenna_spacing=antenna_spacing,
        snr_db=snr_db,
    )

    # Compute Bartlett spectrum
    angle_grid_deg, bartlett_spectrum = bartlett_beamformer(x, antenna_spacing)

    # NEW: Compute Capon spectrum
    angle_grid_deg, capon_spectrum = capon_beamformer(x, antenna_spacing)

    # Find peak AoAs
    bartlett_peak = angle_grid_deg[np.argmax(bartlett_spectrum)]
    capon_peak = angle_grid_deg[np.argmax(capon_spectrum)]

    print(f"True AoA: {true_angle_deg:.1f}°")
    print(f"Bartlett Estimated AoA: {bartlett_peak:.1f}°")
    print(f"Capon Estimated AoA: {capon_peak:.1f}°")

    # Plot everything (modified to include both spectra)
    plot_results(
        x,
        angle_grid_deg,
        bartlett_spectrum,
        capon_spectrum,  # Added Capon spectrum
        true_angle_deg,
        num_antennas,
    )


if __name__ == "__main__":
    main()
