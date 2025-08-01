#!/usr/bin/env python3
"""
SIMD Convolution Verification Script

This script analyzes the output from the Rust SIMD convolution example
to verify that the SIMD implementation produces the same results as
the scalar implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def load_data(filename="simd_comparison.csv"):
    """Load the comparison data from CSV file."""
    try:
        df = pd.read_csv(filename)
        print(f"✅ Loaded {len(df)} samples from {filename}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found.")
        print("   Make sure to run the Rust program first: cargo run --bin simd")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def analyze_errors(df):
    """Perform statistical analysis of the differences between scalar and SIMD."""
    differences = df["difference"].values

    print("\n📊 Error Analysis:")
    print("=" * 50)
    print(f"Total samples: {len(differences)}")
    print(f"Maximum error: {np.max(differences):.2e}")
    print(f"Mean error: {np.mean(differences):.2e}")
    print(f"RMS error: {np.sqrt(np.mean(differences**2)):.2e}")
    print(f"Standard deviation: {np.std(differences):.2e}")
    print(f"99.9th percentile: {np.percentile(differences, 99.9):.2e}")

    # Check if errors are within expected floating-point precision
    machine_epsilon = np.finfo(np.float32).eps
    max_expected_error = machine_epsilon * 100  # Allow some accumulation

    print(f"\nFloating-point precision check:")
    print(f"Machine epsilon (f32): {machine_epsilon:.2e}")
    print(f"Expected max error: {max_expected_error:.2e}")

    if np.max(differences) < max_expected_error:
        print("✅ SIMD implementation is correct (within floating-point precision)")
    else:
        print(
            "⚠️ SIMD implementation may have issues (errors exceed expected precision)"
        )

    return differences


def verify_convolution_properties(df):
    """Verify that the convolution behaves as expected."""
    print("\n🔍 Convolution Properties Verification:")
    print("=" * 50)

    # Check if both outputs have the same DC gain for the moving average filter
    scalar_mean = np.mean(df["scalar_output"])
    simd_mean = np.mean(df["simd_output"])
    input_mean = np.mean(df["input"])

    print(f"Input signal mean: {input_mean:.6f}")
    print(f"Scalar output mean: {scalar_mean:.6f}")
    print(f"SIMD output mean: {simd_mean:.6f}")
    print(f"Mean difference: {abs(scalar_mean - simd_mean):.2e}")

    # For a moving average filter, output mean should equal input mean
    dc_gain_error_scalar = abs(scalar_mean - input_mean)
    dc_gain_error_simd = abs(simd_mean - input_mean)

    print(f"DC gain error (scalar): {dc_gain_error_scalar:.2e}")
    print(f"DC gain error (SIMD): {dc_gain_error_simd:.2e}")

    if dc_gain_error_scalar < 1e-6 and dc_gain_error_simd < 1e-6:
        print("✅ Both implementations preserve DC gain correctly")
    else:
        print("⚠️ DC gain preservation may have issues")


def create_comparison_plots(df):
    """Create visualizations comparing scalar and SIMD results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SIMD vs Scalar Convolution Comparison", fontsize=16)

    # Plot 1: Time series comparison (first 100 samples)
    n_plot = min(100, len(df))
    axes[0, 0].plot(
        df["time"][:n_plot], df["input"][:n_plot], "b-", alpha=0.7, label="Input"
    )
    axes[0, 0].plot(
        df["time"][:n_plot], df["scalar_output"][:n_plot], "r-", label="Scalar Output"
    )
    axes[0, 0].plot(
        df["time"][:n_plot],
        df["simd_output"][:n_plot],
        "g--",
        alpha=0.8,
        label="SIMD Output",
    )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title("Time Series Comparison (First 100 samples)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Error over time
    axes[0, 1].plot(df["time"], df["difference"], "r-", alpha=0.7)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("|Scalar - SIMD|")
    axes[0, 1].set_title("Absolute Error Over Time")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale("log")

    # Plot 3: Scatter plot (scalar vs SIMD)
    axes[1, 0].scatter(df["scalar_output"], df["simd_output"], alpha=0.6, s=1)

    # Perfect correlation line
    min_val = min(df["scalar_output"].min(), df["simd_output"].min())
    max_val = max(df["scalar_output"].max(), df["simd_output"].max())
    axes[1, 0].plot(
        [min_val, max_val], [min_val, max_val], "r--", label="Perfect Correlation"
    )

    axes[1, 0].set_xlabel("Scalar Output")
    axes[1, 0].set_ylabel("SIMD Output")
    axes[1, 0].set_title("Scalar vs SIMD Scatter Plot")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Error histogram
    differences = df["difference"].values
    axes[1, 1].hist(differences, bins=50, alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Absolute Error")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Error Distribution")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("simd_verification_plots.png", dpi=300, bbox_inches="tight")
    print(f"\n📈 Plots saved to: simd_verification_plots.png")
    plt.show()


def calculate_correlation(df):
    """Calculate correlation between scalar and SIMD outputs."""
    correlation = np.corrcoef(df["scalar_output"], df["simd_output"])[0, 1]
    print(f"\n📈 Correlation Analysis:")
    print(f"Correlation coefficient: {correlation:.10f}")

    if correlation > 0.9999999:
        print("✅ Excellent correlation - SIMD implementation is highly accurate")
    elif correlation > 0.999:
        print("✅ Good correlation - SIMD implementation is acceptable")
    else:
        print("⚠️ Poor correlation - SIMD implementation may have significant errors")

    return correlation


def frequency_domain_analysis(df):
    """Compare frequency domain characteristics."""
    print(f"\n🔊 Frequency Domain Analysis:")
    print("=" * 50)

    # Compute FFTs
    scalar_fft = np.fft.fft(df["scalar_output"])
    simd_fft = np.fft.fft(df["simd_output"])

    # Compare magnitude and phase
    mag_diff = np.abs(np.abs(scalar_fft) - np.abs(simd_fft))
    phase_diff = np.abs(np.angle(scalar_fft) - np.angle(simd_fft))

    print(f"Max magnitude difference: {np.max(mag_diff):.2e}")
    print(f"Max phase difference: {np.max(phase_diff):.2e} radians")

    # Focus on significant frequency components (above noise floor)
    mag_threshold = np.max(np.abs(scalar_fft)) * 1e-6
    significant_bins = np.abs(scalar_fft) > mag_threshold

    if np.any(significant_bins):
        sig_mag_diff = np.max(mag_diff[significant_bins])
        sig_phase_diff = np.max(phase_diff[significant_bins])
        print(f"Max significant magnitude difference: {sig_mag_diff:.2e}")
        print(f"Max significant phase difference: {sig_phase_diff:.2e} radians")

        if sig_mag_diff < 1e-6 and sig_phase_diff < 1e-6:
            print("✅ Frequency domain characteristics match well")
        else:
            print("⚠️ Frequency domain differences detected")


def main():
    """Main verification function."""
    print("🔍 SIMD Convolution Verification")
    print("=" * 50)

    # Load data
    df = load_data()

    # Print basic info
    print(f"\nData overview:")
    print(f"  Time range: {df['time'].min():.3f} to {df['time'].max():.3f} seconds")
    print(f"  Input range: {df['input'].min():.3f} to {df['input'].max():.3f}")
    print(
        f"  Scalar output range: {df['scalar_output'].min():.3f} to {df['scalar_output'].max():.3f}"
    )
    print(
        f"  SIMD output range: {df['simd_output'].min():.3f} to {df['simd_output'].max():.3f}"
    )

    # Perform analyses
    differences = analyze_errors(df)
    verify_convolution_properties(df)
    correlation = calculate_correlation(df)
    frequency_domain_analysis(df)

    # Create plots
    create_comparison_plots(df)

    # Final summary
    print(f"\n📋 Summary:")
    print("=" * 50)

    max_error = np.max(differences)
    machine_epsilon = np.finfo(np.float32).eps * 100

    if max_error < machine_epsilon and correlation > 0.9999999:
        print("🎉 SIMD implementation is CORRECT!")
        print("   - Errors are within floating-point precision")
        print("   - Correlation is excellent")
        print("   - Ready for production use")
    elif max_error < 1e-6 and correlation > 0.999:
        print("✅ SIMD implementation is acceptable")
        print("   - Minor precision differences detected")
        print("   - Suitable for most applications")
    else:
        print("❌ SIMD implementation has issues!")
        print("   - Significant errors detected")
        print("   - Needs debugging")

    print(f"\nFiles generated:")
    print(f"  📊 simd_verification_plots.png - Visual comparison")
    print(f"  📈 Check the plots for detailed analysis")


if __name__ == "__main__":
    main()
