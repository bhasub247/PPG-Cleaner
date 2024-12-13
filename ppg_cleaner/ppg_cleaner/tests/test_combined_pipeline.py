import numpy as np
import wfdb
from ppg_cleaner.preprocessing import remove_invalid_values, baseline_wander_removal, remove_out_of_range_bp
from ppg_cleaner.filtering import bandpass_filter
from ppg_cleaner.artifact_removal import hampel_filter
from ppg_cleaner.alignment import find_peaks_and_max_correlation
from ppg_cleaner.combined_pipeline import combined_pipeline
import matplotlib.pyplot as plt

def test_combined_pipeline():
    """
    Test the combined_pipeline function with extracted PPG and ABP signals.
    """
    # File path and PhysioNet directory
    file_path = 'mimic3wdb/30/3000063/3000063_0006'
    pn_dir = 'mimic3wdb/30/3000063'
    fs = 125  # Sampling frequency

    # Extract PPG and ABP signals
    try:
        signals, fields = wfdb.rdsamp(file_path, pn_dir=pn_dir, channel_names=["PLETH", "ABP"])
        if fields["fs"] != fs or "PLETH" not in fields["sig_name"] or "ABP" not in fields["sig_name"]:
            print("Signal does not match expected format or sampling frequency.")
            return

        ppg_idx = fields["sig_name"].index("PLETH")
        abp_idx = fields["sig_name"].index("ABP")

        ppg_signal = signals[:, ppg_idx]
        abp_signal = signals[:, abp_idx]

    except Exception as e:
        print(f"Error fetching signals: {e}")
        return

    # Test the combined pipeline
    try:
        cleaned_ppg, cleaned_abp = combined_pipeline(
            ppg_signal, abp_signal, fs,
            lower_bound=40, upper_bound=200,
            bandpass_low=0.5, bandpass_high=8.0,
            segment_duration=2, overlap=1, peak_distance=50,
            hampel_window_size=10, hampel_threshold=3
        )

        # Plot the results
        time_ppg = np.arange(len(ppg_signal)) / fs
        time_abp = np.arange(len(abp_signal)) / fs

        plt.figure(figsize=(12, 6))

        # Original PPG Signal
        plt.subplot(2, 2, 1)
        plt.plot(time_ppg, ppg_signal, label="Original PPG")
        plt.title("Original PPG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

        # Cleaned PPG Signal
        plt.subplot(2, 2, 2)
        plt.plot(time_ppg[:len(cleaned_ppg)], cleaned_ppg, label="Cleaned PPG", color="blue")
        plt.title("Cleaned PPG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

        # Original ABP Signal
        plt.subplot(2, 2, 3)
        plt.plot(time_abp, abp_signal, label="Original ABP")
        plt.title("Original ABP Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (mmHg)")
        plt.grid()
        plt.legend()

        # Cleaned ABP Signal
        plt.subplot(2, 2, 4)
        plt.plot(time_abp[:len(cleaned_abp)], cleaned_abp, label="Cleaned ABP", color="red")
        plt.title("Cleaned ABP Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (mmHg)")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in pipeline execution: {e}")

# Run the test
if __name__ == "__main__":
    test_combined_pipeline()