import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
import wfdb
import matplotlib.pyplot as plt
from ppg_cleaner.ppg_cleaner.preprocessing import *  # Import all functions from your library

def fetch_signals(file_path, pn_dir, fs):
    """
    Fetch PPG and ABP signals from the specified WFDB record.
    
    Parameters:
    - file_path: Path to the WFDB file.
    - pn_dir: PhysioNet directory containing the WFDB files.
    - fs: Expected sampling frequency.
    
    Returns:
    - ppg_signal: Extracted PPG signal.
    - abp_signal: Extracted ABP signal.
    - fields: Metadata fields from the WFDB file.
    """
    try:
        signals, fields = wfdb.rdsamp(file_path, pn_dir=pn_dir, channel_names=["PLETH", "ABP"])
        if fields["fs"] != fs or "PLETH" not in fields["sig_name"] or "ABP" not in fields["sig_name"]:
            raise ValueError("Signal does not match expected format or sampling frequency.")
        
        ppg_idx = fields["sig_name"].index("PLETH")
        abp_idx = fields["sig_name"].index("ABP")

        ppg_signal = signals[:, ppg_idx]
        abp_signal = signals[:, abp_idx]
        
        return ppg_signal, abp_signal, fields
    except Exception as e:
        print(f"Error fetching signals: {e}")
        return None, None, None

def plot_signals(original_signal, processed_signal, title, fs=125):
    """
    Plot original and processed signals for comparison.
    
    Parameters:
    - original_signal: The original unprocessed signal.
    - processed_signal: The signal after applying a function.
    - title: Title for the comparison plot.
    - fs: Sampling frequency (default: 125 Hz).
    """
    time = np.arange(len(original_signal)) / fs  # Time axis in seconds

    plt.figure(figsize=(12, 6))
    plt.plot(time, original_signal, label="Original Signal", alpha=0.7)
    plt.plot(time, processed_signal, label="Processed Signal", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    file_path = "30/3000063/3000063_0006"  # Adjust to match your dataset path
    pn_dir = "mimic3wdb/30/3000063"
    fs = 125  # Sampling frequency

    # Fetch signals
    ppg_signal, abp_signal, fields = fetch_signals(file_path, pn_dir, fs)
    if ppg_signal is None or abp_signal is None:
        print("Failed to fetch signals.")
        return
    
    ppg_cleaned, _ = remove_invalid_values(ppg_signal, "remove", abp_signal)
    plot_signals(ppg_signal, ppg_cleaned, "Invalid Values Removed (PPG)", fs)

    # # Test preprocessing functions
    # print("Testing Preprocessing Functions...")

    # # 1. Z-Score Normalization
    # ppg_zscore = zscore_normalization(ppg_signal)
    # plot_signals(ppg_signal, ppg_zscore, "Z-Score Normalization (PPG)", fs)

    # # 2. Min-Max Normalization
    # ppg_minmax = min_max_normalization(ppg_signal, 0, 1)
    # plot_signals(ppg_signal, ppg_minmax, "Min-Max Normalization (PPG)", fs)

    # # 3. Rescale Signal
    # ppg_rescaled = rescale_signal(ppg_signal)
    # plot_signals(ppg_signal, ppg_rescaled, "Rescale Signal (PPG)", fs)

    # # 4. Clip Signal
    # ppg_clipped = clip_signal(ppg_signal, lower_bound=0, upper_bound=2)
    # plot_signals(ppg_signal, ppg_clipped, "Clipped Signal (PPG)", fs)

    # # 5. Remove Invalid Values
    # ppg_cleaned, _ = remove_invalid_values(ppg_signal, replace_with="interpolation")
    # plot_signals(ppg_signal, ppg_cleaned, "Invalid Values Removed (PPG)", fs)

    # # 6. Baseline Wander Removal
    # ppg_baseline_removed = baseline_wander_removal(ppg_signal, fs)
    # plot_signals(ppg_signal, ppg_baseline_removed, "Baseline Wander Removed (PPG)", fs)

    # # 7. Downsample Signal
    # ppg_downsampled = downsample_signal(ppg_signal, original_fs=fs, target_fs=50)
    # plot_signals(ppg_signal, ppg_downsampled, "Downsampled Signal (PPG)", fs)

if __name__ == "__main__":
    main()

