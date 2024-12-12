import numpy as np
from ppg_cleaner.preprocessing import (
    remove_invalid_values,
    clip_signal,
    baseline_wander_removal
)
from ppg_cleaner.filtering import bandpass_filter
from ppg_cleaner.artifact_removal import hampel_filter
from ppg_cleaner.alignment import extract_max_correlation_window

def combined_pipeline(ppg_signal, abp_signal, fs, 
                      lower_bound=40, upper_bound=200, 
                      bandpass_low=0.5, bandpass_high=8.0, 
                      window_duration=2, hampel_window_size=10, hampel_threshold=3):
    """
    A combined pipeline function to clean and preprocess PPG and ABP signals.

    Parameters:
    - ppg_signal: Raw PPG signal (NumPy array).
    - abp_signal: Raw ABP signal (NumPy array).
    - fs: Sampling frequency (Hz).
    - lower_bound: Lower bound for clipping (default: 40).
    - upper_bound: Upper bound for clipping (default: 200).
    - bandpass_low: Lower cutoff frequency for bandpass filter (Hz).
    - bandpass_high: Upper cutoff frequency for bandpass filter (Hz).
    - window_duration: Duration of the window (in minutes) for maximum correlation.
    - hampel_window_size: Window size for Hampel filtering.
    - hampel_threshold: Threshold for Hampel filtering.

    Returns:
    - cleaned_ppg: Cleaned PPG signal (NumPy array).
    - cleaned_abp: Cleaned ABP signal (NumPy array).
    """
    # Step 1: Remove invalid values (NaN and Inf) from both signals
    ppg_signal, abp_signal = remove_invalid_values(ppg_signal, abp_signal)

    # Step 2: Clip the signals to physiological bounds
    ppg_signal = clip_signal(ppg_signal, lower_bound=-np.inf, upper_bound=np.inf)  # PPG often doesn't need clipping
    abp_signal = clip_signal(abp_signal, lower_bound=lower_bound, upper_bound=upper_bound)

    # Step 3: Remove baseline wander using detrending
    ppg_signal = baseline_wander_removal(ppg_signal, fs)
    abp_signal = baseline_wander_removal(abp_signal, fs)

    # Step 4: Apply bandpass filter to remove irrelevant frequencies
    ppg_signal = bandpass_filter(ppg_signal, fs, bandpass_low, bandpass_high)
    abp_signal = bandpass_filter(abp_signal, fs, bandpass_low, bandpass_high)

    # Step 5: Remove artifacts using Hampel filter
    ppg_signal = hampel_filter(ppg_signal, hampel_window_size, hampel_threshold)
    abp_signal = hampel_filter(abp_signal, hampel_window_size, hampel_threshold)

    # Step 6: Extract the window with the maximum correlation between PPG and ABP
    cleaned_ppg, cleaned_abp = extract_max_correlation_window(ppg_signal, abp_signal, fs, duration=window_duration)

    return cleaned_ppg, cleaned_abp