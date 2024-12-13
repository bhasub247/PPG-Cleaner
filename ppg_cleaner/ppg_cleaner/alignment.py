import numpy as np
from scipy.signal import find_peaks, correlate

def find_peaks_and_max_correlation(ppg_signal, abp_signal, fs, segment_duration=2, overlap=1, peak_distance=50):
    """
    Find the window with the highest correlation between PPG and ABP peaks.

    Parameters:
    - ppg_signal: PPG signal (NumPy array).
    - abp_signal: ABP signal (NumPy array).
    - fs: Sampling frequency (Hz).
    - segment_duration: Duration of each segment (in minutes).
    - overlap: Overlap between consecutive segments (in minutes).
    - peak_distance: Minimum distance between consecutive peaks (in samples).

    Returns:
    - best_ppg_segment: PPG signal segment with the highest correlation.
    - best_abp_segment: ABP signal segment with the highest correlation.
    - max_corr: Maximum correlation value.
    """
    segment_samples = int(segment_duration * 60 * fs)
    overlap_samples = int(overlap * 60 * fs)
    step_size = segment_samples - overlap_samples

    max_corr = -np.inf
    best_ppg_segment = None
    best_abp_segment = None

    # Iterate over the signal with sliding windows
    for start_idx in range(0, len(ppg_signal) - segment_samples + 1, step_size):
        end_idx = start_idx + segment_samples

        # Extract windowed signals
        ppg_segment = ppg_signal[start_idx:end_idx]
        abp_segment = abp_signal[start_idx:end_idx]

        # Detect peaks in the PPG and ABP signals
        ppg_peaks, _ = find_peaks(ppg_segment, distance=peak_distance)
        abp_peaks, _ = find_peaks(abp_segment, distance=peak_distance)

        # Extract peak values for cross-correlation
        ppg_peak_values = ppg_segment[ppg_peaks]
        abp_peak_values = abp_segment[abp_peaks]

        # Compute cross-correlation of the peaks
        if len(ppg_peak_values) > 1 and len(abp_peak_values) > 1:
            correlation = correlate(ppg_peak_values, abp_peak_values, mode="valid")
            max_corr_in_window = np.max(correlation)

            # Update if this window has the highest correlation
            if max_corr_in_window > max_corr:
                max_corr = max_corr_in_window
                best_ppg_segment = ppg_segment
                best_abp_segment = abp_segment

    return best_ppg_segment, best_abp_segment, max_corr