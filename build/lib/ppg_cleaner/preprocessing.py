import numpy as np
from scipy.signal import detrend
from scipy.signal import resample

def zscore_normalization(signal):
    """
    Normalize the signal to have zero mean and unit variance.
    
    Parameters:
    - signal: NumPy array representing the input signal.
    
    Returns:
    - Normalized signal.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        raise ValueError("Standard deviation is zero, cannot perform z-score normalization.")
    return (signal - mean) / std


def min_max_normalization(signal, min_val, max_val):
    """
    Normalize the signal to a specific range [min_val, max_val].
    
    Parameters:
    - signal: NumPy array representing the input signal.
    - min_val: Desired minimum value of the normalized signal.
    - max_val: Desired maximum value of the normalized signal.
    
    Returns:
    - Normalized signal.
    """
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max - signal_min == 0:
        raise ValueError("Signal has zero range, cannot perform min-max normalization.")
    return ((signal - signal_min) / (signal_max - signal_min)) * (max_val - min_val) + min_val


def rescale_signal(signal):
    """
    Rescale the signal based on its maximum absolute value.
    Useful for normalizing signals with large variations.
    
    Parameters:
    - signal: NumPy array representing the input signal.
    
    Returns:
    - Rescaled signal.
    """
    max_abs_value = np.max(np.abs(signal))
    if max_abs_value == 0:
        raise ValueError("Signal has zero maximum absolute value, cannot rescale.")
    return signal / max_abs_value

def clip_signal(signal, lower_bound, upper_bound):
    """
    Clip the signal to a specified range.
    
    Parameters:
    - signal: Input signal (list or NumPy array).
    - lower_bound: Lower bound for the signal values.
    - upper_bound: Upper bound for the signal values.
    
    Returns:
    - Clipped signal (NumPy array).
    """

    # Convert to NumPy array if not already
    signal = np.array(signal)

    # Clip the signal values
    clipped_signal = np.clip(signal, lower_bound, upper_bound)

    return clipped_signal

def remove_invalid_values(ppg_signal=None, abp_signal=None):
    """
    Remove NaN and infinite values from PPG and/or ABP signals, ensuring alignment if both are provided.

    Parameters:
    - ppg_signal: The PPG signal array (NumPy array), optional.
    - abp_signal: The ABP signal array (NumPy array), optional.

    Returns:
    - cleaned_ppg: Cleaned PPG signal with invalid values removed (or None if not provided).
    - cleaned_abp: Cleaned ABP signal with corresponding indices removed (or None if not provided).
    """
    if ppg_signal is None and abp_signal is None:
        raise ValueError("At least one of 'ppg_signal' or 'abp_signal' must be provided.")

    if ppg_signal is not None:
        ppg_signal = np.array(ppg_signal)  # Ensure it's a NumPy array
    if abp_signal is not None:
        abp_signal = np.array(abp_signal)  # Ensure it's a NumPy array

    if ppg_signal is not None and abp_signal is not None:
        # Create a combined mask for valid values in both signals
        valid_mask = (
            np.isfinite(ppg_signal) &  # PPG signal has no NaN or Inf
            np.isfinite(abp_signal)    # ABP signal has no NaN or Inf
        )
        # Apply the mask to both signals
        cleaned_ppg = ppg_signal[valid_mask]
        cleaned_abp = abp_signal[valid_mask]
        return cleaned_ppg, cleaned_abp

    if ppg_signal is not None:
        # Clean only PPG signal
        valid_mask = np.isfinite(ppg_signal)  # PPG signal has no NaN or Inf
        cleaned_ppg = ppg_signal[valid_mask]
        return cleaned_ppg, None

    if abp_signal is not None:
        # Clean only ABP signal
        valid_mask = np.isfinite(abp_signal)  # ABP signal has no NaN or Inf
        cleaned_abp = abp_signal[valid_mask]
        return None, cleaned_abp

def baseline_wander_removal(signal, fs=None, method='linear'):
    """
    Remove baseline wander from the signal using detrending.

    Parameters:
    - signal: Input signal (list or NumPy array).
    - fs: Sampling frequency (optional, not used in detrending but for context).
    - method: Method for detrending ('linear' or 'constant').
              'linear' removes a linear trend, 'constant' removes the mean.

    Returns:
    - Processed signal with baseline wander removed.
    """
    # Detrend the signal to remove baseline wander
    detrended_signal = detrend(signal, type=method)
    return detrended_signal

def downsample_signal(signal, original_fs, target_fs):
    """
    Downsample the signal to a lower sampling rate.

    Parameters:
    - signal: Input signal (list or NumPy array).
    - original_fs: Original sampling frequency (Hz).
    - target_fs: Target sampling frequency (Hz).

    Returns:
    - Downsampled signal (NumPy array).
    """
    if target_fs >= original_fs:
        raise ValueError("Target sampling frequency must be less than the original frequency.")

    # Calculate the number of samples in the downsampled signal
    num_samples = int(len(signal) * target_fs / original_fs)

    # Resample the signal
    downsampled_signal = resample(signal, num_samples)

    return downsampled_signal

def remove_out_of_range_bp(ppg_signal, abp_signal, min_bp=40, max_bp=200):
    """
    Remove ABP values outside the range [min_bp, max_bp], and remove the corresponding PPG signal values.

    Parameters:
    - ppg_signal: The PPG signal array (NumPy array).
    - abp_signal: The ABP signal array (NumPy array).
    - min_bp: Minimum allowable blood pressure value (default: 40 mmHg).
    - max_bp: Maximum allowable blood pressure value (default: 200 mmHg).

    Returns:
    - cleaned_ppg: Cleaned PPG signal with out-of-range values removed.
    - cleaned_abp: Cleaned ABP signal with out-of-range values removed.
    """
    # Create a mask for ABP values within the valid range
    valid_mask = (abp_signal >= min_bp) & (abp_signal <= max_bp)

    # Apply the mask to both signals
    cleaned_ppg = ppg_signal[valid_mask]
    cleaned_abp = abp_signal[valid_mask]

    return cleaned_ppg, cleaned_abp