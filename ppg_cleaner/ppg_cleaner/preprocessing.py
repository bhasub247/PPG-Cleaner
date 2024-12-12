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

def remove_invalid_values(signal, replace_with='interpolation', secondary_signal=None):
    """
    Remove NaN and Inf values from the signal and optionally replace them.
    If a secondary signal is provided, corresponding invalid indices are removed.

    Parameters:
    - signal (list or np.array): The input signal containing numeric values.
    - replace_with (str): Strategy to handle invalid values:
        - 'interpolation': Replace using linear interpolation.
        - 'zero': Replace with zero.
        - 'mean': Replace with the mean of the valid values.
        - 'remove': Remove the invalid values entirely.
    - secondary_signal (list or np.array, optional): An additional signal where the same
      invalid indices will be handled (removal only).

    Returns:
    - tuple:
        - np.array: Cleaned primary signal with invalid values handled.
        - np.array or None: Correspondingly cleaned secondary signal (if provided).
    """
    signal = np.array(signal)  # Ensure the input is a NumPy array
    if secondary_signal is not None:
        secondary_signal = np.array(secondary_signal)

    # Identify invalid values (NaN or Inf)
    invalid_mask = np.isnan(signal) | np.isinf(signal)

    if not invalid_mask.any():
        return signal, secondary_signal  # Return as-is if no invalid values

    if replace_with == 'interpolation':
        if secondary_signal is not None:
            raise ValueError("Interpolation is not supported for secondary signals. Use 'remove'.")
        valid_indices = ~invalid_mask
        signal[invalid_mask] = np.interp(
            np.flatnonzero(invalid_mask),
            np.flatnonzero(valid_indices),
            signal[valid_indices]
        )
    elif replace_with == 'zero':
        signal[invalid_mask] = 0
        if secondary_signal is not None:
            secondary_signal[invalid_mask] = 0
    elif replace_with == 'mean':
        signal[invalid_mask] = np.nanmean(signal[~np.isinf(signal)])
        if secondary_signal is not None:
            secondary_signal[invalid_mask] = np.nanmean(signal[~np.isinf(signal)])
    elif replace_with == 'remove':
        signal = signal[~invalid_mask]
        if secondary_signal is not None:
            secondary_signal = secondary_signal[~invalid_mask]
    else:
        raise ValueError("Invalid `replace_with` option. Use 'interpolation', 'zero', 'mean', or 'remove'.")

    return signal, secondary_signal

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