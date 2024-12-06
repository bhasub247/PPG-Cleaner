import numpy as np

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