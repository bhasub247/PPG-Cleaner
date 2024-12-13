import numpy as np
from sklearn.decomposition import FastICA


def hampel_filter(signal, window_size, threshold=3):
    """
    Detect and remove outliers using the Hampel method.
    
    Parameters:
    - signal: Input signal (NumPy array).
    - window_size: Half the window size for calculating the median.
    - threshold: Threshold for identifying outliers (default: 3).
    
    Returns:
    - Filtered signal (NumPy array).
    """
    signal_filtered = signal.copy()
    n = len(signal)
    
    for i in range(window_size, n - window_size):
        window = signal[i - window_size:i + window_size + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))  # Median absolute deviation
        
        if mad == 0:  # Avoid division by zero
            continue
        
        # Check if the point is an outlier
        if abs(signal[i] - median) / mad > threshold:
            signal_filtered[i] = median
    
    return signal_filtered


def artifact_detection(signal, fs, threshold=0.1):
    """
    Identify segments with sudden spikes or drops using thresholds or variance.
    
    Parameters:
    - signal: Input signal (NumPy array).
    - fs: Sampling frequency in Hz.
    - threshold: Threshold for detecting sudden changes (default: 0.1).
    
    Returns:
    - artifact_indices: Indices of detected artifacts.
    """
    # Calculate the first derivative to detect changes
    diff_signal = np.diff(signal)
    artifact_indices = np.where(np.abs(diff_signal) > threshold)[0]
    
    return artifact_indices


def motion_artifact_removal(signal, fs, n_components=2):
    """
    Remove artifacts caused by motion using Independent Component Analysis (ICA).
    
    Parameters:
    - signal: Input signal (NumPy array).
    - fs: Sampling frequency in Hz.
    - n_components: Number of components for ICA (default: 2).
    
    Returns:
    - Cleaned signal (NumPy array).
    """
    # Ensure signal is 2D for ICA
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    # Perform ICA
    ica = FastICA(n_components=n_components, random_state=42)
    decomposed_signals = ica.fit_transform(signal)
    
    # Reconstruct the cleaned signal (remove unwanted components)
    reconstructed_signal = ica.inverse_transform(decomposed_signals)
    
    # Return the first component as the cleaned signal
    return reconstructed_signal[:, 0]