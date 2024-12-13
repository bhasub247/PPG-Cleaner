from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch

def bandpass_filter(signal, fs, low_cutoff=0.5, high_cutoff=8.0, order=4):
    """
    Apply a Butterworth bandpass filter to retain PPG-relevant frequencies.
    
    Parameters:
    - signal: Input signal (NumPy array).
    - fs: Sampling frequency in Hz.
    - low_cutoff: Lower cutoff frequency in Hz.
    - high_cutoff: Upper cutoff frequency in Hz.
    - order: Filter order (default: 4).
    
    Returns:
    - Filtered signal (NumPy array).
    """
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)



def notch_filter(signal, fs, notch_freq, quality_factor=30):
    """
    Remove powerline noise (e.g., 50/60 Hz) using a notch filter.
    
    Parameters:
    - signal: Input signal (NumPy array).
    - fs: Sampling frequency in Hz.
    - notch_freq: Frequency to notch out (e.g., 50 or 60 Hz).
    - quality_factor: Quality factor of the notch filter (default: 30).
    
    Returns:
    - Filtered signal (NumPy array).
    """
    nyquist = 0.5 * fs
    normalized_freq = notch_freq / nyquist

    b, a = iirnotch(normalized_freq, quality_factor)
    return filtfilt(b, a, signal)


def lowpass_filter(signal, fs, cutoff, order=4):
    """
    Apply a Butterworth lowpass filter to suppress high-frequency noise.
    
    Parameters:
    - signal: Input signal (NumPy array).
    - fs: Sampling frequency in Hz.
    - cutoff: Cutoff frequency in Hz.
    - order: Filter order (default: 4).
    
    Returns:
    - Filtered signal (NumPy array).
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, signal)

def highpass_filter(signal, fs, cutoff, order=4):
    """
    Apply a Butterworth highpass filter to remove low-frequency components.
    
    Parameters:
    - signal: Input signal (NumPy array).
    - fs: Sampling frequency in Hz.
    - cutoff: Cutoff frequency in Hz.
    - order: Filter order (default: 4).
    
    Returns:
    - Filtered signal (NumPy array).
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    b, a = butter(order, normalized_cutoff, btype='high')
    return filtfilt(b, a, signal)

