# Import specific functions from preprocessing.py via the __init__.py setup
from ppg_cleaner import zscore_normalization, min_max_normalization, rescale_signal

# Test cases for the functions
def test_zscore_normalization():
    signal = [1, 2, 3, 4, 5]
    normalized_signal = zscore_normalization(signal)
    print("Z-Score Normalized Signal:", normalized_signal)

def test_min_max_normalization():
    signal = [1, 2, 3, 4, 5]
    min_val, max_val = 0, 1
    normalized_signal = min_max_normalization(signal, min_val, max_val)
    print("Min-Max Normalized Signal:", normalized_signal)

def test_rescale_signal():
    signal = [-2, -1, 0, 1, 2]
    rescaled_signal = rescale_signal(signal)
    print("Rescaled Signal:", rescaled_signal)

# Run the test cases
if __name__ == "__main__":
    test_zscore_normalization()
    test_min_max_normalization()
    test_rescale_signal()