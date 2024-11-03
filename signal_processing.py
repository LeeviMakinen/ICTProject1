from scipy.signal import savgol_filter
from tkinter import messagebox
from peakAnalyzer import AdvancedPeakDetector


def process_signal(signal_data, window_length, poly_order):
    try:
        return savgol_filter(signal_data, window_length, poly_order)
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid filter parameters: {str(e)}")
        return signal_data

def find_signal_peaks(signal_data, params):
    try:
        # Create detector instance
        detector = AdvancedPeakDetector(sample_rate=50000, target_frequency=50)

        # Detect peaks with the provided parameters
        peaks, properties = detector.detect_peaks(
            signal_data,
            min_prominence_pct=params['prominence_threshold'],
            amplitude_tolerance=params['amplitude_tolerance'],
            high_threshold=params['high_threshold'],
            medium_threshold=params['medium_threshold']
        )

        # Print analysis results
        print("\nPeak Analysis Results:")
        print(f"Number of peaks detected: {properties['peak_count']}")
        print(f"Detected frequency: {properties['actual_frequency']:.2f} Hz")
        print(f"Mean peak interval: {properties['mean_interval']:.2f} samples")
        print(f"Signal quality score: {properties['signal_quality']:.3f}")

        return peaks, properties

    except Exception as e:
        messagebox.showerror("Error", f"Error in peak detection: {str(e)}")
        return [], {'rejected_peaks': [], 'peak_classifications': []}