from scipy.signal import savgol_filter
from tkinter import messagebox
from peakAnalyzer import AdvancedPeakDetector


def process_signal(signal_data, window_length, poly_order):
    try:
        window = int(window_length.get())
        if window % 2 == 0:
            window += 1
        polyorder = int(poly_order.get())
        return savgol_filter(signal_data, window, polyorder)
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid filter parameters: {str(e)}")
        return signal_data


def find_signal_peaks(signal_data, params, time):
    try:
        # Create detector instance
        detector = AdvancedPeakDetector(sample_rate=50000, target_frequency=50)

        # Update detector parameters
        detector.min_prominence = params['prominence_threshold']
        detector.slope_factor = params['slope_factor']
        detector.timing_tolerance = params['timing_tolerance']
        detector.amplitude_tolerance = params['amplitude_tolerance']

        # Detect peaks
        peaks, properties = detector.detect_peaks(signal_data)

        # Print analysis results
        print("\nPeak Analysis Results:")
        print(f"Number of peaks detected: {properties.get('peak_count', 0)}")
        print(f"Detected frequency: {properties.get('actual_frequency', 0):.2f} Hz")
        print(f"Mean peak interval: {properties.get('mean_interval', 0):.2f} samples")
        print(f"Signal quality score: {properties.get('signal_quality', 0):.3f}")

        return peaks, properties.get('rejected_peaks', [])

    except Exception as e:
        messagebox.showerror("Error", f"Error in peak detection: {str(e)}")
        return [], []