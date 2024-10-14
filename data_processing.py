import numpy as np
import logging

def process_chunk(chunk_data):
    try:
        adc1_signal = chunk_data['adc1'].values
        adc2_signal = chunk_data['adc2'].values

        adc1_peaks = detect_significant_peaks(adc1_signal)
        adc2_peaks = detect_significant_peaks(adc2_signal)

        adc1_peak_summary = ""
        adc2_peak_summary = ""

        if len(adc1_peaks) > 0:
            adc1_peak_values = adc1_signal[adc1_peaks]
            adc1_peak_summary = f"ADC1 significant peaks found: {len(adc1_peaks)}, Mean peak value: {np.mean(adc1_peak_values):.2f}"
            logging.debug(adc1_peak_summary)

        if len(adc2_peaks) > 0:
            adc2_peak_values = adc2_signal[adc2_peaks]
            adc2_peak_summary = f"ADC2 significant peaks found: {len(adc2_peaks)}, Mean peak value: {np.mean(adc2_peak_values):.2f}"
            logging.debug(adc2_peak_summary)

        return {
            'adc1': adc1_signal,
            'adc2': adc2_signal,
            'adc1_peaks': adc1_peaks,
            'adc2_peaks': adc2_peaks,
            'adc1_peak_summary': adc1_peak_summary,
            'adc2_peak_summary': adc2_peak_summary
        }
    except Exception as e:
        logging.error(f"Error in peak detection: {e}")
        return None

def detect_significant_peaks(signal_data, threshold=850, min_duration=50):
    """
    Detect peaks where the signal goes above the threshold for a long enough duration.

    :param signal_data: The ADC signal data.
    :param threshold: The threshold value to detect peaks (default is 850).
    :param min_duration: Minimum number of consecutive samples above the threshold to consider it a peak.
    :return: Array of peak indices.
    """
    above_threshold = signal_data > threshold  # Boolean array where signal is above threshold
    peaks = []
    start_idx = None  # To store the starting index of a potential peak

    for i, is_above in enumerate(above_threshold):
        if is_above:
            if start_idx is None:
                start_idx = i  # Start of a potential peak
        else:
            if start_idx is not None:
                # Check if the duration of the peak is long enough
                if i - start_idx >= min_duration:
                    # Peak is valid, store the middle point of the peak
                    peak_idx = (start_idx + i) // 2
                    peaks.append(peak_idx)
                start_idx = None  # Reset start index for the next potential peak

    # Handle case where the signal remains above the threshold until the end
    if start_idx is not None and len(signal_data) - start_idx >= min_duration:
        peak_idx = (start_idx + len(signal_data)) // 2
        peaks.append(peak_idx)

    return np.array(peaks)