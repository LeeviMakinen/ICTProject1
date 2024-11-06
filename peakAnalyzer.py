import numpy as np
from scipy.signal import savgol_filter, hilbert, peak_prominences, find_peaks
from scipy.stats import zscore


class AdvancedPeakDetector:
    def __init__(self, sample_rate=50000, target_frequency=2):
        self.sample_rate = sample_rate
        self.target_frequency = target_frequency
        self.expected_period = int(sample_rate / target_frequency)

    def detect_peaks(
        self,
        signal,
        min_prominence_pct=0.1,
        amplitude_tolerance=4.0,
        high_threshold=0.3,
        medium_threshold=0.09,
    ):
        """
        Enhanced peak detection algorithm with peak classification based on amplitude thresholds.
        """
        normalized = self._prepare_signal(signal)
        signal_median = np.median(normalized)
        signal_iqr = np.percentile(normalized, 75) - np.percentile(normalized, 25)
        noise_floor = signal_median + signal_iqr * 0.5

        peaks = self._find_initial_peaks(normalized, min_prominence_pct, noise_floor)
        peaks, rejected_peaks = self._filter_peaks(
            normalized, peaks, amplitude_tolerance
        )

        # Classify peaks by amplitude
        peak_classifications = self._classify_peaks(
            normalized, peaks, high_threshold, medium_threshold
        )

        properties = self._calculate_properties(signal, peaks)
        properties["rejected_peaks"] = rejected_peaks
        properties["peak_classifications"] = peak_classifications

        return peaks, properties

    def _classify_peaks(self, signal, peaks, high_threshold, medium_threshold):
        """Classify peaks based on amplitude thresholds"""
        if len(peaks) == 0:
            return []

        peak_amplitudes = signal[peaks]
        max_amplitude = np.max(peak_amplitudes)

        # Normalize amplitudes relative to maximum
        normalized_amplitudes = peak_amplitudes / max_amplitude

        classifications = []
        for amp in normalized_amplitudes:
            if amp >= high_threshold:
                classifications.append("high")
            elif amp >= medium_threshold:
                classifications.append("medium")
            else:
                classifications.append("low")

        return classifications

    def _prepare_signal(self, signal):
        """Prepare signal with improved baseline correction"""
        # Calculate noise level for adaptive window size
        noise_level = np.std(np.diff(signal))

        # Adjust window size based on noise level
        base_window = 31
        window_length = min(base_window + int(noise_level * 10), len(signal) // 10)
        if window_length % 2 == 0:
            window_length += 1

        # Apply Savitzky-Golay filter
        smoothed = savgol_filter(signal, window_length, 2)

        # Enhanced baseline correction
        baseline = np.percentile(smoothed, 20)  # Use 20th percentile as baseline
        return smoothed - baseline

    def _find_initial_peaks(self, signal, min_prominence_pct, noise_floor):
        """Find initial peaks using dynamic thresholding"""
        # Calculate signal range excluding outliers
        signal_range = np.percentile(signal, 99) - np.percentile(signal, 1)

        # Dynamic prominence threshold
        min_prominence = max(
            signal_range * (min_prominence_pct / 100),
            noise_floor * 2,  # Ensure minimum separation from noise
        )

        # Find peaks with adaptive parameters
        peaks, _ = find_peaks(
            signal, prominence=min_prominence, distance=int(self.expected_period * 0.5)
        )

        return peaks

    def _filter_peaks(self, signal, peaks, amplitude_tolerance):
        """Enhanced peak filtering with better handling of amplitude variations"""
        if len(peaks) < 2:
            return peaks, []

        # Calculate peak amplitudes
        amplitudes = signal[peaks]

        # Use rolling statistics for local amplitude variations
        window_size = min(20, len(amplitudes))
        rolling_median = np.array(
            [
                np.median(
                    amplitudes[
                        max(0, i - window_size) : min(len(amplitudes), i + window_size)
                    ]
                )
                for i in range(len(amplitudes))
            ]
        )
        rolling_std = np.array(
            [
                np.std(
                    amplitudes[
                        max(0, i - window_size) : min(len(amplitudes), i + window_size)
                    ]
                )
                for i in range(len(amplitudes))
            ]
        )

        # Calculate adaptive thresholds
        lower_bound = rolling_median - rolling_std * amplitude_tolerance
        upper_bound = rolling_median + rolling_std * amplitude_tolerance

        # Identify valid peaks
        valid_mask = (amplitudes >= lower_bound) & (amplitudes <= upper_bound)

        # Additional check for very large peaks
        max_amplitude = np.percentile(amplitudes, 99)
        large_peaks_mask = amplitudes > max_amplitude
        valid_mask = valid_mask | large_peaks_mask  # Include large peaks as valid

        valid_peaks = peaks[valid_mask]
        rejected_peaks = peaks[~valid_mask]

        return valid_peaks, rejected_peaks

    def _calculate_properties(self, signal, peaks):
        """Calculate properties of detected peaks"""
        if len(peaks) < 2:
            return {}

        intervals = np.diff(peaks)
        actual_frequency = self.sample_rate / np.mean(intervals)

        return {
            "peak_count": len(peaks),
            "mean_interval": np.mean(intervals),
            "std_interval": np.std(intervals),
            "actual_frequency": actual_frequency,
            "signal_quality": 1.0 - (np.std(intervals) / np.mean(intervals)),
        }
