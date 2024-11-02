import numpy as np
from scipy.signal import savgol_filter, hilbert, peak_prominences
from scipy.interpolate import interp1d


# to play around with this values to make peak detection actually work
class AdvancedPeakDetector:
    def __init__(self, sample_rate=50000, target_frequency=100,
                 min_prominence=0.01, slope_factor=0.01,
                 timing_tolerance=0.1, amplitude_tolerance=0.1):
        self.sample_rate = sample_rate
        self.target_frequency = target_frequency
        self.expected_period = int(sample_rate / target_frequency)

        # Adjustable parameters
        self.min_prominence = min_prominence  # percentage of signal range
        self.slope_factor = slope_factor  # minimum slope threshold factor
        self.timing_tolerance = timing_tolerance  # percentage tolerance in timing
        self.amplitude_tolerance = amplitude_tolerance  # standard deviation for amplitude tolerance

    def detect_peaks(self, signal):
        """
        Detect peaks using multiple methods combined with improved noise rejection.

        Args:
            signal: numpy array of the signal

        Returns:
            peaks: array of peak indices
            properties: dict containing peak properties
        """
        # Step 1: Prepare and normalize the signal
        normalized_signal = self._normalize_signal(signal)

        # Step 2: Get envelope and calculate threshold
        envelope = self._get_envelope(normalized_signal)
        env_mean = np.mean(envelope)
        env_std = np.std(envelope)

        # Step 3: Get derivative information with more aggressive smoothing
        derivative = self._get_derivative(normalized_signal)

        # Step 4: Find potential peaks using zero crossings
        zero_crossings = self._find_zero_crossings(derivative)

        # Step 5: Calculate signal dynamics
        signal_range = np.ptp(normalized_signal)
        min_prominence = signal_range * self.min_prominence  # Use defined min prominence
        min_slope = np.std(derivative) * self.slope_factor  # Use defined slope factor

        # Step 6: Find peaks with strong prominence and slope
        peaks = self._find_prominent_peaks(normalized_signal, zero_crossings,
                                           min_prominence, min_slope, derivative)

        # Step 7: Validate peaks using timing and amplitude consistency
        final_peaks = self._validate_peaks_timing(peaks, normalized_signal)

        # Step 8: Calculate properties and convert back to original signal indices
        properties = self._calculate_properties(signal, final_peaks)

        return final_peaks, properties

    def _normalize_signal(self, signal):
        """Normalize the signal and apply initial smoothing."""
        # Apply more aggressive smoothing to reduce noise
        smoothed = savgol_filter(signal,
                                 window_length=min(101, len(signal) // 10),
                                 polyorder=3)
        return (smoothed - np.mean(smoothed)) / np.std(smoothed)

    def _get_envelope(self, signal):
        """Get signal envelope using Hilbert transform with improved smoothing."""
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        # More aggressive envelope smoothing
        envelope = savgol_filter(envelope,
                                 window_length=min(201, len(signal) // 5),
                                 polyorder=3)
        return envelope

    def _get_derivative(self, signal):
        """Calculate smoothed derivative with noise reduction."""
        # First derivative
        derivative = savgol_filter(signal,
                                   window_length=min(51, len(signal) // 20),
                                   polyorder=3,
                                   deriv=1)

        # Smooth the derivative
        derivative = savgol_filter(derivative,
                                   window_length=min(31, len(signal) // 30),
                                   polyorder=2)
        return derivative

    def _find_zero_crossings(self, signal):
        """Find zero crossings with negative slope."""
        zero_crossings = []
        for i in range(1, len(signal)):
            if signal[i - 1] > 0 and signal[i] <= 0:
                zero_crossings.append(i - 1)
        return np.array(zero_crossings)

    def _find_prominent_peaks(self, signal, zero_crossings, min_prominence, min_slope, derivative):
        """Find peaks with significant prominence and slope."""
        peaks = []

        for idx in zero_crossings:
            if idx >= len(signal) - 1:
                continue

            # Check local window
            window = self.expected_period // 4
            start = max(0, idx - window)
            end = min(len(signal), idx + window + 1)

            # Calculate prominence
            if idx < len(signal):
                prominences = peak_prominences(signal, [idx], wlen=window * 2)[0]
                if len(prominences) > 0:
                    prominence = prominences[0]
                else:
                    continue

                # Check if it's a local maximum
                if signal[idx] == max(signal[start:end]):
                    # Check prominence
                    if prominence >= min_prominence:
                        # Check slope
                        left_slope = max(abs(derivative[start:idx]))
                        right_slope = max(abs(derivative[idx:end]))
                        if left_slope >= min_slope and right_slope >= min_slope:
                            peaks.append(idx)

        return np.array(peaks)

    def _validate_peaks_timing(self, peaks, signal):
        """Validate peaks using timing and amplitude consistency."""
        if len(peaks) < 2:
            return peaks

        valid_peaks = [peaks[0]]
        expected_period = self.expected_period
        tolerance = expected_period * self.timing_tolerance  # Use defined timing tolerance

        # Calculate amplitude statistics for validation
        peak_amplitudes = signal[peaks]
        amp_mean = np.mean(peak_amplitudes)
        amp_std = np.std(peak_amplitudes)

        for i in range(1, len(peaks)):
            current_interval = peaks[i] - valid_peaks[-1]

            # Check timing
            if abs(current_interval - expected_period) <= tolerance:
                # Check amplitude consistency
                if abs(signal[peaks[i]] - amp_mean) <= self.amplitude_tolerance * amp_std:
                    valid_peaks.append(peaks[i])
            else:
                # Check if it's a multiple of the expected period
                num_periods = round(current_interval / expected_period)
                if num_periods >= 1 and abs(
                        current_interval - num_periods * expected_period) <= tolerance * num_periods:
                    valid_peaks.append(peaks[i])

        return np.array(valid_peaks)

    def _calculate_properties(self, signal, peaks):
        """Calculate properties of the detected peaks."""
        if len(peaks) < 2:
            return {}

        intervals = np.diff(peaks)
        actual_frequency = self.sample_rate / np.mean(intervals)

        # Calculate peak-to-peak amplitudes
        peak_amplitudes = signal[peaks]
        p2p_amplitudes = np.diff(peak_amplitudes)

        return {
            'peak_count': len(peaks),
            'mean_interval': np.mean(intervals),
            'std_interval': np.std(intervals),
            'actual_frequency': actual_frequency,
            'peak_amplitudes': peak_amplitudes,
            'mean_amplitude': np.mean(peak_amplitudes),
            'std_amplitude': np.std(peak_amplitudes),
            'mean_p2p_amplitude': np.mean(np.abs(p2p_amplitudes)),
            'signal_quality': 1.0 - (np.std(intervals) / np.mean(intervals))
        }
