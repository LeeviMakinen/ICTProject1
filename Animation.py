# Animation.py

import numpy as np
from matplotlib.animation import FuncAnimation
from datetime import timedelta

class Animator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.ani = None

    def animate(self, frame):
        if self.analyzer.data_queue.empty():
            return self.analyzer.adc1_line, self.analyzer.adc2_line, self.analyzer.adc1_peaks, self.analyzer.adc2_peaks

        data = self.analyzer.data_queue.get()

        # Update buffers with new data
        self.analyzer.adc1_buffer.extend(data['adc1'][-self.analyzer.display_points:])
        self.analyzer.adc2_buffer.extend(data['adc2'][-self.analyzer.display_points:])

        # Update peak counts
        self.analyzer.adc1_peak_count += len(data['adc1_peaks'])
        self.analyzer.adc2_peak_count += len(data['adc2_peaks'])

        # Calculate time values for the entire display buffer
        self.analyzer.current_time = self.analyzer.processed_duration
        start_time = self.analyzer.current_time - self.analyzer.display_time
        time_values = np.linspace(start_time, self.analyzer.current_time, len(self.analyzer.adc1_buffer))

        # Update peaks and their corresponding times
        self.update_peaks(data, time_values)

        # Update statistics label
        self.update_statistics()

        # Update line plots for ADC1 and ADC2
        self.analyzer.adc1_line.set_data(time_values, list(self.analyzer.adc1_buffer))
        self.analyzer.adc2_line.set_data(time_values, list(self.analyzer.adc2_buffer))

        # Update peak markers
        self.analyzer.adc1_peaks.set_data(list(self.analyzer.adc1_peak_times), list(self.analyzer.adc1_peak_values))
        self.analyzer.adc2_peaks.set_data(list(self.analyzer.adc2_peak_times), list(self.analyzer.adc2_peak_values))

        # Adjust plot limits for the sliding window
        for ax in [self.analyzer.ax1, self.analyzer.ax2]:
            ax.set_xlim(max(0, start_time), self.analyzer.current_time)

        return self.analyzer.adc1_line, self.analyzer.adc2_line, self.analyzer.adc1_peaks, self.analyzer.adc2_peaks

    def update_peaks(self, data, time_values):
        # Append new peaks to the peak buffers and calculate their correct time values
        buffer_start_index = len(self.analyzer.adc1_buffer) - len(data['adc1'])

        for peak in data['adc1_peaks']:
            peak_time = time_values[buffer_start_index + peak]
            peak_value = self.analyzer.adc1_buffer[buffer_start_index + peak]
            self.analyzer.adc1_peak_times.append(peak_time)
            self.analyzer.adc1_peak_values.append(peak_value)
            print(f"ADC1 Peak at time {peak_time}: {peak_value}")  # Print statement for ADC1 peaks

        buffer_start_index = len(self.analyzer.adc2_buffer) - len(data['adc2'])

        for peak in data['adc2_peaks']:
            peak_time = time_values[buffer_start_index + peak]
            peak_value = self.analyzer.adc2_buffer[buffer_start_index + peak]
            self.analyzer.adc2_peak_times.append(peak_time)
            self.analyzer.adc2_peak_values.append(peak_value)
            print(f"ADC2 Peak at time {peak_time}: {peak_value}")  # Print statement for ADC2 peaks

    def update_statistics(self):
        stats_text = f"Processed: {timedelta(seconds=int(self.analyzer.processed_duration))} | "
        stats_text += f"ADC1 Peaks: {self.analyzer.adc1_peak_count} | ADC2 Peaks: {self.analyzer.adc2_peak_count}"
        self.analyzer.stats_label.config(text=stats_text)

    def start_animation(self):
        if self.ani is None:
            self.ani = FuncAnimation(
                self.analyzer.fig,
                self.animate,
                interval=self.analyzer.animation_speed,
                blit=True,
                cache_frame_data=False
            )
