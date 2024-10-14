import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from queue import Queue
import logging
from datetime import timedelta
import pandas as pd
import numpy as np
from collections import deque
from matplotlib.animation import FuncAnimation  # Add this import

from gui_components import create_widgets
from plot_setup import setup_plot
from data_processing import process_chunk
from utils import CHUNK_SIZE, SAMPLE_RATE, DISPLAY_TIME, ANIMATION_SPEED

class DualADCSignalAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual ADC Signal Analyzer")

        # Initialize parameters
        self.chunk_size = CHUNK_SIZE
        self.sample_rate = SAMPLE_RATE
        self.display_time = DISPLAY_TIME
        self.display_points = self.display_time * self.sample_rate
        self.filepath = None
        self.animation_speed = ANIMATION_SPEED

        # Data buffers for both ADC channels
        self.adc1_buffer = deque(maxlen=self.display_points)
        self.adc2_buffer = deque(maxlen=self.display_points)

        # Peak detection parameters
        self.peak_distance = int(0.5 * self.sample_rate)
        self.recent_adc1_peaks = []
        self.recent_adc2_peaks = []

        # Threading control
        self.stop_event = threading.Event()
        self.data_queue = Queue()

        # Setup GUI and plots
        self.create_widgets()
        self.setup_plot()

        # Statistics
        self.adc1_peak_count = 0
        self.adc2_peak_count = 0
        self.processed_duration = 0
        self.current_time = 0

    def create_widgets(self):
        create_widgets(self)

    def setup_plot(self):
        setup_plot(self)

    def load_file(self):
        try:
            self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if self.filepath:
                preview_df = pd.read_csv(self.filepath, nrows=5)
                if 'adc1' not in preview_df.columns or 'adc2' not in preview_df.columns:
                    raise ValueError("CSV file must contain 'adc1' and 'adc2' columns")
                messagebox.showinfo("File Loaded", f"Loaded: {self.filepath}")
        except Exception as e:
            logging.error(f"Error loading file: {e}")
            messagebox.showerror("Error", str(e))

    def process_data_thread(self):
        try:
            total_chunks = sum(1 for _ in pd.read_csv(self.filepath, chunksize=self.chunk_size))
            processed_chunks = 0

            for chunk in pd.read_csv(self.filepath, chunksize=self.chunk_size):
                if self.stop_event.is_set():
                    break

                result = process_chunk(chunk)
                if result is not None:
                    self.data_queue.put(result)

                processed_chunks += 1
                self.update_progress(processed_chunks / total_chunks * 100)

        except Exception as e:
            logging.error(f"Error in data processing thread: {e}")
            messagebox.showerror("Processing Error", str(e))

    def update_progress(self, value):
        self.progress['value'] = value
        self.processed_duration += self.chunk_size / self.sample_rate
        self.root.update_idletasks()

    def animate(self, frame):
        if self.data_queue.empty():
            return self.adc1_line, self.adc2_line, self.adc1_peaks, self.adc2_peaks

        data = self.data_queue.get()

        # Update buffers
        self.adc1_buffer.extend(data['adc1'])
        self.adc2_buffer.extend(data['adc2'])

        # Update peak counts
        self.adc1_peak_count += len(data['adc1_peaks'])
        self.adc2_peak_count += len(data['adc2_peaks'])

        # Write peak information to console
        self.write_peak_info(data['adc1_peaks'], data['adc2_peaks'],
                             data['adc1_peak_summary'], data['adc2_peak_summary'])

        # Update statistics label
        stats_text = f"Processed: {timedelta(seconds=int(self.processed_duration))} | "
        stats_text += f"ADC1 Peaks: {self.adc1_peak_count} | ADC2 Peaks: {self.adc2_peak_count}"
        self.stats_label.config(text=stats_text)

        # Calculate time values for display buffer
        self.current_time = self.processed_duration
        start_time = max(0, self.current_time - self.display_time)
        time_values = np.linspace(start_time, self.current_time, len(self.adc1_buffer))

        # Update plots
        self.adc1_line.set_data(time_values, list(self.adc1_buffer))
        self.adc2_line.set_data(time_values, list(self.adc2_buffer))

        # Update peak markers
        adc1_peak_x = []
        adc1_peak_y = []
        adc2_peak_x = []
        adc2_peak_y = []

        if len(data['adc1_peaks']) > 0:
            buffer_start_index = max(0, len(self.adc1_buffer) - len(data['adc1']))
            for peak in data['adc1_peaks']:
                adjusted_peak = buffer_start_index + peak
                if adjusted_peak < len(self.adc1_buffer):
                    adc1_peak_x.append(time_values[adjusted_peak])
                    adc1_peak_y.append(self.adc1_buffer[adjusted_peak])

        if len(data['adc2_peaks']) > 0:
            buffer_start_index = max(0, len(self.adc2_buffer) - len(data['adc2']))
            for peak in data['adc2_peaks']:
                adjusted_peak = buffer_start_index + peak
                if adjusted_peak < len(self.adc2_buffer):
                    adc2_peak_x.append(time_values[adjusted_peak])
                    adc2_peak_y.append(self.adc2_buffer[adjusted_peak])

        self.adc1_peaks.set_data(adc1_peak_x, adc1_peak_y)
        self.adc2_peaks.set_data(adc2_peak_x, adc2_peak_y)

        # Adjust plot limits for 10-second window
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(start_time, self.current_time)

        return self.adc1_line, self.adc2_line, self.adc1_peaks, self.adc2_peaks

    def write_peak_info(self, adc1_peaks, adc2_peaks, adc1_summary, adc2_summary):
        current_time = self.processed_duration

        # Print peak summaries
        if adc1_summary:
            print(f"Time: {current_time:.2f}s - {adc1_summary}")
        if adc2_summary:
            print(f"Time: {current_time:.2f}s - {adc2_summary}")

        # Print individual peak information
        if len(adc1_peaks) > 0:
            for peak in adc1_peaks:
                peak_time = current_time - (len(self.adc1_buffer) - peak) / self.sample_rate
                peak_value = self.adc1_buffer[peak]
                print(f"ADC1 Peak at {peak_time:.2f}s: {peak_value:.2f}")

        if len(adc2_peaks) > 0:
            for peak in adc2_peaks:
                peak_time = current_time - (len(self.adc2_buffer) - peak) / self.sample_rate
                peak_value = self.adc2_buffer[peak]
                print(f"ADC2 Peak at {peak_time:.2f}s: {peak_value:.2f}")

        print("-" * 50)  # Print a separator line for better readability

    def start_analysis(self):
        if not self.filepath:
            messagebox.showerror("Error", "Please load a file first!")
            return

        try:
            self.stop_event.clear()
            self.adc1_peak_count = 0
            self.adc2_peak_count = 0
            self.processed_duration = 0
            self.current_time = 0
            self.recent_adc1_peaks = []
            self.recent_adc2_peaks = []

            # Clear existing data
            self.adc1_buffer.clear()
            self.adc2_buffer.clear()

            # Start processing thread
            processing_thread = threading.Thread(target=self.process_data_thread)
            processing_thread.start()

            # Start animation
            self.ani = FuncAnimation(
                self.fig,
                self.animate,
                interval=self.animation_speed,
                blit=True,
                cache_frame_data=False
            )

        except Exception as e:
            logging.error(f"Error starting analysis: {e}")
            messagebox.showerror("Error", str(e))

    def stop_analysis(self):
        self.stop_event.set()
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()