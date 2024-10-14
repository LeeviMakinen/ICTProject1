import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from collections import deque
import threading
from queue import Queue
import logging
from datetime import timedelta


class DualADCSignalAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual ADC Signal Analyzer")

        # Initialize parameters
        self.chunk_size = 50000
        self.sample_rate = 50000
        self.display_time = 10  # Display 10 seconds of data
        self.display_points = self.display_time * self.sample_rate
        self.filepath = None
        self.animation_speed = 100

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
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)

        ttk.Button(control_frame, text="Load Data File", command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Start Analysis", command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Analysis", command=self.stop_analysis).pack(side=tk.LEFT, padx=5)

        threshold_frame = ttk.Frame(self.root)
        threshold_frame.pack(pady=5)
        ttk.Label(threshold_frame, text="Peak Sensitivity:").pack(side=tk.LEFT)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=1.0, to=3.0,
                                          orient=tk.HORIZONTAL, length=200)
        self.threshold_slider.set(2.0)
        self.threshold_slider.pack(side=tk.LEFT)

        stats_frame = ttk.Frame(self.root)
        stats_frame.pack(pady=5)
        self.stats_label = ttk.Label(stats_frame, text="Processed: 0s | ADC1 Peaks: 0 | ADC2 Peaks: 0")
        self.stats_label.pack()

        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack(pady=5)

    def setup_plot(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # ADC1 plot
        self.adc1_line, = self.ax1.plot([], [], 'b-', label='ADC1 Signal')
        self.adc1_peaks, = self.ax1.plot([], [], 'rx', label='ADC1 Peaks',
                                         markersize=10, markeredgewidth=2)
        self.ax1.set_title('ADC1 Signal')
        self.ax1.set_ylim(400, 1500)
        self.ax1.legend()

        # ADC2 plot
        self.adc2_line, = self.ax2.plot([], [], 'g-', label='ADC2 Signal')
        self.adc2_peaks, = self.ax2.plot([], [], 'rx', label='ADC2 Peaks',
                                         markersize=10, markeredgewidth=2)
        self.ax2.set_title('ADC2 Signal')
        self.ax2.set_ylim(400, 1500)
        self.ax2.legend()

        for ax in [self.ax1, self.ax2]:
            ax.grid(True)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ADC Value')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

    def process_chunk(self, chunk_data):
        try:
            adc1_signal = chunk_data['adc1'].values
            adc2_signal = chunk_data['adc2'].values

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

            adc1_peaks = detect_significant_peaks(adc1_signal)
            adc2_peaks = detect_significant_peaks(adc2_signal)

            if len(adc1_peaks) > 0:
                adc1_peak_values = adc1_signal[adc1_peaks]
                logging.debug(f"ADC1 significant peaks found: {len(adc1_peaks)}, "
                              f"Mean peak value: {np.mean(adc1_peak_values):.2f}")

            if len(adc2_peaks) > 0:
                adc2_peak_values = adc2_signal[adc2_peaks]
                logging.debug(f"ADC2 significant peaks found: {len(adc2_peaks)}, "
                              f"Mean peak value: {np.mean(adc2_peak_values):.2f}")

            return {
                'adc1': adc1_signal,
                'adc2': adc2_signal,
                'adc1_peaks': adc1_peaks,
                'adc2_peaks': adc2_peaks
            }
        except Exception as e:
            logging.error(f"Error in peak detection: {e}")
            return None

    def process_data_thread(self):
        try:
            total_chunks = sum(1 for _ in pd.read_csv(self.filepath, chunksize=self.chunk_size))
            processed_chunks = 0

            for chunk in pd.read_csv(self.filepath, chunksize=self.chunk_size):
                if self.stop_event.is_set():
                    break

                result = self.process_chunk(chunk)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    root = tk.Tk()
    app = DualADCSignalAnalyzer(root)
    root.mainloop()