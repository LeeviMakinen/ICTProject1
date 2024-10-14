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
from Animation import Animator


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


        self.adc1_peak_times = deque()
        self.adc1_peak_values = deque()
        self.adc2_peak_times = deque()
        self.adc2_peak_values = deque()

        self.animator = Animator(self)

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

                # Update the label with the filename
                self.filename_label.config(text=f"Loaded: {self.filepath.split('/')[-1]}",fg="darkgreen",font=("Helvetica", 10, "bold"))  # Display only the filename

                self.flash_label()
        except Exception as e:
            logging.error(f"Error loading file: {e}")
            messagebox.showerror("Error", str(e))

    def flash_label(self, count=6):  # Flash 6 times
               if count > 0:
                # Toggle label color
                current_color = self.filename_label.cget("fg")
                new_color = "white" if current_color == "darkgreen" else "darkgreen"
                self.filename_label.config(fg=new_color)

                # Call this method again after 200ms
                self.root.after(200, self.flash_label, count - 1)


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



    def write_peak_info(self, adc1_peaks, adc2_peaks, adc1_summary, adc2_summary):
        current_time = self.processed_duration

        # Print peak summaries
        #if adc1_summary:
         #   print(f"Time: {current_time:.2f}s - {adc1_summary}")
        #if adc2_summary:
         #   print(f"Time: {current_time:.2f}s - {adc2_summary}")

        # Print individual peak information
        if len(adc1_peaks) > 0:
            for peak in adc1_peaks:
                peak_time = current_time - (len(self.adc1_buffer) - peak) / self.sample_rate
                peak_value = self.adc1_buffer[peak]
                print(f"ADC1 Peak at {peak_time:.2f}s: {peak_value:.2f}")
                print("-" * 50)  # Print a separator line for better readability
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
            self.animator.start_animation()

        except Exception as e:
            logging.error(f"Error starting analysis: {e}")
            messagebox.showerror("Error", str(e))

    def stop_analysis(self):
        self.stop_event.set()
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()


    def zoom(self, event):

        zoom_factor = 0.9  # How much to zoom in/out with each scroll step
        ax = event.inaxes

        if ax is None:
            return  # Ignore if we're not inside any axis

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        # Zoom in/out on x-axis
        if event.button == 'up':  # Scroll up to zoom in
            scale_factor = zoom_factor
        elif event.button == 'down':  # Scroll down to zoom out
            scale_factor = 1 / zoom_factor
        else:
            return  # Only handle scroll wheel (button = up/down)

        # Get the current x and y position of the cursor in data coordinates
        xdata = event.xdata
        ydata = event.ydata

        # Calculate new limits
        new_xlim = [(xdata - (xdata - cur_xlim[0]) * scale_factor),
                    (xdata + (cur_xlim[1] - xdata) * scale_factor)]
        new_ylim = [(ydata - (ydata - cur_ylim[0]) * scale_factor),
                    (ydata + (cur_ylim[1] - ydata) * scale_factor)]

        # Set new limits
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        # Redraw the canvas to reflect the changes

        self.fig.canvas.draw_idle()
