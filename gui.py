import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from signal_processing import process_signal, find_signal_peaks
from file_operations import load_csv, convert_to_npy, load_npy

class SignalAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Analyzer with NPY Support")
        self.sample_rate = 50000  # 50kHz sampling rate
        self.setup_gui()
        self.data = None

    def setup_gui(self):
        # Control Panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5, padx=5, fill=tk.X)

        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations")
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        ttk.Button(file_frame, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Convert to NPY", command=self.convert_to_npy).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Load NPY", command=self.load_npy).pack(side=tk.LEFT, padx=5)

        # Filter settings
        filter_frame = ttk.LabelFrame(control_frame, text="Filter Settings")
        filter_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        ttk.Label(filter_frame, text="Window Length:").pack(side=tk.LEFT, padx=5)
        self.window_length = ttk.Entry(filter_frame, width=6)
        self.window_length.insert(0, "51")
        self.window_length.pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Polynomial Order:").pack(side=tk.LEFT, padx=5)
        self.poly_order = ttk.Entry(filter_frame, width=4)
        self.poly_order.insert(0, "3")
        self.poly_order.pack(side=tk.LEFT, padx=5)

        # Peak detection settings
        peak_frame = ttk.LabelFrame(control_frame, text="Peak Detection")
        peak_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        ttk.Label(peak_frame, text="Min Height:").pack(side=tk.LEFT, padx=5)
        self.peak_height = ttk.Entry(peak_frame, width=6)
        self.peak_height.insert(0, "800")
        self.peak_height.pack(side=tk.LEFT, padx=5)

        ttk.Label(peak_frame, text="Min Distance:").pack(side=tk.LEFT, padx=5)
        self.peak_distance = ttk.Entry(peak_frame, width=6)
        self.peak_distance.insert(0, "1000")
        self.peak_distance.pack(side=tk.LEFT, padx=5)

        # Update button
        ttk.Button(control_frame, text="Update Analysis", command=self.update_analysis).pack(side=tk.LEFT, padx=5)

        # Create the main figure and add toolbar
        self.setup_plots()

    def setup_plots(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_csv(self):
        self.data = load_csv()

    def convert_to_npy(self):
        if self.data is not None:
            convert_to_npy(self.data)

    def load_npy(self):
        self.data = load_npy()
        self.update_analysis()

    def update_analysis(self):
        if self.data is None:
            return

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        # Downsample the data
        downsample_rate = 5  # Change this value as needed
        downsampled_data = self.data.iloc[::downsample_rate]

        # Time vector (in seconds)
        time = np.arange(len(downsampled_data)) / self.sample_rate

        # Process signals
        filtered_adc1 = process_signal(downsampled_data['adc1'], self.window_length, self.poly_order)
        filtered_adc2 = process_signal(downsampled_data['adc2'], self.window_length, self.poly_order)

        # Find peaks and log the results
        peaks_adc1 = find_signal_peaks(filtered_adc1, self.peak_height, self.peak_distance, time)
        peaks_adc2 = find_signal_peaks(filtered_adc2, self.peak_height, self.peak_distance, time)

        # Plot ADC1
        self.ax1.plot(time, downsampled_data['adc1'], 'b-', alpha=0.3, label='Raw ADC1')
        self.ax1.plot(time, filtered_adc1, 'b-', label='Filtered ADC1')
        self.ax1.plot(time[peaks_adc1], filtered_adc1[peaks_adc1], 'rx', label=f'Peaks ({len(peaks_adc1)})')

        # Plot ADC2
        self.ax2.plot(time, downsampled_data['adc2'], 'g-', alpha=0.3, label='Raw ADC2')
        self.ax2.plot(time, filtered_adc2, 'g-', label='Filtered ADC2')
        self.ax2.plot(time[peaks_adc2], filtered_adc2[peaks_adc2], 'rx', label=f'Peaks ({len(peaks_adc2)})')

        # Set limits and labels
        self.ax1.set_title('ADC1 Signal Analysis')
        self.ax2.set_title('ADC2 Signal Analysis')

        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ADC Value')
            ax.grid(True)
            ax.legend()

        self.fig.tight_layout()
        self.canvas.draw_idle()  # Use draw_idle for efficiency