import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from signal_processing import process_signal, find_signal_peaks
from file_operations import load_csv, convert_to_npy, load_npy

class SignalAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Signal Analyzer")
        self.sample_rate = 50000  # 50kHz sampling rate
        self.setup_gui()
        self.data = None

    def setup_gui(self):
        # Main container for all controls
        control_container = ttk.Frame(self.root)
        control_container.pack(pady=5, padx=5, fill=tk.X)

        # Left side - File operations
        file_frame = ttk.LabelFrame(control_container, text="File Operations")
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        ttk.Button(file_frame, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Convert to NPY", command=self.convert_to_npy).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Load NPY", command=self.load_npy).pack(side=tk.LEFT, padx=5)

        # Center - Filter settings
        filter_frame = ttk.LabelFrame(control_container, text="Filter Settings")
        filter_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        ttk.Label(filter_frame, text="Window Length:").pack(side=tk.LEFT, padx=5)
        self.window_length = ttk.Entry(filter_frame, width=6)
        self.window_length.insert(0, "31")
        self.window_length.pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Polynomial Order:").pack(side=tk.LEFT, padx=5)
        self.poly_order = ttk.Entry(filter_frame, width=4)
        self.poly_order.insert(0, "2")
        self.poly_order.pack(side=tk.LEFT, padx=5)

        # Right side - Peak Detection Controls
        peak_frame = ttk.LabelFrame(control_container, text="Peak Detection Parameters")
        peak_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        # Prominence control
        ttk.Label(peak_frame, text="Min Prominence (%):").pack(side=tk.LEFT, padx=5)
        self.prominence_threshold = ttk.Entry(peak_frame, width=6)
        self.prominence_threshold.insert(0, "0.1")
        self.prominence_threshold.pack(side=tk.LEFT, padx=5)

        # Amplitude tolerance control
        ttk.Label(peak_frame, text="Amp Tolerance (σ):").pack(side=tk.LEFT, padx=5)
        self.amplitude_tolerance = ttk.Entry(peak_frame, width=6)
        self.amplitude_tolerance.insert(0, "4.0")
        self.amplitude_tolerance.pack(side=tk.LEFT, padx=5)

        # Update button
        ttk.Button(control_container, text="Update Analysis", command=self.update_analysis).pack(side=tk.LEFT, padx=5)

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
        if self.data is not None:
            self.update_analysis()

    def get_peak_params(self):
        """Get peak detection parameters from GUI inputs"""
        try:
            params = {
                'prominence_threshold': float(self.prominence_threshold.get()),
                'amplitude_tolerance': float(self.amplitude_tolerance.get())
            }
            return params
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
            return None

    def update_analysis(self):
        if self.data is None:
            return

        try:
            # Get filter parameters
            window = int(self.window_length.get())
            if window % 2 == 0:
                window += 1
            poly_order = int(self.poly_order.get())

            # Clear previous lines on the axes
            self.ax1.cla()
            self.ax2.cla()

            # Get peak detection parameters
            peak_params = self.get_peak_params()
            if peak_params is None:
                return

            # Increase downsample rate if needed
            downsample_rate = 10
            downsampled_data = self.data.iloc[::downsample_rate]

            # Time vector (in seconds)
            time = np.arange(len(downsampled_data)) / (self.sample_rate / downsample_rate)

            # Process signals
            filtered_adc1 = process_signal(downsampled_data['adc1'], window, poly_order)
            filtered_adc2 = process_signal(downsampled_data['adc2'], window, poly_order)

            # Find peaks with parameters
            peaks_adc1, rejected_peaks_adc1 = find_signal_peaks(filtered_adc1, peak_params)
            peaks_adc2, rejected_peaks_adc2 = find_signal_peaks(filtered_adc2, peak_params)

            # Plot ADC1
            self.ax1.plot(time, downsampled_data['adc1'], 'b-', alpha=0.3, label='Raw ADC1')
            self.ax1.plot(time, filtered_adc1, 'b-', label='Filtered ADC1')
            if len(peaks_adc1) > 0:
                self.ax1.plot(time[peaks_adc1], filtered_adc1[peaks_adc1], 'rx',
                              label=f'Valid Peaks ({len(peaks_adc1)})')
            if rejected_peaks_adc1 is not None and len(rejected_peaks_adc1) > 0:
                self.ax1.plot(time[rejected_peaks_adc1], filtered_adc1[rejected_peaks_adc1], 'bx',
                              label=f'Rejected ({len(rejected_peaks_adc1)})')

            # Plot ADC2
            self.ax2.plot(time, downsampled_data['adc2'], 'g-', alpha=0.3, label='Raw ADC2')
            self.ax2.plot(time, filtered_adc2, 'g-', label='Filtered ADC2')
            if len(peaks_adc2) > 0:
                self.ax2.plot(time[peaks_adc2], filtered_adc2[peaks_adc2], 'rx',
                              label=f'Valid Peaks ({len(peaks_adc2)})')
            if rejected_peaks_adc2 is not None and len(rejected_peaks_adc2) > 0:
                self.ax2.plot(time[rejected_peaks_adc2], filtered_adc2[rejected_peaks_adc2], 'bx',
                              label=f'Rejected ({len(rejected_peaks_adc2)})')

            # Set fixed legend location
            for ax in [self.ax1, self.ax2]:
                ax.legend(loc="upper right")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True)

            # Refresh canvas
            self.fig.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Error", f"Error updating analysis: {str(e)}")