import os.path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, HORIZONTAL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from signal_processing import process_signal, find_signal_peaks
from file_operations import load_csv, convert_to_npy, load_npy


class SignalAnalyzer:
    def __init__(self, root):
        self.title_label = None
        self.convert_button = None
        self.root = root
        self.root.title("Signal Processing App")
        self.sample_rate = 50000  # 50kHz sampling rate
        self.setup_gui()
        self.data = None
        self.filename = None

        self.data_type = None
        self.peaks_data = None

    def setup_gui(self):

        self.title_label = ttk.Label(
            self.root,
            text="Signal Processing App",
            font=("Helvetica", 16),
            foreground="black",
        )
        self.title_label.pack(pady=5)

        # Custom title card for more control over colors

        # Main container for all controls
        control_container = ttk.Frame(self.root)
        control_container.pack(pady=5, padx=5, fill=tk.X)

        # Left side - File operations
        file_frame = ttk.LabelFrame(control_container, text="File Operations")
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        ttk.Button(file_frame, text="Load CSV", command=self.load_csv).pack(
            side=tk.LEFT, padx=5
        )

        # Defining the conversion button for specific actions in a later function
        self.convert_button = ttk.Button(
            file_frame, text="Convert to NPY", command=self.start_convert_to_npy
        )
        self.convert_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(file_frame, text="Load NPY", command=self.load_npy).pack(
            side=tk.LEFT, padx=5
        )

        # Center - Filter settings
        filter_frame = ttk.LabelFrame(control_container, text="Filter Settings")
        filter_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        ttk.Label(filter_frame, text="Window Length:").pack(side=tk.LEFT, padx=5)
        self.window_length = tk.Scale(
            filter_frame, from_=3, to=101, resolution=2, orient=HORIZONTAL
        )
        self.window_length.set(31)  # Default value
        self.window_length.pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Polynomial Order:").pack(side=tk.LEFT, padx=5)
        self.poly_order = tk.Scale(
            filter_frame, from_=1, to=5, resolution=1, orient=HORIZONTAL
        )
        self.poly_order.set(2)  # Default value
        self.poly_order.pack(side=tk.LEFT, padx=5)

        # Right side - Peak Detection Controls
        peak_frame = ttk.LabelFrame(control_container, text="Peak Detection Parameters")
        peak_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        # Prominence control
        ttk.Label(peak_frame, text="Min Prominence (%):").pack(side=tk.LEFT, padx=5)
        self.prominence_threshold = tk.Scale(
            peak_frame, from_=0.01, to=100.0, resolution=0.01, orient=HORIZONTAL
        )
        self.prominence_threshold.set(45)
        self.prominence_threshold.pack(side=tk.LEFT, padx=5)

        ttk.Label(peak_frame, text="Amp Tolerance (σ):").pack(side=tk.LEFT, padx=5)
        self.amplitude_tolerance = tk.Scale(
            peak_frame, from_=1.0, to=10.0, resolution=0.1, orient=HORIZONTAL
        )
        self.amplitude_tolerance.set(4.0)
        self.amplitude_tolerance.pack(side=tk.LEFT, padx=5)

        threshold_frame = ttk.LabelFrame(
            control_container, text="Peak Classification Thresholds"
        )
        threshold_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)

        # High threshold control
        ttk.Label(threshold_frame, text="High Peak Threshold:\n(% of max)").pack(
            side=tk.LEFT, padx=5
        )
        self.high_threshold = tk.Scale(
            threshold_frame, from_=10, to=100, resolution=0.1, orient=HORIZONTAL
        )
        self.high_threshold.set(30)  # Default value
        self.high_threshold.pack(side=tk.LEFT, padx=5)

        # Medium threshold control
        ttk.Label(threshold_frame, text="Medium Peak Threshold:\n(% of max)").pack(
            side=tk.LEFT, padx=5
        )
        self.medium_threshold = tk.Scale(
            threshold_frame, from_=0.1, to=50, resolution=0.1, orient=HORIZONTAL
        )
        self.medium_threshold.set(9)  # Default value
        self.medium_threshold.pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(control_container)
        action_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        # Update button
        ttk.Button(
            action_frame, text="Update Analysis", command=self.update_analysis
        ).pack(side=tk.TOP, padx=5)

        reset_button = ttk.Button(
            action_frame, text="Reset Defaults", command=self.reset_to_defaults
        )
        reset_button.pack(side=tk.TOP, padx=5, pady=5)

        ttk.Button(file_frame, text="Export Peaks", command=self.export_peaks).pack(
            side=tk.LEFT, padx=5
        )
        # Button for exporting to csv file after analysis
        # Add Import Peaks button
        ttk.Button(file_frame, text="Import Peaks", command=self.import_peaks).pack(
            side=tk.LEFT, padx=5
        )

        # Create the main figure and add toolbar
        self.setup_plots()

    def setup_plots(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_to_defaults(self):

        # Set default values for each parameter
        default_values = {
            "window_length": 31,
            "poly_order": 2,
            "prominence_threshold": 0.01,
            "amplitude_tolerance": 4.0,
            "high_threshold": 30,
            "medium_threshold": 9,
        }

        # Reset sliders
        self.window_length.set(default_values["window_length"])
        self.poly_order.set(default_values["poly_order"])
        self.prominence_threshold.set(default_values["prominence_threshold"])
        self.amplitude_tolerance.set(default_values["amplitude_tolerance"])
        self.high_threshold.set(default_values["high_threshold"])
        self.medium_threshold.set(default_values["medium_threshold"])

    def load_csv(self):
        """Load CSV file and handle both ADC and peaks data formats"""
        result = load_csv()
        if result[0] is None:
            return

        data, filename, data_type = result
        self.filename = os.path.basename(filename)

        if data_type == "adc":
            self.data = data
            self.data_type = "adc"
            self.peaks_data = None  # Clear any existing peaks data
            # self.update_analysis()

        self.root.title(f"Signal Analyzer - {self.filename}")
        self.title_label.config(text=f"Signal Analyzer - {self.filename}")

    def import_peaks(self):
        """Specifically import peaks data from a CSV file"""
        result = load_csv()
        if result[0] is None:
            return

        data, filename, data_type = result

        if data_type == "peaks":
            self.peaks_data = data
            self.filename = os.path.basename(filename)

            # Update window title and label with peaks filename
            self.root.title(f"Signal Analyzer - Peaks: {self.filename}")
            self.title_label.config(
                text=f"Signal Analyzer - Peaks: {self.filename}"
            )

            # Plot the imported peaks
            self.plot_peaks_only()
        else:
            messagebox.showinfo("Info", "Please select a peaks CSV file.")

    def plot_peaks_only(self):
        if self.peaks_data is None:
            messagebox.showinfo("Info", "No peaks data loaded.")
            return

        # Clear previous plots
        self.ax1.cla()
        self.ax2.cla()

        # Define constants for peak classifications
        WATER_Y_VALUE = 0.5
        TISSUE_Y_VALUE = 1

        # Plot peaks based on their labels and connect them with lines
        for label, group in self.peaks_data.groupby("label"):
            # Sort peaks by their startTime to ensure proper line connection
            group = group.sort_values("startTime")

            times = group["startTime"].values  # Use startTime for x-axis

            # Assign y-values based on label type (0.5 for water peaks, 1 for tissue peaks)
            if label == "water":
                y_values = [WATER_Y_VALUE] * len(times)  # Water peaks at y = 0.5
                marker = "o"  # Marker for water peaks
                color = "blue"  # Color for water peaks (arbitrary choice)
            else:  # Tissue label
                y_values = [TISSUE_Y_VALUE] * len(times)  # Tissue peaks at y = 1
                marker = "^"  # Marker for tissue peaks
                color = "red"  # Color for tissue peaks (arbitrary choice)

            # Assign the same axis for all peaks, as ADC1/ADC2 is no longer relevant
            ax = self.ax1  # Use ax1 for both water and tissue peaks

            # Only add label for the first occurrence of each label in the legend
            legend_labels = [
                item.get_label() for item in ax.get_legend_handles_labels()[0]
            ]
            if label not in legend_labels:
                ax.scatter(times, y_values, label=label, color=color, marker=marker)
            else:
                ax.scatter(times, y_values, color=color, marker=marker)

            # Create a continuous x-axis by adding points between peaks with y=0
            continuous_times = []
            continuous_y = []

            for i in range(1, len(times)):
                continuous_times.append(times[i - 1])
                continuous_y.append(y_values[i - 1])  # Peak value for the current peak

                # Add a "gap" between peaks
                gap_start = times[i - 1]
                gap_end = times[i]
                continuous_times.extend([gap_start, gap_end])
                continuous_y.extend([0, 0])  # Y value for gap is 0

            continuous_times.append(times[-1])  # Add the last peak
            continuous_y.append(y_values[-1])

            # Plot the continuous line connecting peaks and gaps
            ax.plot(
                continuous_times,
                continuous_y,
                color=color,
                linestyle="-",
                linewidth=0.8,
                alpha=0.7,
            )

        # Update plot legends and refresh canvas
        self.ax1.legend(loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def start_convert_to_npy(self):
        # disable button

        self.convert_button.config(state=tk.DISABLED)
        self.convert_button.config(text="Converting...")
        self.root.update_idletasks()  # Forces UI update

        # Run conversion
        if self.data is not None:
            self.convert_to_npy()

        # Re-enable button
        self.convert_button.config(state=tk.NORMAL)
        self.convert_button.config(text="Convert to NPY")

    def convert_to_npy(self):
        if self.data is not None:
            convert_to_npy(self.data)

    def load_npy(self):
        self.data, self.filename = load_npy()
        self.filename = os.path.basename(self.filename)
        self.root.title(f"Signal Analyzer - {self.filename}")
        self.title_label.config(text=f"Signal Analyzer - {self.filename}")

        if self.data is None:
            return

        # Clear previous lines on the axes
        self.ax1.cla()
        self.ax2.cla()

        # Increase downsample rate if needed
        downsample_rate = 10
        downsampled_data = self.data.iloc[::downsample_rate]

        # Time vector (in seconds)
        time = np.arange(len(downsampled_data)) / (self.sample_rate / downsample_rate)

        # Plot raw ADC1 and ADC2 data
        self.ax1.plot(time, downsampled_data["adc1"], "b-", label="ADC1")
        self.ax2.plot(time, downsampled_data["adc2"], "g-", label="ADC2")

        # Refresh canvas
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def get_peak_params(self):
        """Get peak detection parameters from GUI inputs"""
        try:
            params = {
                "prominence_threshold": float(self.prominence_threshold.get()),
                "amplitude_tolerance": float(self.amplitude_tolerance.get()),
                "high_threshold": float(self.high_threshold.get()) / 100,
                "medium_threshold": float(self.medium_threshold.get()) / 100,
            }
            return params
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
            return None

    def update_analysis(self):
        if self.data is None:
            return

        try:
            window = int(self.window_length.get())
            if window % 2 == 0:  # Ensure window length is odd
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
            time = np.arange(len(downsampled_data)) / (
                self.sample_rate / downsample_rate
            )

            # Process signals
            filtered_adc1 = process_signal(downsampled_data["adc1"], window, poly_order)
            filtered_adc2 = process_signal(downsampled_data["adc2"], window, poly_order)

            # Find peaks with parameters
            peaks_adc1, properties_adc1 = find_signal_peaks(filtered_adc1, peak_params)
            peaks_adc2, properties_adc2 = find_signal_peaks(filtered_adc2, peak_params)

            # Plot ADC1
            self.ax1.plot(
                time, downsampled_data["adc1"], "b-", alpha=0.3, label="Raw ADC1"
            )
            self.ax1.plot(time, filtered_adc1, "b-", label="Filtered ADC1")

            # Plot classified peaks for ADC1
            if len(peaks_adc1) > 0:
                classifications = properties_adc1["peak_classifications"]
                if classifications:  # Check if classifications exist
                    high_peaks = [
                        p for p, c in zip(peaks_adc1, classifications) if c == "high"
                    ]
                    medium_peaks = [
                        p for p, c in zip(peaks_adc1, classifications) if c == "medium"
                    ]
                    low_peaks = [
                        p for p, c in zip(peaks_adc1, classifications) if c == "low"
                    ]

                    if high_peaks:
                        self.ax1.plot(
                            time[high_peaks],
                            filtered_adc1[high_peaks],
                            "x",
                            color="red",
                            label=f"High Peaks ({len(high_peaks)})",
                            markersize=10,
                        )
                    if medium_peaks:
                        self.ax1.plot(
                            time[medium_peaks],
                            filtered_adc1[medium_peaks],
                            "x",
                            color="yellow",
                            label=f"Medium Peaks ({len(medium_peaks)})",
                            markersize=8,
                        )
                    if low_peaks:
                        self.ax1.plot(
                            time[low_peaks],
                            filtered_adc1[low_peaks],
                            "x",
                            color="orange",
                            label=f"Low Peaks ({len(low_peaks)})",
                            markersize=6,
                        )

            # Plot rejected peaks for ADC1
            rejected_peaks_adc1 = properties_adc1["rejected_peaks"]
            if rejected_peaks_adc1 is not None and len(rejected_peaks_adc1) > 0:
                self.ax1.plot(
                    time[rejected_peaks_adc1],
                    filtered_adc1[rejected_peaks_adc1],
                    "x",
                    color="blue",
                    label=f"Rejected ({len(rejected_peaks_adc1)})",
                    markersize=6,
                )

            # Plot ADC2
            self.ax2.plot(
                time, downsampled_data["adc2"], "g-", alpha=0.3, label="Raw ADC2"
            )
            self.ax2.plot(time, filtered_adc2, "g-", label="Filtered ADC2")

            # Plot classified peaks for ADC2
            if len(peaks_adc2) > 0:
                classifications = properties_adc2["peak_classifications"]
                if classifications:  # Check if classifications exist
                    high_peaks = [
                        p for p, c in zip(peaks_adc2, classifications) if c == "high"
                    ]
                    medium_peaks = [
                        p for p, c in zip(peaks_adc2, classifications) if c == "medium"
                    ]
                    low_peaks = [
                        p for p, c in zip(peaks_adc2, classifications) if c == "low"
                    ]

                    if high_peaks:
                        self.ax2.plot(
                            time[high_peaks],
                            filtered_adc2[high_peaks],
                            "x",
                            color="red",
                            label=f"High Peaks ({len(high_peaks)})",
                            markersize=10,
                        )
                    if medium_peaks:
                        self.ax2.plot(
                            time[medium_peaks],
                            filtered_adc2[medium_peaks],
                            "x",
                            color="yellow",
                            label=f"Medium Peaks ({len(medium_peaks)})",
                            markersize=8,
                        )
                    if low_peaks:
                        self.ax2.plot(
                            time[low_peaks],
                            filtered_adc2[low_peaks],
                            "x",
                            color="orange",
                            label=f"Low Peaks ({len(low_peaks)})",
                            markersize=6,
                        )

            # Plot rejected peaks for ADC2
            rejected_peaks_adc2 = properties_adc2["rejected_peaks"]
            if rejected_peaks_adc2 is not None and len(rejected_peaks_adc2) > 0:
                self.ax2.plot(
                    time[rejected_peaks_adc2],
                    filtered_adc2[rejected_peaks_adc2],
                    "x",
                    color="blue",
                    label=f"Rejected ({len(rejected_peaks_adc2)})",
                    markersize=6,
                )

            # Set fixed legend location
            for ax in [self.ax1, self.ax2]:
                ax.legend(loc="upper right")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.grid(True)

            # Refresh canvas
            self.fig.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Error", f"Error updating analysis: {str(e)}")
            raise  # This will help with debugging by showing the full error traceback

    def export_peaks(self):
        """Export plotted data with the assumption that all low peaks are water, and anything above is tissue"""
        if self.data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            # Get target file location
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Export Peaks Data",
            )

            if not save_path:
                return

            # Get current params
            window = int(self.window_length.get())
            if window % 2 == 0:
                window += 1
            poly_order = int(self.poly_order.get())
            peak_params = self.get_peak_params()

            if peak_params is None:
                return

            # Keep downsample rate
            downsample_rate = 10
            downsampled_data = self.data.iloc[::downsample_rate]
            time = np.arange(len(downsampled_data)) / (
                self.sample_rate / downsample_rate
            )

            filtered_adc1 = process_signal(downsampled_data["adc1"], window, poly_order)
            filtered_adc2 = process_signal(downsampled_data["adc2"], window, poly_order)

            peaks_adc1, properties_adc1 = find_signal_peaks(filtered_adc1, peak_params)
            peaks_adc2, properties_adc2 = find_signal_peaks(filtered_adc2, peak_params)

            # Create list for peak data
            peaks_data = []

            def get_label(peak_class):  # Helper function to classify found peaks
                # If peak is low (below medium threshold), label as water
                # If peak is medium or high, label as tissue
                return "water" if peak_class == "low" else "tissue"

            # ADC1 peaks
            if len(peaks_adc1) > 0:
                for peak_idx, peak_class in zip(
                    peaks_adc1, properties_adc1["peak_classifications"]
                ):
                    peak_time = round(time[peak_idx], 5)  # Round to 5 decimal places
                    peaks_data.append(
                        {
                            "startTime": peak_time,
                            "endTime": peak_time,
                            "label": get_label(peak_class),
                        }
                    )

            # ADC2 peaks
            if len(peaks_adc2) > 0:
                for peak_idx, peak_class in zip(
                    peaks_adc2, properties_adc2["peak_classifications"]
                ):
                    peak_time = round(time[peak_idx], 5)  # Round to 5 decimal places
                    peaks_data.append(
                        {
                            "startTime": peak_time,
                            "endTime": peak_time,
                            "label": get_label(peak_class),
                        }
                    )

            # Sort by start time
            import pandas as pd

            peaks_df = pd.DataFrame(peaks_data)
            peaks_df = peaks_df.sort_values("startTime")

            # Doublecheck that timestamps are rounded correctly
            peaks_df.to_csv(save_path, index=False, float_format="%.5f")

            messagebox.showinfo(
                "Success", f"Peaks data exported successfully to {save_path}"
            )

        except Exception as e:  # Whoopsie daisies moment
            messagebox.showerror("Error", f"Error exporting peaks: {str(e)}")
