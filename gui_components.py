import tkinter as tk
from tkinter import ttk

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

    self.filename_label = tk.Label(self.root, text="No file loaded", anchor='w')
    self.filename_label.pack(side=tk.TOP, fill=tk.Y)  # Adjust position as needed


    stats_frame = ttk.Frame(self.root)
    stats_frame.pack(pady=5)
    self.stats_label = ttk.Label(stats_frame, text="Processed: 0s | ADC1 Peaks: 0 | ADC2 Peaks: 0")
    self.stats_label.pack()

    self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
    self.progress.pack(pady=5)