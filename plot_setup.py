import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

def setup_plot(self):
    self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # ADC1 plot
    self.adc1_line, = self.ax1.plot([], [], 'b-', label='ADC1 Signal')
    self.adc1_peaks, = self.ax1.plot([], [], 'rx', label='ADC1 Peaks',
                                     markersize=10, markeredgewidth=2)
    self.ax1.set_title('ADC1 Signal')
    self.ax1.set_ylim(390, 1520)
    self.ax1.legend()



    # ADC2 plot
    self.adc2_line, = self.ax2.plot([], [], 'g-', label='ADC2 Signal')
    self.adc2_peaks, = self.ax2.plot([], [], 'rx', label='ADC2 Peaks',
                                     markersize=10, markeredgewidth=2)
    self.ax2.set_title('ADC2 Signal')
    self.ax2.set_ylim(390,1520)
    self.ax2.legend()

    for ax in [self.ax1, self.ax2]:
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ADC Value')


    self.fig.tight_layout() #Clipping legend was annoying me

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    self.fig.canvas.mpl_connect('scroll_event', self.zoom)