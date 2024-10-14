import tkinter as tk
from dual_adc_analyzer import DualADCSignalAnalyzer

if __name__ == "__main__":
    root = tk.Tk()
    app = DualADCSignalAnalyzer(root)
    root.mainloop()