from scipy.signal import savgol_filter, find_peaks
from tkinter import messagebox

def process_signal(signal_data, window_length, poly_order):
    try:
        window = int(window_length.get())
        if window % 2 == 0:
            window += 1
        polyorder = int(poly_order.get())
        return savgol_filter(signal_data, window, polyorder)
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid filter parameters: {str(e)}")
        return signal_data


def find_signal_peaks(signal_data, peak_height, peak_distance, time):
    try:
        height = float(peak_height.get())
        distance = int(peak_distance.get())
        peaks, _ = find_peaks(signal_data, height=height, distance=distance)

        # Print the peak values and their timestamps
        for peak in peaks:
            print(f"Peak detected: Value = {signal_data[peak]}, Timestamp = {time[peak]}")

        return peaks
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid peak detection parameters: {str(e)}")
        return []
