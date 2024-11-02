import pandas as pd
import numpy as np
from tkinter import filedialog, messagebox
import logging

def load_csv():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filepath:
        return None

    try:
        data = pd.read_csv(filepath)
        if 'adc1' not in data.columns or 'adc2' not in data.columns:
            raise ValueError("CSV must contain 'adc1' and 'adc2' columns")
        return data
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        messagebox.showerror("Error", str(e))
        return None

def convert_to_npy(data):
    save_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
    if not save_path:
        return

    try:
        np_array = np.column_stack((data['adc1'].astype('float16').values, data['adc2'].astype('float16').values))
        np.save(save_path, np_array)
        messagebox.showinfo("Success", "File saved successfully")
    except Exception as e:
        logging.error(f"Error saving NPY file: {e}")
        messagebox.showerror("Error", str(e))

def load_npy():
    filepath = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
    if not filepath:
        return None

    try:
        np_array = np.load(filepath)
        return pd.DataFrame(np_array, columns=['adc1', 'adc2']).astype({'adc1': 'float16', 'adc2': 'float16'})
    except Exception as e:
        logging.error(f"Error loading NPY file: {e}")
        messagebox.showerror("Error", str(e))
        return None
