import pandas as pd
import numpy as np
from tkinter import filedialog, messagebox
import logging


def load_csv():
    """
    Load a CSV file, detecting whether it's ADC data or peaks data based on columns.

    Returns:
        tuple: (data, filename, file_type)
        - data: pandas DataFrame containing the loaded data
        - filename: name of the loaded file
        - file_type: 'adc' or 'peaks' indicating the type of data loaded
    """
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    filename = filepath.title()
    if not filepath:
        return None, None, None

    try:
        data = pd.read_csv(filepath)

        # Check for peaks data format
        peaks_columns = {"startTime", "endTime", "label"}
        if peaks_columns.issubset(set(data.columns)):
            # Validate peaks data
            if (
                data[peaks_columns].isna().any().any()
            ):              # This special sauce checks if data contains empty fields (NaN)
                raise ValueError("Peaks data contains missing values")
            return data, filename, "peaks"

        # Check for ADC data format
        adc_columns = {"adc1", "adc2"}
        if adc_columns.issubset(set(data.columns)):
            # Validate ADC data
            if not all(data[col].notna().all() for col in adc_columns):
                raise ValueError("ADC data contains missing values")
            if not all(
                data[col].dtype.kind in "if" for col in adc_columns
            ):  # check if numeric
                raise ValueError("ADC columns must contain numeric data")
            return data, filename, "adc"

        raise ValueError(
            "CSV must contain either 'adc1' and 'adc2' columns OR 'startTime', 'endTime', and 'label' columns"
        )

    except Exception as e:
        logging.error(f"Error loading file: {e}")
        messagebox.showerror("Error", str(e))
        return None, None, None


def convert_to_npy(data):
    save_path = filedialog.asksaveasfilename(
        defaultextension=".npy", filetypes=[("NumPy files", "*.npy")]
    )
    if not save_path:
        return

    try:
        # Convert to float32 first for any calculations, then to int16 for saving
        np_array = np.column_stack(
            (
                data["adc1"].astype("float32").values,
                data["adc2"].astype("float32").values,
            )
        )
        # Convert to int16 before saving
        np_array = np_array.astype("int16")
        np.save(save_path, np_array)
        messagebox.showinfo("Success", "File saved successfully")

    except Exception as e:
        logging.error(f"Error saving NPY file: {e}")
        messagebox.showerror("Error", str(e))


def load_npy():
    filepath = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
    filename = filepath.title()
    if not filepath:
        return None, None

    try:
        # Load as int16 first
        np_array = np.load(filepath)
        # Convert to float32 for processing
        np_array = pd.DataFrame(np_array, columns=["adc1", "adc2"]).astype(
            {"adc1": "float32", "adc2": "float32"}
        )
        return np_array, filename

    except Exception as e:
        logging.error(f"Error loading NPY file: {e}")
        messagebox.showerror("Error", str(e))
        return None, None
