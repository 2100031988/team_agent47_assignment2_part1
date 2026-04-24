import numpy as np

def load_csv(
    filepath,
    delimiter=",",
    dtype=float,
    missing_values=None,
    fill_value=np.nan,
    skip_header=0
):
    
    if missing_values is None:
        missing_values = ["", "NA", "null", "None"]

    try:
        data = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=fill_value,
            skip_header=skip_header,
            autostrip=True
        )
    except OSError:
        raise FileNotFoundError(f"File not found: {filepath}")

    if data is None or data.size == 0:
        raise ValueError("CSV file is empty or invalid")

    # Ensure consistent 2D shape
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data
