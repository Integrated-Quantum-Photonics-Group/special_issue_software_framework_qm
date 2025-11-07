"""
 * Module Name: json_coding
 * Description: make it possible to handle with
                arrays in main.py
 * Author: Yannick Strocka
 * Created On: October 30, 2025
 * Last Modified: October 31, 2025
 * Version: 1.0
"""

import numpy as np

# ---------------------------------------------------
# Robust encoder/decoder that preserves shapes
# ---------------------------------------------------
def _complex_to_pair(z):
    # ensure python floats
    return [float(np.real(z)), float(np.imag(z))]

def encode_array(a: np.ndarray):
    """Encode a numpy array as a dict with shape and data (real/imag pairs)."""
    arr = np.asarray(a)  # ensure ndarray
    shape = arr.shape
    # produce nested list of [real, imag] pairs
    data = []
    for row in arr.tolist():
        # row may be a scalar if arr is 1-D; make it iterable
        if not isinstance(row, list):
            row = [row]
        data.append([_complex_to_pair(x) for x in row])
    return {"__ndarray__": True, "shape": list(shape), "data": data}

def encode_for_json(obj):
    """Recursively encode python objects: arrays -> dicts, complex -> [re,im]."""
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, complex):
        return _complex_to_pair(obj)
    if isinstance(obj, dict):
        return {k: encode_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [encode_for_json(x) for x in obj]
    # primitive types: int/float/str/bool/None
    return obj

def decode_array(d):
    """Decode dict created by encode_array back to a numpy complex array."""
    shape = tuple(d["shape"])
    data = d["data"]
    # convert nested [re,im] pairs back to complex numbers
    rows = []
    for row in data:
        rows.append([complex(x[0], x[1]) for x in row])
    arr = np.array(rows, dtype=complex)
    # If shape was 1-D (e.g., (2,), (1,2),(2,1)), reshape to exact shape stored
    if arr.size != np.prod(shape):
        raise ValueError("decoded data size mismatch vs stored shape")
    arr = arr.reshape(shape)
    return arr

def decode_from_json(obj):
    """Recursively decode encoded JSON objects back to python (numpy) objects."""
    if isinstance(obj, dict):
        # detect ndarray encoding
        if obj.get("__ndarray__") is True:
            return decode_array(obj)
        else:
            return {k: decode_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        # could be a complex pair [re,im] or list of things
        # Heuristic: if list length==2 and both elements are numbers -> treat as complex pair
        if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            return complex(obj[0], obj[1])
        return [decode_from_json(x) for x in obj]
    return obj
