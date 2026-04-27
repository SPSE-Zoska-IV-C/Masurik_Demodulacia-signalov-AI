import os
import numpy as np
from scipy.signal import resample

def process_file(path, target_samples, method="pad"):
    data = np.fromfile(path, dtype=np.complex64)
    current_len = len(data)

    if method == "pad":
        if current_len < target_samples:
            pad = np.zeros(target_samples - current_len, dtype=np.complex64)
            data = np.concatenate([data, pad])
        elif current_len > target_samples:
            data = data[:target_samples]
    
    elif method == "resample":
        if current_len != target_samples:
            
            data = resample(data, target_samples).astype(np.complex64)

    
    data.tofile(path)

def apply_augmentation_bulk(folder, target_samples, method, progress_callback=None):
    if not os.path.exists(folder):
        return
    
    files = [f for f in os.listdir(folder) if f.endswith(".complex")]
    total = len(files)

    for i, fname in enumerate(files):
        if progress_callback:
            progress_callback(i, total)
        
        path = os.path.join(folder, fname)
        process_file(path, target_samples, method)
        
    if progress_callback:
        progress_callback(total, total)
