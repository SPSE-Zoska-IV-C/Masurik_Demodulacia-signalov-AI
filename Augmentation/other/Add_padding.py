import os
import numpy as np


FOLDER = "./dataset2"  
TARGET_SAMPLES = 9000

for fname in os.listdir(FOLDER):
    if fname.endswith(".complex"):
        path = os.path.join(FOLDER, fname)
        data = np.fromfile(path, dtype=np.complex64)
        if len(data) < TARGET_SAMPLES:
            pad = np.zeros(TARGET_SAMPLES - len(data), dtype=np.complex64)
            data = np.concatenate([data, pad])
            data.tofile(path)
        elif len(data) > TARGET_SAMPLES:
            data[:TARGET_SAMPLES].tofile(path)
