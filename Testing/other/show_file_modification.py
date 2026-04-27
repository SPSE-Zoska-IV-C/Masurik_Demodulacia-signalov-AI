import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

TARGET_LEN = 9000


signal = np.fromfile("Test/00001.complex", dtype=np.complex64)


def time_stretch_zoh(signal, target_len):
    """
    Time-stretch a signal by repeating samples (zero-order hold)
    until target_len is reached.
    """
    N = len(signal)
    stretch_factor = target_len // N

    stretched = np.repeat(signal, stretch_factor)

    # Handle remainder if length is still short
    if len(stretched) < target_len:
        remainder = target_len - len(stretched)
        stretched = np.concatenate([stretched, signal[:remainder]])

    return stretched[:target_len]



N = len(signal)
if N >= TARGET_LEN:
    raise ValueError("Input file must have less than 9000 samples")

time_stretched_signal = time_stretch_zoh(signal, TARGET_LEN)

stretched_signal = resample(signal, TARGET_LEN)

pad_length = TARGET_LEN - N
padded_signal = np.pad(signal, (0, pad_length), mode="constant")

def plot_complex(signal, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(signal.real, label="Real")
    plt.plot(signal.imag, label="Imag", linestyle="--")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

# --------------------------------------------------
# CREATE & SAVE PLOTS
# --------------------------------------------------
plot_complex(
    signal,
    f"Original Complex Signal ({N} samples)",
    "original_complex_signal.png"
)

plot_complex(
    stretched_signal,
    "Stretched Complex Signal (9000 samples, averaged)",
    "stretched_complex_signal.png"
)

plot_complex(
    padded_signal,
    "Zero-Padded Complex Signal (9000 samples)",
    "padded_complex_signal.png"
)

plot_complex(
    time_stretched_signal,
    "Time-Stretched ASK Signal (Zero-Order Hold)",
    "time_stretched_complex_signal.png"
)


print("All plots generated and saved successfully.")
