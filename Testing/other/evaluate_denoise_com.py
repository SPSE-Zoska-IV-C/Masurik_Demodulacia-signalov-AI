import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
DATA_DIR = "GeneratedPairs"
MODEL_PATH = "complex_denoiser_ver3.pth"
FILE_ID = "00006"     # <-- change this
NUM_SAMPLES = 9000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# COMPLEX MODEL (MUST MATCH TRAINING)
# =====================
class ComplexConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        self.rr = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.ri = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.ir = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.ii = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, xr, xi):
        real = self.rr(xr) - self.ii(xi)
        imag = self.ri(xr) + self.ir(xi)
        return real, imag


class ComplexReLU(nn.Module):
    def forward(self, xr, xi):
        return torch.relu(xr), torch.relu(xi)


class ComplexDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = ComplexConv1d(1, 32, 9, padding=4)
        self.a1 = ComplexReLU()

        self.c2 = ComplexConv1d(32, 64, 9, padding=4)
        self.a2 = ComplexReLU()

        self.c3 = ComplexConv1d(64, 64, 9, padding=4)
        self.a3 = ComplexReLU()

        self.c4 = ComplexConv1d(64, 32, 9, padding=4)
        self.a4 = ComplexReLU()

        self.c5 = ComplexConv1d(32, 1, 9, padding=4)

    def forward(self, x):
        xr, xi = x[:, 0:1], x[:, 1:2]

        xr, xi = self.c1(xr, xi)
        xr, xi = self.a1(xr, xi)

        xr, xi = self.c2(xr, xi)
        xr, xi = self.a2(xr, xi)

        xr, xi = self.c3(xr, xi)
        xr, xi = self.a3(xr, xi)

        xr, xi = self.c4(xr, xi)
        xr, xi = self.a4(xr, xi)

        xr, xi = self.c5(xr, xi)

        return torch.cat([xr, xi], dim=1)


# =====================
# LOAD COMPLEX FILE
# =====================
def load_complex(path):
    data = np.fromfile(path, dtype=np.complex64)
    if len(data) != NUM_SAMPLES:
        raise ValueError(f"{path} has wrong length")
    return data


# =====================
# MAIN
# =====================
def main():
    noisy_path = os.path.join(DATA_DIR, f"{FILE_ID}.complex")
    clean_path = os.path.join(DATA_DIR, f"{FILE_ID}_noiseless.complex")

    noisy = load_complex(noisy_path)
    clean = load_complex(clean_path)

    # Load model
    model = ComplexDenoiser().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Prepare input
    x = np.stack([noisy.real, noisy.imag], axis=0)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)

    # Denoise
    with torch.no_grad():
        y = model(x)[0].cpu().numpy()

    denoised = y[0] + 1j * y[1]

    # =====================
    # PLOTTING
    # =====================
    t = np.arange(NUM_SAMPLES)

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, noisy.real, label="Real")
    plt.plot(t, noisy.imag, label="Imag", alpha=0.7)
    plt.title("Noisy Signal")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, denoised.real, label="Real")
    plt.plot(t, denoised.imag, label="Imag", alpha=0.7)
    plt.title("Complex CNN Denoised Signal")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, clean.real, label="Real")
    plt.plot(t, clean.imag, label="Imag", alpha=0.7)
    plt.title("Ground Truth (Noiseless)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
