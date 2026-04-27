import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


DATA_DIR = r"C:\Users\masur\Downloads\Masurik_Demodulacia_signalov_AI-main\Masurik_Demodulacia_signalov_AI-main\ask_dataset"
MODEL_PATH = r"C:\Users\masur\Downloads\Masurik_Demodulacia_signalov_AI-main\Masurik_Demodulacia_signalov_AI-main\denoiser_FINAL.pth"
FILE_ID = "00001"  
NUM_SAMPLES = 9000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn

class ComplexDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=9, padding=4)
        )
    def forward(self, x):
        return self.net(x)

def load_complex(path):
    data = np.fromfile(path, dtype=np.complex64)
    if len(data) != NUM_SAMPLES:
        raise ValueError(f"{path} has {len(data)} samples, expected {NUM_SAMPLES}")
    return data



def main():
    
    noisy_path = os.path.join(DATA_DIR, f"{FILE_ID}.complex")
    clean_path = os.path.join(DATA_DIR, f"{FILE_ID}_noiseless.complex")

    
    noisy = load_complex(noisy_path)
    clean = load_complex(clean_path)

    
    model = ComplexDenoiser().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    
    x = np.stack([noisy.real, noisy.imag], axis=0)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)

    
    with torch.no_grad():
        y = model(x)[0].cpu().numpy()

    denoised = y[0] + 1j * y[1]

    
    
    
    t = np.arange(NUM_SAMPLES)

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, noisy.real, label="Real")
    plt.plot(t, noisy.imag, label="Imag", alpha=0.7)
    plt.title("original")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, denoised.real, label="Real")
    plt.plot(t, denoised.imag, label="Imag", alpha=0.7)
    plt.title("denoised")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, clean.real, label="Real")
    plt.plot(t, clean.imag, label="Imag", alpha=0.7)
    plt.title("noiseless")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("denoising_plots.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
