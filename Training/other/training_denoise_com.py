import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATA_DIR = "GeneratedPairs"
NUM_SAMPLES = 9000
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-3
NUM_WORKERS = 4
MODEL_SAVE_PATH = "complex_denoiser_ver3.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ComplexWaveformDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted(
            f for f in os.listdir(root_dir)
            if f.endswith(".complex") and "_noiseless" not in f
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy_name = self.files[idx]
        clean_name = noisy_name.replace(".complex", "_noiseless.complex")

        noisy = np.fromfile(
            os.path.join(self.root_dir, noisy_name),
            dtype=np.complex64
        )
        clean = np.fromfile(
            os.path.join(self.root_dir, clean_name),
            dtype=np.complex64
        )

        if len(noisy) != NUM_SAMPLES:
            raise ValueError("Unexpected sample length")

        noisy = np.stack([noisy.real, noisy.imag], axis=0)
        clean = np.stack([clean.real, clean.imag], axis=0)

        return (
            torch.from_numpy(noisy).float(),
            torch.from_numpy(clean).float()
        )


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


def complex_mse(pred, target):
    pr, pi = pred[:, 0], pred[:, 1]
    tr, ti = target[:, 0], target[:, 1]
    return torch.mean((pr - tr) ** 2 + (pi - ti) ** 2)


def main():
    dataset = ComplexWaveformDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = ComplexDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for noisy, clean in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = complex_mse(output, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6e}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
