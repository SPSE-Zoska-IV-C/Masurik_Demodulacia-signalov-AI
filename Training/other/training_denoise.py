import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


class ComplexWaveformDataset(Dataset):
    def __init__(self, root_dir, num_samples=9000):
        self.root_dir = root_dir
        self.num_samples = num_samples

        self.noisy_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".complex") and "_noiseless" not in f
        ])

    def __len__(self):
        return len(self.noisy_files)

    def _load_complex(self, path):
        data = np.fromfile(path, dtype=np.complex64)
        assert len(data) == self.num_samples
        return data

    def __getitem__(self, idx):
        noisy_name = self.noisy_files[idx]
        clean_name = noisy_name.replace(".complex", "_noiseless.complex")

        noisy = self._load_complex(os.path.join(self.root_dir, noisy_name))
        clean = self._load_complex(os.path.join(self.root_dir, clean_name))

        noisy = np.stack([noisy.real, noisy.imag], axis=0)
        clean = np.stack([clean.real, clean.imag], axis=0)

        return (
            torch.from_numpy(noisy).float(),
            torch.from_numpy(clean).float()
        )



class ComplexDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(32, 2, kernel_size=9, padding=4)
        )

    def forward(self, x):
        return self.net(x)
    


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ComplexWaveformDataset(r"/home/soram/Masurik_Demodulacia_signalov_AI/ask_dataset")
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = ComplexDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True
        )

        for noisy, clean in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1} finished | Avg loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "complex_test.pth")


if __name__ == "__main__":
    main()
