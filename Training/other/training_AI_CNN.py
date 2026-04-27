import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import matplotlib.pyplot as plt

class ComplexRadioDataset(Dataset):
    def __init__(self, data_dir, dtype=np.complex64, max_samples=4096):
        self.data_dir = data_dir
        self.file_pairs = []
        self.dtype = dtype
        self.max_samples = max_samples

        for i in range(5000):
            base = f"{i:05d}"
            complex_path = os.path.join(data_dir, f"{base}.complex")
            text_path = os.path.join(data_dir, f"{base}.txt")
            if os.path.exists(complex_path) and os.path.exists(text_path):
                self.file_pairs.append((complex_path, text_path))

        print(f"{len(self.file_pairs)} samples found in '{data_dir}'")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        complex_path, text_path = self.file_pairs[idx]

        # --- Load IQ data ---
        data = np.fromfile(complex_path, dtype=self.dtype)

        if self.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
            data = data.view(np.complex64)
        else:
            max_val = np.abs(data).max() + 1e-8
            data = data / max_val

        if len(data) > self.max_samples:
            data = data[:self.max_samples]
        elif len(data) < self.max_samples:
            padded = np.zeros(self.max_samples, dtype=np.complex64)
            padded[:len(data)] = data
            data = padded

        X = torch.tensor(data, dtype=torch.complex64)

        # --- Load bits from text file ---
        # Expected format — first line: "bits: 0101010110..."
        bits = np.zeros(32, dtype=np.float32)
        with open(text_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("bits:"):
                    bit_str = line.split(":", 1)[1].strip()
                    parsed = np.array([int(b) for b in bit_str if b in ("0", "1")], dtype=np.float32)
                    length = min(len(parsed), 32)
                    bits[:length] = parsed[:length]
                    break

        y = torch.tensor(bits, dtype=torch.float32)
        return X, y


# ============================================================
#                   Complex CNN Layers
# ============================================================

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, stride=stride)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, stride=stride)

    def forward(self, x):
        xr, xi = x.real, x.imag
        real = self.real(xr) - self.imag(xi)
        imag = self.real(xi) + self.imag(xr)
        return torch.complex(real, imag)


class ComplexAbsPool1d(nn.Module):
    """Max-pool based on magnitude — picks the complex entry with the highest magnitude in each window."""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        xr, xi = x.real, x.imag
        mag = torch.sqrt(xr ** 2 + xi ** 2)

        mag_pooled, indices = nn.functional.max_pool1d(
            mag,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            return_indices=True,
        )

        B, C, _ = mag_pooled.shape
        idx = indices.flatten(2)
        xr_pool = xr.flatten(2).gather(2, idx)
        xi_pool = xi.flatten(2).gather(2, idx)
        return torch.complex(xr_pool, xi_pool)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features)
        self.bn_imag = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return torch.complex(self.bn_real(x.real), self.bn_imag(x.imag))


class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(torch.relu(x.real), torch.relu(x.imag))


# ============================================================
#               Complex CNN → Linear Decoder Model
# ============================================================

class ComplexCNNModel32Bit(nn.Module):
    def __init__(self, input_len):
        super().__init__()

        self.conv = nn.Sequential(
            ComplexConv1d(1, 16, kernel_size=9, padding=4),
            ComplexBatchNorm1d(16),
            ComplexReLU(),
            ComplexAbsPool1d(kernel_size=2),

            ComplexConv1d(16, 32, kernel_size=7, padding=3),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
            ComplexAbsPool1d(kernel_size=2),

            ComplexConv1d(32, 64, kernel_size=5, padding=2),
            ComplexBatchNorm1d(64),
            ComplexReLU(),
            ComplexAbsPool1d(kernel_size=2),
        )

        final_len = input_len // 8

        self.fc = nn.Sequential(
            nn.Linear(final_len * 64 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, N)
        x = self.conv(x)

        xr = x.real.reshape(x.size(0), -1)
        xi = x.imag.reshape(x.size(0), -1)
        x = torch.cat([xr, xi], dim=1)

        return torch.sigmoid(self.fc(x))


# ============================================================
#                      Helpers
# ============================================================

def compute_bit_accuracy(preds, targets):
    """Round sigmoid outputs to 0/1 and compare against targets."""
    rounded = torch.round(preds)
    return (rounded == targets).float().mean().item()


def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training & Validation Metrics", fontsize=14)

    axes[0, 0].plot(epochs, train_losses, color="steelblue", marker="o", markersize=3)
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("BCE Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, train_accs, color="darkorange", marker="o", markersize=3)
    axes[0, 1].set_title("Train Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Bit Accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, val_losses, color="steelblue", marker="s", markersize=3, linestyle="--")
    axes[1, 0].set_title("Validation Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("BCE Loss")
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, val_accs, color="darkorange", marker="s", markersize=3, linestyle="--")
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Bit Accuracy")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=150)
    print("Plot saved to training_metrics.png")
    plt.show()


# ============================================================
#                      Training Loop
# ============================================================

def train_model(
    train_dir,
    val_dir=None,       # Pass a separate validation folder, or None to auto-split
    val_split=0.1,      # Only used when val_dir is None
    epochs=20,
    batch_size=8,
    lr=1e-4,
):
    # --- Build datasets ---
    train_dataset = ComplexRadioDataset(train_dir)

    if val_dir is not None:
        val_dataset = ComplexRadioDataset(val_dir)
        print(f"Using separate validation dataset: '{val_dir}'")
    else:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        print(f"Splitting train set: {train_size} train / {val_size} val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)

    # --- Model ---
    sample_X, _ = train_dataset[0]
    input_len = sample_X.numel()
    model = ComplexCNNModel32Bit(input_len)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- History ---
    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(epochs):
        start_time = time.time()

        # -- Train --
        model.train()
        total_loss, total_acc = 0.0, 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_acc  += compute_bit_accuracy(preds.detach(), y_batch)

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_train_acc  = total_acc  / max(1, len(train_loader))

        # -- Validate --
        model.eval()
        total_val_loss, total_val_acc = 0.0, 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                preds = model(X_val)
                loss = criterion(preds, y_val)
                if not torch.isfinite(loss):
                    continue
                total_val_loss += loss.item()
                total_val_acc  += compute_bit_accuracy(preds, y_val)

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        avg_val_acc  = total_val_acc  / max(1, len(val_loader))

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch+1:>3}/{epochs}] "
            f"Train Loss: {avg_train_loss:.6f}  Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.6f}  Acc: {avg_val_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    torch.save(model.state_dict(), "COMPLEX_CNN_radio_model_32bit.pth")
    print("Model saved to COMPLEX_CNN_radio_model_32bit.pth")

    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    return model


if __name__ == "__main__":
    train_model(
        train_dir=r"C:\Users\Maros\Downloads\Masurik_Demod\Masurik_Demod\Dataset-3-Train-constant",
        val_dir=None,       # e.g. val_dir="Validation_data" to use a separate folder
        val_split=0.1,
        epochs=20,
        batch_size=8,
        lr=1e-4,
    )