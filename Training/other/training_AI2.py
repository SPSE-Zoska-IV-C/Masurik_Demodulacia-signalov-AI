import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class ComplexRadioDataset(Dataset):
    def __init__(self, data_dir, dtype=np.complex64, max_samples=4096, augment=False):
        self.data_dir = data_dir
        self.file_pairs = []
        self.dtype = dtype
        self.max_samples = max_samples
        self.augment = augment
        for i in range(5000):
            base = f"{i:05d}"
            complex_path = os.path.join(data_dir, f"{base}.complex")
            text_path = os.path.join(data_dir, f"{base}.txt")
            if os.path.exists(complex_path) and os.path.exists(text_path):
                self.file_pairs.append((complex_path, text_path))
        print(f"{len(self.file_pairs)} Pairs found in '{data_dir}'")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        complex_path, text_path = self.file_pairs[idx]
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

        # ── Augmentation (train only) ────────────────────────────────────────
        if self.augment:
            # Random phase rotation — rotates the constellation, label-preserving
            phase = np.random.uniform(0, 2 * np.pi)
            data = data * np.exp(1j * phase).astype(np.complex64)

            # Small additive noise
            noise_std = np.random.uniform(0.0, 0.05)
            noise = (np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape))
            data = data + (noise * noise_std).astype(np.complex64)

            # Re-normalise after augmentation
            max_val = np.abs(data).max() + 1e-8
            data = data / max_val

        X = torch.tensor(data, dtype=torch.complex64)

        # Parse "bits: 0101010110..." format
        with open(text_path, "r") as f:
            bits = np.zeros(32, dtype=np.float32)
            for line in f:
                line = line.strip()
                if line.lower().startswith("bits:"):
                    bit_str = line.split(":", 1)[1].strip()
                    parsed = [float(b) for b in bit_str if b in ("0", "1")]
                    parsed = parsed[:32]
                    bits[:len(parsed)] = parsed
                    break

        y = torch.tensor(bits, dtype=torch.float32)
        return X, y


# ── Complex-valued building blocks ──────────────────────────────────────────

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_real = nn.Linear(in_features, out_features)
        self.W_imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        real = self.W_real(x.real) - self.W_imag(x.imag)
        imag = self.W_real(x.imag) + self.W_imag(x.real)
        return torch.complex(real, imag)


class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(torch.relu(x.real), torch.relu(x.imag))


class ComplexDropout(nn.Module):
    """Applies the same dropout mask to real and imaginary parts."""
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # One mask shared across real & imag so the complex unit is dropped together
        mask = torch.ones(x.real.shape, device=x.real.device)
        mask = torch.nn.functional.dropout(mask, p=self.p, training=True)
        return torch.complex(x.real * mask, x.imag * mask)


class ComplexModel32Bit(nn.Module):
    def __init__(self, input_dim, dropout_p=0.3):
        super().__init__()
        # Reduced width compared to original — fewer params = less overfitting
        self.net = nn.Sequential(
            ComplexLinear(input_dim, 512),  ComplexReLU(), ComplexDropout(dropout_p),
            ComplexLinear(512, 256),        ComplexReLU(), ComplexDropout(dropout_p),
            ComplexLinear(256, 128),        ComplexReLU(), ComplexDropout(dropout_p),
            ComplexLinear(128, 64),         ComplexReLU(), ComplexDropout(dropout_p),
            ComplexLinear(64, 32),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x).real)


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Round float outputs to {0,1} and compare bit-by-bit."""
    rounded = preds.round()
    correct = (rounded == targets).float().mean().item()
    return correct


def plot_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, train_losses, color="steelblue", linewidth=1.8)
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("BCE Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_accs, color="darkorange", linewidth=1.8)
    axes[0, 1].set_title("Train Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Bit Accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, val_losses, color="firebrick", linewidth=1.8)
    axes[1, 0].set_title("Validation Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("BCE Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, val_accs, color="seagreen", linewidth=1.8)
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Bit Accuracy")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("Saved training_history.png")
    plt.show()


# ── Main training loop ───────────────────────────────────────────────────────

def train_model(
    data_dir,
    val_data_dir=None,
    epochs=50,
    batch_size=16,
    lr=1e-4,
    val_split=0.1,
    dropout_p=0.3,          # dropout probability — increase if still overfitting
    weight_decay=1e-4,      # L2 regularization — increase if still overfitting
    augment=True,           # phase rotation + noise augmentation on training data
    patience=10,            # early stopping: stop if val loss doesn't improve
):
    import time

    # ── Datasets ──
    train_full = ComplexRadioDataset(data_dir, augment=augment)

    if val_data_dir is not None:
        print(f"Using separate validation dataset: '{val_data_dir}'")
        val_dataset = ComplexRadioDataset(val_data_dir, augment=False)
        train_dataset = train_full
    else:
        val_size = int(len(train_full) * val_split)
        train_size = len(train_full) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_full, [train_size, val_size]
        )
        print(f"Auto split — train: {train_size}, val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ── Model ──
    sample_X, _ = train_full[0]
    input_dim = sample_X.numel()
    model = ComplexModel32Bit(input_dim, dropout_p=dropout_p)
    criterion = nn.BCELoss()

    # weight_decay = L2 regularization built into Adam
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Reduce LR when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── History & early stopping ──
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss = float("inf")
    best_epoch    = 0
    epochs_no_improve = 0

    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        t0 = time.time()

        # — Train —
        model.train()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        train_bar = tqdm(train_loader, desc=f"  Train {epoch+1}/{epochs}", leave=False, unit="batch")
        for X_batch, y_batch in train_bar:
            X_batch = X_batch.view(X_batch.size(0), -1)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_acc  += compute_accuracy(preds.detach(), y_batch)
            n_batches  += 1
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(1, n_batches)
        avg_train_acc  = total_acc  / max(1, n_batches)

        # — Validate —
        model.eval()
        val_loss, val_acc, val_batches = 0.0, 0.0, 0

        val_bar = tqdm(val_loader, desc=f"  Val   {epoch+1}/{epochs}", leave=False, unit="batch")
        with torch.no_grad():
            for X_val, y_val in val_bar:
                X_val = X_val.view(X_val.size(0), -1)
                preds = model(X_val)
                loss = criterion(preds, y_val)
                if not torch.isfinite(loss):
                    continue
                val_loss += loss.item()
                val_acc  += compute_accuracy(preds, y_val)
                val_batches += 1
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / max(1, val_batches)
        avg_val_acc  = val_acc  / max(1, val_batches)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        scheduler.step(avg_val_loss)

        epoch_bar.set_postfix(
            tr_loss=f"{avg_train_loss:.4f}",
            tr_acc=f"{avg_train_acc:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            val_acc=f"{avg_val_acc:.4f}",
            time=f"{time.time()-t0:.1f}s",
        )

        # — Save best model & early stopping —
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best was epoch {best_epoch})")
                break

    print(f"\nBest val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Best model saved as best_model.pth")

    plot_history(train_losses, val_losses, train_accs, val_accs)
    return model


if __name__ == "__main__":
    train_model(
        data_dir=r"C:\Users\Maros\Downloads\Masurik_Demod\Masurik_Demod\Dataset-3-Train-constant",
        val_data_dir=r"C:\Users\Maros\Downloads\Masurik_Demod\Masurik_Demod\Dataset-3-constant",
    )