import os
import time
import uuid
import inspect
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split




class ComplexRadioBitsDataset(Dataset):
    def __init__(self, data_dir, dtype=np.complex64, max_samples=9000, num_bits=32):
        self.data_dir = data_dir
        self.dtype = dtype
        self.max_samples = max_samples
        self.num_bits = num_bits
        self.file_pairs = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")

        complex_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".complex")])

        for fname in complex_files:
            base = os.path.splitext(fname)[0]
            complex_path = os.path.join(data_dir, f"{base}.complex")
            text_path = os.path.join(data_dir, f"{base}.txt")
            if os.path.exists(text_path):
                self.file_pairs.append((complex_path, text_path))

        if not self.file_pairs:
            raise RuntimeError(f"No valid (.complex, .txt) pairs found in '{data_dir}'")

        print(f"{len(self.file_pairs)} paired samples found in '{data_dir}'")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        complex_path, text_path = self.file_pairs[idx]

        data = np.fromfile(complex_path, dtype=self.dtype)

        if self.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
            data = data.view(np.complex64)
        elif self.dtype in (np.complex64, np.float32):
            max_val = np.abs(data).max() + 1e-8
            data = data / max_val

        if len(data) > self.max_samples:
            data = data[:self.max_samples]
        elif len(data) < self.max_samples:
            padded = np.zeros(self.max_samples, dtype=np.complex64)
            padded[:len(data)] = data
            data = padded

        x = np.stack((data.real, data.imag), axis=0).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32)

        bits = np.zeros(self.num_bits, dtype=np.float32)
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("bits:"):
                    bit_str = line.split(":", 1)[1].strip()
                    parsed = np.array([int(b) for b in bit_str if b in ("0", "1")], dtype=np.float32)
                    length = min(len(parsed), self.num_bits)
                    bits[:length] = parsed[:length]
                    break

        y = torch.tensor(bits, dtype=torch.float32)
        return x, y



def compute_bit_accuracy(preds, targets):
    rounded = torch.round(preds)
    return (rounded == targets).float().mean().item()


def load_model_from_architecture(arch_path, max_samples=9000, num_bits=32):

    module_name = f"dynamic_model_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, arch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import architecture file: {arch_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "build_model") and callable(module.build_model):
        fn = module.build_model
        sig = inspect.signature(fn)
        kwargs = {}
        if "max_samples" in sig.parameters:
            kwargs["max_samples"] = max_samples
        if "num_bits" in sig.parameters:
            kwargs["num_bits"] = num_bits
        return fn(**kwargs)

    preferred_names = [
        "ComplexCNN1D",
        "ComplexBitCNN",
        "BitClassifier",
        "Model",
        "Net",
    ]

    for name in preferred_names:
        if hasattr(module, name):
            cls = getattr(module, name)
            if inspect.isclass(cls) and issubclass(cls, nn.Module):
                sig = inspect.signature(cls.__init__)
                kwargs = {}
                if "max_samples" in sig.parameters:
                    kwargs["max_samples"] = max_samples
                if "num_bits" in sig.parameters:
                    kwargs["num_bits"] = num_bits
                return cls(**kwargs)

    for _, obj in vars(module).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            sig = inspect.signature(obj.__init__)
            kwargs = {}
            if "max_samples" in sig.parameters:
                kwargs["max_samples"] = max_samples
            if "num_bits" in sig.parameters:
                kwargs["num_bits"] = num_bits
            return obj(**kwargs)

    raise ValueError(
        f"No usable nn.Module found in '{arch_path}'. "
        f"Define build_model(...) or a model class."
    )


def _make_store(train_losses, train_accs, val_losses, val_accs):
    return {
        "steps": list(range(1, len(train_losses) + 1)),
        "losses": train_losses,
        "acc_steps": list(range(1, len(train_accs) + 1)),
        "acc_values": [a * 100.0 for a in train_accs],
        "val_loss_steps": list(range(1, len(val_losses) + 1)),
        "val_losses": val_losses,
        "val_acc_steps": list(range(1, len(val_accs) + 1)),
        "val_acc_values": [a * 100.0 for a in val_accs],
    }


def train_model_process(
    train_path,
    arch_path,
    batch_size,
    epochs,
    lr,
    save_name,
    val_path=None,
    set_progress=None,
    val_split=0.1,
    num_bits=32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset_full = ComplexRadioBitsDataset(train_path, max_samples=9000, num_bits=num_bits)
    max_samples = train_dataset_full.max_samples

    if val_path:
        train_dataset = train_dataset_full
        val_dataset = ComplexRadioBitsDataset(val_path, max_samples=max_samples, num_bits=num_bits)
    else:
        val_size = max(1, int(len(train_dataset_full) * val_split))
        train_size = max(1, len(train_dataset_full) - val_size)
        if train_size + val_size > len(train_dataset_full):
            val_size = len(train_dataset_full) - train_size

        train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          drop_last=False, num_workers=0)

    model = load_model_from_architecture(
        arch_path=arch_path,
        max_samples=max_samples,
        num_bits=num_bits
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    total_train_steps = epochs * len(train_loader)
    total_val_steps = epochs * len(val_loader)
    total_steps = total_train_steps + total_val_steps
    global_step = 0

    step_losses: list = []
    step_accs:   list = []
    val_loss_steps: list = []
    val_losses_log: list = []
    val_acc_steps:  list = []
    val_accs_log:   list = []

    def _make_store_steps():
        return {
            "steps":          list(range(1, len(step_losses) + 1)),
            "losses":         step_losses[:],
            "acc_steps":      list(range(1, len(step_accs) + 1)),
            "acc_values":     [a * 100.0 for a in step_accs],
            "val_loss_steps": val_loss_steps[:],
            "val_losses":     val_losses_log[:],
            "val_acc_steps":  val_acc_steps[:],
            "val_acc_values": [a * 100.0 for a in val_accs_log],
        }

    if set_progress:
        set_progress((
            0,
            max(total_steps, 1),
            "Initializing bit-prediction training...",
            _make_store_steps()
        ))

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        epoch_loss = 0.0
        epoch_acc  = 0.0
        n_train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            preds = torch.clamp(preds, 1e-6, 1 - 1e-6)

            loss = criterion(preds, y_batch)
            if not torch.isfinite(loss):
                global_step += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step_loss = loss.item()
            step_acc  = compute_bit_accuracy(preds.detach(), y_batch)

            step_losses.append(step_loss)
            step_accs.append(step_acc)

            epoch_loss += step_loss
            epoch_acc  += step_acc
            n_train_batches += 1
            global_step += 1


            if set_progress:
                status = (
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {step_loss:.6f}, Acc: {step_acc * 100:.2f}%"
                )
                set_progress((
                    global_step,
                    total_steps,
                    status,
                    _make_store_steps()
                ))

        model.eval()
        total_val_loss = 0.0
        total_val_acc  = 0.0
        n_val_batches  = 0
        n_val_total    = len(val_loader)


        if set_progress:
            status = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Validating... (0/{n_val_total} batches)"
            )
            set_progress((
                global_step,
                total_steps,
                status,
                _make_store_steps()
            ))

        with torch.no_grad():
            for val_batch_idx, (X_val, y_val) in enumerate(val_loader):
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                preds = model(X_val)
                preds = torch.clamp(preds, 1e-6, 1 - 1e-6)

                loss = criterion(preds, y_val)
                if not torch.isfinite(loss):
                    global_step += 1
                    continue

                total_val_loss += loss.item()
                total_val_acc  += compute_bit_accuracy(preds, y_val)
                n_val_batches  += 1
                global_step    += 1

                if set_progress:
                    running_val_loss = total_val_loss / max(1, n_val_batches)
                    running_val_acc  = total_val_acc  / max(1, n_val_batches)
                    status = (
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Validating ({val_batch_idx + 1}/{n_val_total}) | "
                        f"Val Loss: {running_val_loss:.6f}, "
                        f"Val Acc: {running_val_acc * 100:.2f}%"
                    )
                    set_progress((
                        global_step,
                        total_steps,
                        status,
                        _make_store_steps()
                    ))

        avg_val_loss   = total_val_loss / max(1, n_val_batches)
        avg_val_acc    = total_val_acc  / max(1, n_val_batches)
        avg_train_loss = epoch_loss     / max(1, n_train_batches)
        avg_train_acc  = epoch_acc      / max(1, n_train_batches)

        val_loss_steps.append(global_step)
        val_losses_log.append(avg_val_loss)
        val_acc_steps.append(global_step)
        val_accs_log.append(avg_val_acc)

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        epoch_time = time.time() - epoch_start
        status = (
            f"Epoch {epoch + 1}/{epochs} done | "
            f"Train Loss: {avg_train_loss:.6f}, Acc: {avg_train_acc * 100:.2f}% | "
            f"Val Loss: {avg_val_loss:.6f}, Acc: {avg_val_acc * 100:.2f}% | "
            f"{epoch_time:.2f}s"
        )

        if set_progress:
            set_progress((
                global_step,
                total_steps,
                status,
                _make_store_steps()
            ))

        print(status)

    save_path = save_name
    if not save_path.endswith(".pth"):
        save_path += ".pth"

    torch.save(model.state_dict(), save_path)
    return f"Hotovo, model uložený do {save_path}"
