import os
import re
import inspect
import importlib
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader



class ComplexWaveformDataset(Dataset):
    def __init__(self, root_dir, num_samples=9000):
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.noisy_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".complex") and "_noiseless" not in f
            and "_denoised" not in f
        ])

    def __len__(self):
        return len(self.noisy_files)

    def _load_complex(self, path):
        data = np.fromfile(path, dtype=np.complex64)
        if len(data) > self.num_samples:
            data = data[:self.num_samples]
        elif len(data) < self.num_samples:
            data = np.pad(data, (0, self.num_samples - len(data)))
        return data

    def __getitem__(self, idx):
        noisy_name = self.noisy_files[idx]
        clean_name = noisy_name.replace(".complex", "_noiseless.complex")
        noisy = self._load_complex(os.path.join(self.root_dir, noisy_name))
        clean = self._load_complex(os.path.join(self.root_dir, clean_name))
        return (
            torch.from_numpy(np.stack([noisy.real, noisy.imag], axis=0)).float(),
            torch.from_numpy(np.stack([clean.real, clean.imag], axis=0)).float(),
            noisy_name
        )



def _lowpass_filter(signal, cutoff_ratio=0.05, order=5):
    b, a = butter(order, cutoff_ratio, btype='low')
    return filtfilt(b, a, signal)


def _ask_demodulate(iq_samples, samples_per_bit):
    envelope = np.abs(iq_samples)
    filtered = _lowpass_filter(envelope)
    filtered = filtered / (np.max(filtered) + 1e-9)
    threshold = (np.percentile(filtered, 75) + np.percentile(filtered, 25)) / 2
    bits = []
    start_index = (samples_per_bit + samples_per_bit // 2) // 2
    for i in range(start_index, len(filtered), samples_per_bit):
        start = max(0, i - samples_per_bit // 2)
        end   = min(len(filtered), i + samples_per_bit // 2)
        mean_val = np.mean(filtered[start:end])
        bits.append(1 if mean_val > threshold else 0)
    return bits


def _resample_complex(signal, target_len):
    src_len = len(signal)
    if src_len == target_len:
        return signal.astype(np.complex64)
    old_idx = np.linspace(0, src_len - 1, src_len)
    new_idx = np.linspace(0, src_len - 1, target_len)
    return (np.interp(new_idx, old_idx, signal.real)
            + 1j * np.interp(new_idx, old_idx, signal.imag)).astype(np.complex64)


def _parse_metadata(txt_path):
    if not os.path.exists(txt_path):
        return None, None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        bits_match = re.search(r"bits:\s*([01]+)", text)
        spb_match  = re.search(r"samples_per_bit:\s*(\d+)", text)
        if not bits_match or not spb_match:
            return None, None
        return [int(b) for b in bits_match.group(1)], int(spb_match.group(1))
    except Exception:
        return None, None



def _evaluate_accuracy(model, dataset, device, max_files=30):
    root_dir = dataset.root_dir
    total_bits = total_correct = 0
    indices = np.random.choice(len(dataset),
                               size=min(max_files, len(dataset)),
                               replace=False)
    with torch.no_grad():
        for idx in indices:
            noisy_tensor, _, noisy_name = dataset[idx]
            true_bits, spb = _parse_metadata(
                os.path.join(root_dir, noisy_name.replace(".complex", ".txt")))
            if true_bits is None:
                continue
            out = model(noisy_tensor.unsqueeze(0).to(device))
            out_np = out.squeeze(0).cpu().numpy()
            denoised_iq = _resample_complex(
                out_np[0] + 1j * out_np[1], len(true_bits) * spb)
            pred_bits = _ask_demodulate(denoised_iq, spb)
            min_len = min(len(pred_bits), len(true_bits))
            if min_len == 0:
                continue
            total_correct += sum(p == t for p, t in
                                 zip(pred_bits[:min_len], true_bits[:min_len]))
            total_bits += min_len
    return (total_correct / total_bits) if total_bits > 0 else None


def _evaluate_loss(model, dataset, device, criterion, max_files=30):
    indices = np.random.choice(len(dataset),
                               size=min(max_files, len(dataset)),
                               replace=False)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in indices:
            noisy, clean, _ = dataset[idx]
            noisy = noisy.unsqueeze(0).to(device)
            clean = clean.unsqueeze(0).to(device)
            loss = criterion(model(noisy), clean)
            total_loss += loss.item()
            count += 1
    return (total_loss / count) if count > 0 else None


def get_available_models(directory="../Model_architectures"):
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".py")]


def load_model_instance(filepath):
    spec = importlib.util.spec_from_file_location("dynamic_model", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            return obj()
    raise ValueError("Nenašla sa nn.Module trieda.")




EMPTY_HISTORY = {
    "steps":          [],
    "losses":         [],
    "acc_steps":      [],
    "acc_values":     [],
    "val_loss_steps": [],
    "val_losses":     [],
    "val_acc_steps":  [],
    "val_acc_values": [],
}


def train_model_process(dataset_path, model_path, batch_size, epochs, lr,
                        save_name, val_path=None, set_progress=None):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset   = ComplexWaveformDataset(dataset_path)
    loader    = DataLoader(dataset, batch_size=batch_size,
                           shuffle=True, num_workers=0)
    has_val   = val_path and os.path.isdir(val_path) and len(
                    [f for f in os.listdir(val_path)
                     if f.endswith(".complex") and "_noiseless" not in f]) > 0
    val_dataset = ComplexWaveformDataset(val_path) if has_val else None

    model     = load_model_instance(model_path).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history      = {k: list(v) for k, v in EMPTY_HISTORY.items()}
    total_steps  = len(loader) * epochs
    current_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (noisy, clean, _) in enumerate(loader):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            loss = criterion(model(noisy), clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            current_step += 1
            avg_loss = running_loss / (i + 1)

            if i % 5 == 0 or i == len(loader) - 1:
                history["steps"].append(current_step)
                history["losses"].append(round(avg_loss, 6))

            if current_step % 10 == 0:
                model.eval()

                acc = _evaluate_accuracy(model, dataset, device, max_files=30)
                if acc is not None:
                    history["acc_steps"].append(current_step)
                    history["acc_values"].append(round(acc * 100, 2))

                if val_dataset is not None:
                    v_loss = _evaluate_loss(model, val_dataset, device,
                                            criterion, max_files=30)
                    v_acc  = _evaluate_accuracy(model, val_dataset, device,
                                               max_files=30)
                    if v_loss is not None:
                        history["val_loss_steps"].append(current_step)
                        history["val_losses"].append(round(v_loss, 6))
                    if v_acc is not None:
                        history["val_acc_steps"].append(current_step)
                        history["val_acc_values"].append(round(v_acc * 100, 2))

                model.train()

            if set_progress and (i % 5 == 0 or i == len(loader) - 1):
                last_acc  = (f"{history['acc_values'][-1]:.1f}%"
                             if history["acc_values"] else "–")
                last_vacc = (f"{history['val_acc_values'][-1]:.1f}%"
                             if history["val_acc_values"] else "–")
                status = (f"Epoch {epoch+1}/{epochs} | "
                          f"Step {current_step}/{total_steps} | "
                          f"Loss: {avg_loss:.5f} | "
                          f"Acc: {last_acc} | Val Acc: {last_vacc}")
                set_progress((current_step, total_steps, status,
                              {k: list(v) for k, v in history.items()}))

    torch.save(model.state_dict(), save_name)
    return f"Hotovo, model uložený ako {save_name}"
