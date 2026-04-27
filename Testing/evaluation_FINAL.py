import os
import re
import importlib.util
import numpy as np
import torch
from scipy.signal import butter, filtfilt



def lowpass_filter(signal: np.ndarray, cutoff_ratio: float = 0.05, order: int = 5) -> np.ndarray:
    b, a = butter(order, cutoff_ratio, btype="low")
    return filtfilt(b, a, signal)


def ask_demodulate(iq_samples: np.ndarray, samples_per_bit: int) -> list[int]:
    envelope = np.abs(iq_samples)
    filtered = lowpass_filter(envelope, cutoff_ratio=0.05)
    filtered = filtered / (np.max(filtered) + 1e-9)

    threshold = (np.percentile(filtered, 75) + np.percentile(filtered, 25)) / 2

    bits: list[int] = []
    sample_points: list[int] = []

    start_index = (samples_per_bit + samples_per_bit // 2) // 2
    for i in range(start_index, len(filtered), samples_per_bit):
        start = max(0, i - samples_per_bit // 2)
        end   = min(len(filtered), i + samples_per_bit // 2)
        mean_val = np.mean(filtered[start:end])
        bits.append(1 if mean_val > threshold else 0)
        sample_points.append(i)

    return bits, sample_points, filtered, threshold



def parse_txt_metadata(txt_path: str) -> dict:

    meta = {}
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "bits":
                meta["bits"] = [int(b) for b in val if b in "01"]
            elif key == "samples_per_bit":
                meta["samples_per_bit"] = int(val)
            elif key == "samp_rate":
                meta["samp_rate"] = int(float(val))
            elif key == "frequency":
                meta["frequency"] = float(val)
            elif key == "noise_amp":
                meta["noise_amp"] = float(val)
    return meta



_ID_PATTERN = re.compile(r"^\d{5}$")


def list_evaluable_files(folder: str) -> list[str]:
    if not folder or not os.path.exists(folder):
        return []
    ids = []
    for fname in os.listdir(folder):
        name, ext = os.path.splitext(fname)
        if ext == ".complex" and _ID_PATTERN.match(name):
            if os.path.exists(os.path.join(folder, name + ".txt")):
                ids.append(name)
    return sorted(ids)


def list_weight_files(base_dir: str) -> list[str]:
    if not os.path.exists(base_dir):
        return []
    return sorted([f for f in os.listdir(base_dir) if f.endswith(".pth")])


def get_available_models(arch_dir: str) -> list[str]:
    if not os.path.exists(arch_dir):
        return []
    return sorted([f for f in os.listdir(arch_dir) if f.endswith(".py")])



def _load_model(arch_path: str, weights_path: str, device: torch.device):
    spec   = importlib.util.spec_from_file_location("arch_module", arch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


    model_cls = None
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        try:
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
                model_cls = obj
                break
        except TypeError:
            pass

    if model_cls is None:
        raise ValueError(f"No nn.Module subclass found in {arch_path}")

    model = model_cls().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_evaluation_model(arch_path: str, weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(arch_path, weights_path, device)
    return model, device


def _resample_to_original(signal: np.ndarray, original_length: int) -> np.ndarray:

    n_in  = len(signal)
    n_out = original_length
    if n_in == n_out:
        return signal

    spectrum = np.fft.fft(signal)
    if n_out < n_in:
        half = n_out // 2
        new_spectrum = np.concatenate([spectrum[:half], spectrum[n_in - half:]])
    else:
        half = n_in // 2
        new_spectrum = np.concatenate([
            spectrum[:half],
            np.zeros(n_out - n_in, dtype=complex),
            spectrum[half:]
        ])
    resampled = np.fft.ifft(new_spectrum) * (n_out / n_in)
    return resampled.astype(np.complex64)



def _snr_db(signal: np.ndarray, reference: np.ndarray) -> float:
    noise  = signal - reference
    s_pow  = np.mean(np.abs(reference) ** 2)
    n_pow  = np.mean(np.abs(noise)     ** 2)
    if n_pow < 1e-12:
        return float("inf")
    return 10 * np.log10(s_pow / n_pow)




AUGMENTED_LEN = 9000   # the fixed length used during training


def run_evaluation(
    data_dir: str,
    arch_path: str,
    weights_path: str,
    file_id: str,
    progress_callback=None,
    model=None,
    device=None,
) -> dict:
    def _prog(msg, cur, total):
        if progress_callback:
            progress_callback(msg, cur, total)

    _prog("Súbor sa načítava", 0, 5)

    complex_path = os.path.join(data_dir, file_id + ".complex")
    txt_path     = os.path.join(data_dir, file_id + ".txt")

    if not os.path.exists(complex_path):
        raise FileNotFoundError(f"Missing {complex_path}")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing {txt_path}")

    noisy_aug = np.fromfile(complex_path, dtype=np.complex64)
    meta      = parse_txt_metadata(txt_path)

    samples_per_bit  = meta["samples_per_bit"]
    bits_true        = meta["bits"]
    original_length  = len(bits_true) * samples_per_bit

    _prog("Model sa spúšťa", 1, 5)

    if model is None or device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = _load_model(arch_path, weights_path, device)

    x = np.stack([noisy_aug.real, noisy_aug.imag], axis=0)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.inference_mode():
        y_t = model(x_t)

    y = y_t.squeeze(0).cpu().numpy()
    denoised_aug = (y[0] + 1j * y[1]).astype(np.complex64)

    _prog("Prevzorkovávanie", 2, 5)

    noisy_orig    = _resample_to_original(noisy_aug,    original_length)
    denoised_orig = _resample_to_original(denoised_aug, original_length)

    noiseless_path = os.path.join(data_dir, file_id + "_noiseless.complex")
    if os.path.exists(noiseless_path):
        noiseless_aug  = np.fromfile(noiseless_path, dtype=np.complex64)
        noiseless_orig = _resample_to_original(noiseless_aug, original_length)
        has_noiseless  = True
    else:
        noiseless_aug  = None
        noiseless_orig = None
        has_noiseless  = False

    _prog("Demodácia", 3, 5)

    bits_noisy,    sp_noisy,    filt_noisy,    thr_noisy    = ask_demodulate(noisy_orig,    samples_per_bit)
    bits_denoised, sp_denoised, filt_denoised, thr_denoised = ask_demodulate(denoised_orig, samples_per_bit)

    _prog("Počítanie metrík", 4, 5)

    def _ber(pred, ref):
        n = min(len(pred), len(ref))
        if n == 0:
            return 1.0
        return sum(p != r for p, r in zip(pred[:n], ref[:n])) / n

    ber_noisy    = _ber(bits_noisy,    bits_true)
    ber_denoised = _ber(bits_denoised, bits_true)

    if has_noiseless:
        ref = noiseless_orig
        mse_noisy    = float(np.mean(np.abs(noisy_orig    - ref) ** 2))
        mse_denoised = float(np.mean(np.abs(denoised_orig - ref) ** 2))
        snr_noisy    = _snr_db(noisy_orig,    ref)
        snr_denoised = _snr_db(denoised_orig, ref)
    else:
        mse_noisy    = float(np.mean(np.abs(noisy_orig - denoised_orig) ** 2))
        mse_denoised = 0.0
        snr_noisy    = _snr_db(noisy_orig, denoised_orig)
        snr_denoised = float("inf")

    _prog("Hotovo!", 5, 5)

    return {
        "mode":             "denoise",
        "noisy":            noisy_orig,
        "denoised":         denoised_orig,
        "noiseless":        noiseless_orig,
        "has_noiseless":    has_noiseless,
        "noisy_aug":        noisy_aug,
        "denoised_aug":     denoised_aug,
        "bits_true":        bits_true,
        "bits_noisy":       bits_noisy,
        "bits_denoised":    bits_denoised,
        "sample_points_noisy":    sp_noisy,
        "sample_points_denoised": sp_denoised,
        "filtered_noisy":         filt_noisy,
        "filtered_denoised":      filt_denoised,
        "threshold_noisy":        float(thr_noisy),
        "threshold_denoised":     float(thr_denoised),
        "mse_noisy":        mse_noisy,
        "mse_denoised":     mse_denoised,
        "snr_noisy":        float(snr_noisy),
        "snr_denoised":     float(snr_denoised),
        "ber_noisy":        ber_noisy,
        "ber_denoised":     ber_denoised,
        "samples_per_bit":  samples_per_bit,
        "file_id":          file_id,
        "device":           str(device),
    }


def run_evaluation_bits(
    data_dir: str,
    arch_path: str,
    weights_path: str,
    file_id: str,
    progress_callback=None,
    model=None,
    device=None,
) -> dict:
    def _prog(msg, cur, total):
        if progress_callback:
            progress_callback(msg, cur, total)

    _prog("Načítavanie súboru", 0, 5)

    complex_path = os.path.join(data_dir, file_id + ".complex")
    txt_path     = os.path.join(data_dir, file_id + ".txt")

    if not os.path.exists(complex_path):
        raise FileNotFoundError(f"Missing {complex_path}")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing {txt_path}")

    noisy_aug = np.fromfile(complex_path, dtype=np.complex64)
    meta      = parse_txt_metadata(txt_path)
    bits_true_all = meta["bits"]
    bits_true = bits_true_all[:32]
    samples_per_bit = meta["samples_per_bit"]

    _prog("Spúšťanie modelu", 1, 5)

    if model is None or device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = _load_model(arch_path, weights_path, device)

    x = np.stack([noisy_aug.real, noisy_aug.imag], axis=0)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.inference_mode():
        y_t = model(x_t)

    y = y_t.squeeze().cpu().numpy()

    _prog("Dekódovanie bitov", 2, 5)

    if np.any(y > 1.0) or np.any(y < 0.0):
        y = 1.0 / (1.0 + np.exp(-y.astype(np.float64)))

    bits_predicted = [int(b > 0.5) for b in y]

    _prog("Tradičná ASK demodulácia", 3, 5)

    original_length = len(bits_true_all) * samples_per_bit
    noisy_orig = _resample_to_original(noisy_aug, original_length)
    bits_traditional_all, _, _, _ = ask_demodulate(noisy_orig, samples_per_bit)
    bits_traditional = bits_traditional_all[:32]

    _prog("Počítanie metrík", 4, 5)

    n = min(len(bits_predicted), len(bits_true))
    ber = 1.0 if n == 0 else sum(p != r for p, r in zip(bits_predicted[:n], bits_true[:n])) / n

    n_trad = min(len(bits_traditional), len(bits_true))
    ber_traditional = 1.0 if n_trad == 0 else sum(
        p != r for p, r in zip(bits_traditional[:n_trad], bits_true[:n_trad])
    ) / n_trad

    _prog("Hotovo", 5, 5)

    return {
        "mode":              "bits",
        "bits_true":         bits_true,
        "bits_predicted":    bits_predicted,
        "bits_traditional":  bits_traditional,
        "ber":               ber,
        "ber_traditional":   ber_traditional,
        "file_id":           file_id,
        "device":            str(device),
    }



_AUGMENTED_SAMPLES = 9000


def _ber(pred, ref):
    n = min(len(pred), len(ref))
    if n == 0:
        return 1.0
    return sum(p != r for p, r in zip(pred[:n], ref[:n])) / n


def _load_and_pad(path: str, num_samples: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.complex64)
    if len(data) > num_samples:
        data = data[:num_samples]
    elif len(data) < num_samples:
        data = np.pad(data, (0, num_samples - len(data)))
    return data


def _load_and_pad_normalised(path: str, num_samples: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.complex64)
    max_val = np.abs(data).max() + 1e-8
    data = data / max_val
    if len(data) > num_samples:
        data = data[:num_samples]
    elif len(data) < num_samples:
        padded = np.zeros(num_samples, dtype=np.complex64)
        padded[:len(data)] = data
        data = padded
    return data






