import os
import re
import numpy as np
import torch
import torch.nn as nn

DATA_DIR = r"C:\Users\masur\Downloads\Masurik_Demodulacia_signalov_AI-main\Masurik_Demodulacia_signalov_AI-main\Dataset-2"
MODEL_PATH = r"C:\Users\masur\Downloads\Masurik_Demodulacia_signalov_AI-main\Masurik_Demodulacia_signalov_AI-main\denoiser_FINAL.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    if len(data) == 0:
        raise ValueError(f"{path} is empty")
    return data


def parse_metadata(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing metadata file: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    bits_match = re.search(r"bits:\s*([01]+)", text)
    spb_match = re.search(r"samples_per_bit:\s*(\d+)", text)

    if not bits_match:
        raise ValueError(f"Could not find 'bits:' in {txt_path}")
    if not spb_match:
        raise ValueError(f"Could not find 'samples_per_bit:' in {txt_path}")

    bits = bits_match.group(1)
    samples_per_bit = int(spb_match.group(1))
    target_len = len(bits) * samples_per_bit

    return bits, samples_per_bit, target_len


def resample_complex(signal, target_len):
    src_len = len(signal)

    if src_len == target_len:
        return signal.astype(np.complex64)

    old_idx = np.linspace(0, src_len - 1, src_len)
    new_idx = np.linspace(0, src_len - 1, target_len)

    real_resampled = np.interp(new_idx, old_idx, signal.real)
    imag_resampled = np.interp(new_idx, old_idx, signal.imag)

    return (real_resampled + 1j * imag_resampled).astype(np.complex64)


def denoise_signal(model, noisy_complex):
    x = np.stack([noisy_complex.real, noisy_complex.imag], axis=0)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)  # [1, 2, N]

    with torch.no_grad():
        y = model(x)[0].cpu().numpy()

    denoised = y[0] + 1j * y[1]
    return denoised.astype(np.complex64)


def should_process(filename):
    if not filename.endswith(".complex"):
        return False
    if filename.endswith("_noiseless.complex"):
        return False
    if filename.endswith("_denoised.complex"):
        return False
    return True


def main():
    model = ComplexDenoiser().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    files = sorted(f for f in os.listdir(DATA_DIR) if should_process(f))

    if not files:
        print("No input .complex files found.")
        return

    print(f"Found {len(files)} file(s) to process.")

    for filename in files:
        input_path = os.path.join(DATA_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(DATA_DIR, f"{base_name}.txt")
        output_path = os.path.join(DATA_DIR, f"{base_name}_denoised.complex")

        try:
            noisy = load_complex(input_path)
            bits, samples_per_bit, target_len = parse_metadata(txt_path)

            denoised = denoise_signal(model, noisy)
            denoised_resampled = resample_complex(denoised, target_len)
            denoised_resampled.tofile(output_path)

            print(
                f"[OK] {filename} -> {os.path.basename(output_path)} | "
                f"bits={len(bits)}, samples_per_bit={samples_per_bit}, "
                f"output_samples={target_len}"
            )

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")


if __name__ == "__main__":
    main()