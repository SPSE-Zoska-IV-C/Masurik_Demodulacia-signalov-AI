import os
import numpy as np
import torch
from training_ai import ComplexModel32Bit  


def load_complex_file(file_path, dtype=np.complex64, max_samples=4096):
    data = np.fromfile(file_path, dtype=dtype)

    if dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
        data = data.view(np.complex64)
    elif dtype in [np.complex64, np.float32]:
        max_val = np.abs(data).max() + 1e-8
        data = data / max_val

    if len(data) > max_samples:
        data = data[:max_samples]
    elif len(data) < max_samples:
        padded = np.zeros(max_samples, dtype=dtype)
        padded[:len(data)] = data
        data = padded

    data_real_imag = np.stack((data.real, data.imag), axis=1).flatten()
    X = torch.tensor(data_real_imag, dtype=torch.float32).unsqueeze(0)
    return X


def evaluate_model(model_path, data_dir, dtype=np.complex64, max_samples=4096):
    
    file_pairs = []
    for i in range(5000):
        base = f"{i:06d}"
        complex_path = os.path.join(data_dir, f"{base}.complex")
        text_path = os.path.join(data_dir, f"{base}_text.txt")
        if os.path.exists(complex_path) and os.path.exists(text_path):
            file_pairs.append((complex_path, text_path))

    if not file_pairs:
        print("No valid file pairs found!")
        return

    
    sample_X = load_complex_file(file_pairs[0][0], dtype, max_samples)
    input_dim = sample_X.shape[1]

    
    model = ComplexModel32Bit(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    total_correct = 0
    total_bits = 0

    print(f"Evaluating {len(file_pairs)} files...\n")

    for complex_path, text_path in file_pairs:
        X = load_complex_file(complex_path, dtype, max_samples)

        with open(text_path, "r") as f:
            val = np.float32(f.read().strip())

        true_bits = np.unpackbits(np.frombuffer(val.tobytes(), dtype=np.uint8))


        with torch.no_grad():
            preds = model(X).squeeze().numpy()
        pred_bits = (preds > 0.5).astype(np.uint8)

        correct = np.sum(pred_bits == true_bits)
        acc = (correct / 32.0) * 100.0

        total_correct += correct
        total_bits += 32

        base_name = os.path.basename(complex_path)
        print(f"{base_name}: {acc:.2f}% bit accuracy")

    avg_acc = (total_correct / total_bits) * 100.0
    print(f"\nAverage bit accuracy: {avg_acc:.2f}%")



if __name__ == "__main__":
    model_file = "ANN_radio_model_32bit.pth"
    data_dir = "Testing_data3"
    evaluate_model(model_file, data_dir)
