

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

import new_demodulation as demod


def read_ground_truth(path: str) -> str:
    with open(path, 'r') as f:
        txt = f.read()
    
    bits = ''.join(ch for ch in txt if ch in '01')
    return bits


def compare_bits(gt: str, rx: str) -> float:
    """Return percentage of correct bits with respect to ground truth length.
    Missing received bits are considered incorrect.
    """
    if len(gt) == 0:
        return 0.0
    matches = 0
    for i, ch in enumerate(gt):
        if i < len(rx) and rx[i] == ch:
            matches += 1
    return matches / len(gt) * 100.0


def evaluate_folder(folder: str, samples_per_bit: int, mode: str, out_csv: str) -> int:
    entries: List[dict] = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.complex')])
    if not files:
        print(f"No .complex files found in {folder}", file=sys.stderr)
        return 2

    total_bits = 0
    total_matches = 0

    for fname in files:
        complex_path = os.path.join(folder, fname)
        base = fname[:-len('.complex')]
        gt_name = f"{base}_text.txt"
        gt_path = os.path.join(folder, gt_name)
        if not os.path.exists(gt_path):
            print(f"Warning: ground truth {gt_name} not found for {fname}, skipping", file=sys.stderr)
            continue

        
        samples = demod.read_complex64_file(complex_path)
        if samples.size == 0:
            print(f"Warning: {fname} contains no samples, skipping", file=sys.stderr)
            continue

        envelope = abs(samples)
        rx_bits_list, offset, _ = demod.demodulate_envelope(envelope, samples_per_bit, mode)
        rx_bits = ''.join(str(b) for b in rx_bits_list)

        gt_bits = read_ground_truth(gt_path)

        pct = compare_bits(gt_bits, rx_bits)

        matches = int(round(pct * len(gt_bits) / 100.0)) if len(gt_bits) > 0 else 0
        total_bits += len(gt_bits)
        total_matches += matches

        entries.append({
            'file': fname,
            'gt_len': len(gt_bits),
            'rx_len': len(rx_bits),
            'matches': matches,
            'accuracy_pct': f"{pct:.2f}",
            'offset': offset,
        })

    
    overall_pct = (total_matches / total_bits * 100.0) if total_bits > 0 else 0.0

    
    with open(out_csv, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['file', 'gt_len', 'rx_len', 'matches', 'accuracy_pct', 'offset'])
        writer.writeheader()
        for e in entries:
            writer.writerow(e)
        writer.writerow({'file': 'OVERALL', 'gt_len': total_bits, 'rx_len': '', 'matches': total_matches, 'accuracy_pct': f"{overall_pct:.2f}", 'offset': ''})

    print(f"Wrote evaluation to {out_csv}")
    print(f"Overall accuracy: {overall_pct:.2f}% ({total_matches}/{total_bits})")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate demodulation over a folder')
    parser.add_argument('--folder', required=True, help='Folder containing .complex and _text.txt files')
    parser.add_argument('--samples-per-bit', type=int, default=128, help='Samples per bit')
    parser.add_argument('--mode', choices=['regular', 'mary'], default='regular', help='Demodulation mode')
    parser.add_argument('--out', default='eval_results.csv', help='Output CSV file')

    args = parser.parse_args(argv)
    return evaluate_folder(args.folder, args.samples_per_bit, args.mode, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
