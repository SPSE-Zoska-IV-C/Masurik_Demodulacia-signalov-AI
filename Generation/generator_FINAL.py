import os
import numpy as np
import random
import argparse
from scipy.signal import firwin, lfilter

def generate_ask(
    bits,
    samp_rate=1_280_000,
    frequency=500_000,
    noise_amp=0.1,
    samples_per_bit=128,
    output_file="output_signal.complex",
    noiseless_file=None,
    metadata_file=None
):
    bits = np.array(bits, dtype=np.float32)

    symbols = np.repeat(bits, samples_per_bit)


    num_taps = 129
    lpf_taps = firwin(numtaps=num_taps, cutoff=8000, fs=samp_rate, window="hamming")
    delay = (num_taps - 1) // 2
    

    padded_symbols = np.concatenate([symbols, np.full(delay, bits[-1])])

    baseband = lfilter(lpf_taps, 1.0, padded_symbols)[delay:]

    baseband = baseband[:len(symbols)]

    t = np.arange(len(baseband)) / samp_rate
    carrier = np.exp(1j * 2 * np.pi * frequency * t)

    modulated = baseband.astype(np.complex64) * carrier.astype(np.complex64)

    noise = (np.random.normal(scale=noise_amp, size=len(modulated)) + 
             1j * np.random.normal(scale=noise_amp, size=len(modulated))).astype(np.complex64)

    output = modulated + noise
    output.tofile(output_file)
    
    if noiseless_file:
        modulated.tofile(noiseless_file)


    if metadata_file:
        with open(metadata_file, "w") as f:
            f.write(f"bits: {''.join(map(str, bits.astype(int)))}\n")
            f.write(f"samples_per_bit: {samples_per_bit}\n")
            f.write(f"samp_rate: {samp_rate}\n")
            f.write(f"frequency: {frequency}\n")
            f.write(f"noise_amp: {noise_amp}\n")

def generate_bulk(folder, num_files, bits_min, bits_max, freq_min, freq_max, 
                  spb_min, spb_max, noise_min, noise_max, samp_rate, progress_callback=None):
    os.makedirs(folder, exist_ok=True)
    for i in range(num_files):
        if progress_callback: progress_callback(i, num_files)
        
        numbits = random.randint(int(bits_min), int(bits_max))
        bits = [random.randint(0, 1) for _ in range(numbits)]
        freq = random.uniform(freq_min, freq_max)  
        samples_per_bit = random.randint(int(spb_min), int(spb_max))
        noise_amp = random.uniform(noise_min, noise_max)  
        
        base_name = os.path.join(folder, f"{i:05d}")
        
        generate_ask(
            bits=bits, 
            samp_rate=samp_rate, 
            frequency=freq, 
            noise_amp=noise_amp,
            samples_per_bit=samples_per_bit,
            output_file=f"{base_name}.complex",
            noiseless_file=f"{base_name}_noiseless.complex",
            metadata_file=f"{base_name}.txt"
        )
    if progress_callback: progress_callback(num_files, num_files)



def load_complex_file(filepath):
    data = np.fromfile(filepath, dtype=np.complex64)
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate ASK-modulated IQ samples (no GNU Radio)")
    parser.add_argument("--bulk-generate", type=str, default=None, help="Folder to generate 5000 pairs of noisy/noiseless complex files")
    parser.add_argument("--outfile", type=str, default="output_signal.complex")
    parser.add_argument("--noiseless-outfile", type=str, default=None, help="Output file for noiseless signal")
    parser.add_argument("--bits", type=str, default="random")
    parser.add_argument("--bits-outfile", type=str, default="demodulated_bits.txt")
    parser.add_argument("--numbits", type=int, default=32)
    parser.add_argument("--samp-rate", type=float, default=1_280_000)
    parser.add_argument("--freq", type=float, default=500_000)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--samples-per-bit", type=int, default=128)

    args = parser.parse_args()

    if args.bulk_generate:
        generate_bulk(
            folder=args.bulk_generate,
            num_files=20000,
            bits_min=1, bits_max=32,
            freq_min=100_000, freq_max=1_000_000,
            spb_min=32, spb_max=256,
            noise_min=0.1, noise_max=2.0,
            samp_rate=args.samp_rate
        )
        return

    if args.bits == "random":
        bits = [random.randint(0, 1) for _ in range(args.numbits)]
    else:
        bits = [int(b) for b in args.bits.strip()]

    try:
        with open(args.bits_outfile, "w") as f:
            f.write("".join(str(b) for b in bits) + "\n")
    except Exception as e:
        print(f"Warning: could not write bits file: {e}")

    noiseless_file = args.noiseless_outfile
    if noiseless_file is None:
        if '.' in args.outfile:
            noiseless_file = args.outfile.rsplit('.', 1)[0] + '_noiseless.complex'
        else:
            noiseless_file = args.outfile + '_noiseless.complex'

    generate_ask(
        bits=bits,
        samp_rate=args.samp_rate,
        frequency=args.freq,
        noise_amp=args.noise,
        samples_per_bit=args.samples_per_bit,
        output_file=args.outfile,
        noiseless_file=noiseless_file
    )

if __name__ == "__main__":
    main()
