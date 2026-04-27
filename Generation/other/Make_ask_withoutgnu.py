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
    noiseless_file=None
):
    bits = np.array(bits, dtype=np.float32)


    symbols = np.repeat(bits, samples_per_bit)


    lpf_taps = firwin(
        numtaps=129,
        cutoff=8000,
        fs=samp_rate,
        window="hamming"
    )


    delay = (len(lpf_taps) - 1) // 2
    pad = np.zeros(delay, dtype=np.float32)
    symbols = np.concatenate([symbols, pad])

    baseband = lfilter(lpf_taps, 1.0, symbols)

    t = np.arange(len(baseband)) / samp_rate
    carrier = np.exp(1j * 2 * np.pi * frequency * t)


    modulated = baseband.astype(np.complex64) * carrier.astype(np.complex64)

    noise = (
        np.random.normal(scale=noise_amp, size=len(modulated))
        + 1j * np.random.normal(scale=noise_amp, size=len(modulated))
    ).astype(np.complex64)


    output = modulated + noise

    # Write noisy output
    output.astype(np.complex64).tofile(output_file)
    print(f"Written {len(output)} complex samples to {output_file}")

    # Write noiseless output if requested
    if noiseless_file:
        modulated.astype(np.complex64).tofile(noiseless_file)
        print(f"Written {len(modulated)} complex samples to {noiseless_file} (noiseless)")


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
    parser.add_argument("--bits-per-sample", type=int, default=128)

    args = parser.parse_args()

    if args.bulk_generate:
        folder = args.bulk_generate
        os.makedirs(folder, exist_ok=True)
        for i in range(20000):
            numbits = random.randint(1, 32)
            bits = [random.randint(0, 1) for _ in range(numbits)]
            freq = random.uniform(100_000, 1_000_000)  
            samples_per_bit = random.randint(32, 256)
            noise_amp = random.uniform(0.1, 2)  
            noisy_file = os.path.join(folder, f"{i:05d}.complex")
            noiseless_file = os.path.join(folder, f"{i:05d}_noiseless.complex")
            generate_ask(
                bits=bits,
                samp_rate=args.samp_rate,
                frequency=freq,
                noise_amp=noise_amp,
                samples_per_bit=samples_per_bit,
                output_file=noisy_file,
                noiseless_file=noiseless_file
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
        # Default to outfile name with '_noiseless' before extension
        if '.' in args.outfile:
            noiseless_file = args.outfile.rsplit('.', 1)[0] + '_noiseless.complex'
        else:
            noiseless_file = args.outfile + '_noiseless.complex'

    generate_ask(
        bits=bits,
        samp_rate=args.samp_rate,
        frequency=args.freq,
        noise_amp=args.noise,
        samples_per_bit=args.bits_per_sample,
        output_file=args.outfile,
        noiseless_file=noiseless_file
    )


if __name__ == "__main__":
    main()
