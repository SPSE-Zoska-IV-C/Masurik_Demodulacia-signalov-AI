from gnuradio import gr, blocks, analog, filter
from gnuradio.filter import firdes
import numpy as np
import random
import argparse
from gnuradio.fft import window


class CreateASK(gr.top_block):

    def __init__(self, bits, samp_rate=1280000, frequency=500000,
                 noise_amp=1.5, output_file="output_signal.complex",
                 samples_per_bit=128):
        gr.top_block.__init__(self, "ASK Generator")

        self.samp_rate = samp_rate
        self.frequency = frequency
        self.noise_amp = noise_amp
        self.bits = bits
        self.output_file = output_file
        self.samples_per_bit = samples_per_bit

        
        symbols = np.array(bits, dtype=np.complex64)

        
        
        lpf_taps = firdes.low_pass(
            1.0, samp_rate,
            8000, 20000,
            window.WIN_HAMMING,
            6.76
        )

        
        filter_delay_samples = (len(lpf_taps) - 1) // 2

        
        
        extra_symbols = int(np.ceil(filter_delay_samples / float(self.samples_per_bit)))

        if extra_symbols > 0:
            
            pad = np.zeros(extra_symbols, dtype=np.complex64)
            symbols = np.concatenate([symbols, pad])

        
        self.src = blocks.vector_source_c(symbols.tolist(), False)

        
        self.repeat = blocks.repeat(gr.sizeof_gr_complex, int(self.samples_per_bit))

        
        self.lpf = filter.fir_filter_ccf(1, lpf_taps)

        self.carrier = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, frequency, 1, 0)

        self.modulated = blocks.multiply_vcc(1)

        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amp)
        self.adder = blocks.add_vcc(1)

        self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, output_file, False)
        self.file_sink.set_unbuffered(True)
        
        self.connect(self.src, self.repeat)
        self.connect(self.repeat, self.lpf)
        self.connect(self.lpf, (self.modulated, 0))
        self.connect(self.carrier, (self.modulated, 1))
        self.connect(self.modulated, (self.adder, 0))
        self.connect(self.noise, (self.adder, 1))
        self.connect(self.adder, self.file_sink)


def main():
    parser = argparse.ArgumentParser(description="Generovanie vzoriek")
    parser.add_argument("--outfile", type=str, default="output_signal.complex",
                        help="Výstupný súbor pre signál")
    parser.add_argument("--bits", type=str, default="random",
                        help="bitová postupnosť")
    parser.add_argument("--bits-outfile", type=str, default="demodulated_bits.txt",
                        help="Tekxtový súbor pre bitovú sekvenciu")
    parser.add_argument("--numbits", type=int, default=32,
                        help="počet bitov")
    parser.add_argument("--samp-rate", type=float, default=1280000,
                        help="Vzorkovacia frekvencia")
    parser.add_argument("--freq", type=float, default=500000,
                        help="Nosná frekvenia")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Amplitúda šumu")
    parser.add_argument("--bits-per-sample", type=int, default=128,
                        help="Počet vzoriek na 1 bit")

    args = parser.parse_args()

    if args.bits == "random":
        bits = [random.randint(0, 1) for _ in range(args.numbits)]
    else:
        bits = [int(b) for b in args.bits.strip()]

    
    try:
        with open(args.bits_outfile, "w") as bf:
            
            bf.write("".join(str(b) for b in bits) + "\n")
        
    except Exception as e:
        print(f"Nepodarilo sa zapísať bity '{args.bits_outfile}': {e}")

    
    

    tb = CreateASK(bits, samp_rate=args.samp_rate,
                   frequency=args.freq,
                   noise_amp=args.noise,
                   output_file=args.outfile,
                   samples_per_bit=args.bits_per_sample)

    tb.start()
    tb.wait()
    


if __name__ == "__main__":
    main()
