import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

filename = r'C:\Users\masur\Downloads\Masurik_Demodulacia_signalov_AI-main\Masurik_Demodulacia_signalov_AI-main\test\00000_noiseless.complex'       
sample_rate = 1_280_000       
samples_per_bit = 128       
data_type = np.complex64   
threshold = 0.5      

iq_data = np.fromfile(filename, dtype=data_type)

envelope = np.abs(iq_data)
envelope = envelope / np.max(envelope)  

def lowpass_filter(signal, cutoff_hz, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


cutoff_hz = sample_rate / (2 * samples_per_bit)
envelope_smooth = lowpass_filter(envelope, cutoff_hz=cutoff_hz, fs=sample_rate)

num_bits = len(envelope_smooth) // samples_per_bit
demodulated_bits = []

for i in range(num_bits):
    bit_chunk = envelope_smooth[i*samples_per_bit : (i+1)*samples_per_bit]
    avg_amplitude = np.mean(bit_chunk)
    demodulated_bits.append(1 if avg_amplitude > threshold else 0)

demodulated_bits = np.array(demodulated_bits, dtype=int)

plot_samples = 2048
t = np.arange(plot_samples)
demod_bits_expanded = np.repeat(demodulated_bits, samples_per_bit)[:plot_samples]

plt.figure(figsize=(12,6))
plt.plot(t, envelope_smooth[:plot_samples], label='Smoothed Envelope')
plt.step(t, demod_bits_expanded, label='Demodulated Bits', linestyle='--')
plt.legend()
plt.title('ASK Demodulation')
plt.xlabel('Sample index')
plt.show()

np.savetxt('demodulated_bits.txt', demodulated_bits, fmt='%d')
print(f"Demodulation complete: {len(demodulated_bits)} bits extracted.")
