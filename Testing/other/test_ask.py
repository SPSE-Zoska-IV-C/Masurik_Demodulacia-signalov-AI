import numpy as np
import matplotlib.pyplot as plt

iq = np.fromfile(r"C:\Users\masur\Downloads\Masurik_Demodulacia_signalov_AI-main\Masurik_Demodulacia_signalov_AI-main\test\00005_noiseless.complex", dtype=np.complex64)

plt.plot(np.imag(iq[:10000]))
plt.title("ASK signál (reálna časť)")

plt.savefig("stredny_sum.png", dpi=300)
plt.show()