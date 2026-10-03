import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend

FS = 30  # resampling rate (Hz)
LOW, HIGH = 0.7, 4.0  # 42 to 240 BPM

t = np.load("rppg_times.npy")
rgb = np.load("rppg_rgb.npy")  # columns: R, G, B

real_fps = len(t) / (t[-1] - t[0])
print(f"Recorded FPS: {real_fps:.1f}, duration: {t[-1] - t[0]:.1f}s")

# 1. Resample to a uniform rate (webcam FPS is not perfectly steady)
t_uni = np.arange(t[0], t[-1], 1 / FS)
green = np.interp(t_uni, t, rgb[:, 1])

# 2. Remove slow drift, normalize
sig = detrend(green)
sig = (sig - sig.mean()) / sig.std()

# 3. Bandpass filter around heart-rate frequencies
sos = butter(3, [LOW, HIGH], btype="bandpass", fs=FS, output="sos")
filt = sosfiltfilt(sos, sig)

# 4. FFT (zero-padded for finer frequency steps)
nfft = 2 ** 15
spectrum = np.abs(np.fft.rfft(filt * np.hanning(len(filt)), n=nfft))
freqs = np.fft.rfftfreq(nfft, d=1 / FS)

mask = (freqs >= LOW) & (freqs <= HIGH)
peak_freq = freqs[mask][np.argmax(spectrum[mask])]
bpm = peak_freq * 60
print(f"Estimated heart rate: {bpm:.1f} BPM")

# Plots
fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(t_uni, sig, color="gray")
ax[0].set_title("Normalized raw green signal")
ax[1].plot(t_uni, filt, color="green")
ax[1].set_title("Bandpass filtered (0.7 to 4 Hz)")
ax[2].plot(freqs[mask] * 60, spectrum[mask], color="red")
ax[2].axvline(bpm, linestyle="--", color="black")
ax[2].set_title(f"Spectrum, peak at {bpm:.1f} BPM")
ax[2].set_xlabel("BPM")
plt.tight_layout()
plt.show()