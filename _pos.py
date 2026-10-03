import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend

FS = 30
LOW, HIGH = 0.7, 4.0

t = np.load("rppg_times.npy")
rgb = np.load("rppg_rgb.npy")  # columns: R, G, B

# Resample all channels to a uniform rate
t_uni = np.arange(t[0], t[-1], 1 / FS)
C = np.column_stack([np.interp(t_uni, t, rgb[:, i]) for i in range(3)])  # R, G, B

sos = butter(3, [LOW, HIGH], btype="bandpass", fs=FS, output="sos")


def green_method(C):
    return detrend(C[:, 1])


def chrom_method(C):
    Cn = C / C.mean(axis=0)
    R, G, B = Cn[:, 0], Cn[:, 1], Cn[:, 2]
    Xs = 3 * R - 2 * G
    Ys = 1.5 * R + G - 1.5 * B
    alpha = Xs.std() / (Ys.std() + 1e-8)
    return Xs - alpha * Ys


def pos_method(C, win_sec=1.6):
    l = int(win_sec * FS)
    N = len(C)
    H = np.zeros(N)
    for n in range(l - 1, N):
        m = n - l + 1
        seg = C[m:n + 1]
        Cn = seg / seg.mean(axis=0)
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
        h = S1 + (S1.std() / (S2.std() + 1e-8)) * S2
        H[m:n + 1] += h - h.mean()
    return H


def bpm_from(sig):
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    filt = sosfiltfilt(sos, sig)
    nfft = 2 ** 15
    spec = np.abs(np.fft.rfft(filt * np.hanning(len(filt)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1 / FS)
    mask = (freqs >= LOW) & (freqs <= HIGH)
    peak = freqs[mask][np.argmax(spec[mask])] * 60
    # SNR: energy near peak (+/- 0.1 Hz) vs rest of band
    near = mask & (np.abs(freqs - peak / 60) <= 0.1)
    snr = 10 * np.log10(spec[near].sum() ** 2 / (spec[mask & ~near].sum() ** 2 + 1e-12))
    return peak, snr, filt, freqs[mask] * 60, spec[mask]


methods = {"Green": green_method, "CHROM": chrom_method, "POS": pos_method}
results = {}
print(f"{'Method':8} {'BPM':>7} {'SNR (dB)':>10}")
for name, fn in methods.items():
    bpm, snr, filt, f, s = bpm_from(fn(C))
    results[name] = (filt, f, s, bpm)
    print(f"{name:8} {bpm:7.1f} {snr:10.2f}")

fig, ax = plt.subplots(3, 2, figsize=(12, 8))
for i, (name, (filt, f, s, bpm)) in enumerate(results.items()):
    ax[i, 0].plot(t_uni, filt)
    ax[i, 0].set_title(f"{name}: filtered signal")
    ax[i, 1].plot(f, s, color="red")
    ax[i, 1].axvline(bpm, linestyle="--", color="black")
    ax[i, 1].set_title(f"{name}: spectrum, peak {bpm:.1f} BPM")
ax[2, 1].set_xlabel("BPM")
plt.tight_layout()
plt.show()