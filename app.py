import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend

FS = 30
LOW, HIGH = 0.7, 4.0
NAMES = ["Green", "CHROM", "POS"]

st.set_page_config(page_title="Contactless Heart Rate (rPPG)", layout="wide")
st.title("Contactless Heart Rate Monitor (rPPG)")
st.caption("Educational project. Not a medical device and not for diagnosis.")

# ------------------------------------------------------------------
# Section 1: accuracy evaluation from eval_log.csv
# ------------------------------------------------------------------
st.header("1. Accuracy vs reference device")

if os.path.exists("eval_log.csv"):
    df = pd.read_csv("eval_log.csv")
    rows = []
    for n in NAMES:
        err = df[n] - df["reference"]
        rows.append({"Method": n,
                     "MAE (BPM)": err.abs().mean(),
                     "RMSE (BPM)": np.sqrt((err ** 2).mean()),
                     "Bias (BPM)": err.mean()})
    metrics = pd.DataFrame(rows).set_index("Method")

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(metrics.style.format("{:.2f}"))
        st.bar_chart(metrics["MAE (BPM)"])
    with c2:
        fig, ax = plt.subplots(figsize=(5, 5))
        for n in NAMES:
            ax.scatter(df["reference"], df[n], label=n)
        lo, hi = df["reference"].min() - 5, df["reference"].max() + 5
        ax.plot([lo, hi], [lo, hi], "k--", label="Perfect")
        ax.set_xlabel("Reference BPM")
        ax.set_ylabel("Estimated BPM")
        ax.legend()
        st.pyplot(fig)
    st.write(f"Samples: {len(df)}")
    st.dataframe(df)
else:
    st.info("eval_log.csv nahi mili. Pehle step7_live_eval.py chalakar samples lo.")

# ------------------------------------------------------------------
# Section 2: analyze a recorded signal (from step3_signal.py)
# ------------------------------------------------------------------
st.header("2. Analyze a recorded signal")


def green_method(C):
    return detrend(C[:, 1])


def chrom_method(C):
    Cn = C / C.mean(axis=0)
    R, G, B = Cn[:, 0], Cn[:, 1], Cn[:, 2]
    Xs = 3 * R - 2 * G
    Ys = 1.5 * R + G - 1.5 * B
    return Xs - (Xs.std() / (Ys.std() + 1e-8)) * Ys


def pos_method(C, win_sec=1.6):
    l = int(win_sec * FS)
    H = np.zeros(len(C))
    for n in range(l - 1, len(C)):
        m = n - l + 1
        seg = C[m:n + 1]
        Cn = seg / seg.mean(axis=0)
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
        h = S1 + (S1.std() / (S2.std() + 1e-8)) * S2
        H[m:n + 1] += h - h.mean()
    return H


def bpm_from(sig):
    sos = butter(3, [LOW, HIGH], btype="bandpass", fs=FS, output="sos")
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    filt = sosfiltfilt(sos, sig)
    nfft = 2 ** 15
    spec = np.abs(np.fft.rfft(filt * np.hanning(len(filt)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1 / FS)
    mask = (freqs >= LOW) & (freqs <= HIGH)
    return freqs[mask][np.argmax(spec[mask])] * 60, filt, freqs[mask] * 60, spec[mask]


if os.path.exists("rppg_times.npy") and os.path.exists("rppg_rgb.npy"):
    t = np.load("rppg_times.npy")
    rgb = np.load("rppg_rgb.npy")
    t_uni = np.arange(t[0], t[-1], 1 / FS)
    C = np.column_stack([np.interp(t_uni, t, rgb[:, i]) for i in range(3)])

    fns = {"Green": green_method, "CHROM": chrom_method, "POS": pos_method}
    cols = st.columns(3)
    for col, (name, fn) in zip(cols, fns.items()):
        bpm, filt, f, s = bpm_from(fn(C))
        with col:
            st.metric(name, f"{bpm:.1f} BPM")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.plot(f, s, color="red")
            ax.axvline(bpm, linestyle="--", color="black")
            ax.set_xlabel("BPM")
            st.pyplot(fig)
else:
    st.info("rppg_times.npy / rppg_rgb.npy nahi mili. step3_signal.py se recording karo.")