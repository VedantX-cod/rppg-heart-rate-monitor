# Contactless Heart Rate Monitor (rPPG)

Estimates heart rate from a normal webcam by measuring tiny skin colour changes on the face (remote photoplethysmography). Compares three methods: Green channel, CHROM and POS.

> **Disclaimer:** Educational project only. This is not a medical device and must not be used for diagnosis or treatment decisions.

## How it works
1. Webcam frames captured with OpenCV
2. MediaPipe Face Mesh locates the face; forehead and both cheeks are used as regions of interest
3. Mean RGB of the regions is recorded for every frame
4. Signal is resampled to 30 Hz, detrended and bandpass filtered (0.7 to 4 Hz, i.e. 42 to 240 BPM)
5. FFT peak gives the heart rate
6. Three extraction methods compared: Green channel, CHROM, POS

## Results
Reference device: `<smartwatch / pulse oximeter / manual count>`  
Samples: `<N>`, conditions: `<resting, after walking, after exercise, slight head movement>`

| Method | MAE (BPM) | RMSE (BPM) |
|--------|-----------|------------|
| Green  | `<fill>`  | `<fill>`   |
| CHROM  | `<fill>`  | `<fill>`   |
| POS    | `<fill>`  | `<fill>`   |

(Fill these from the output of `step7_live_eval.py` or the dashboard.)

## Setup
Python 3.11 is recommended (MediaPipe does not support 3.13).

```
py -3.11 -m venv rppg_env
rppg_env\Scripts\activate
pip install -r requirements.txt
```

## Run
| File | Purpose |
|------|---------|
| `step2_roi.py` | Face and ROI detection demo |
| `step3_signal.py` | Record 30 s of signal |
| `step4_bpm.py` | Filter + FFT on the recording |
| `step5_live.py` | Live BPM (green channel) |
| `step6_pos.py` | Compare Green / CHROM / POS offline |
| `step7_live_eval.py` | Live BPM with method switching and MAE evaluation |
| `app.py` | Streamlit dashboard (`streamlit run app.py`) |

## Limitations
- Sensitive to head movement and lighting changes
- Skin tone and camera quality affect accuracy
- Small sample size, results are indicative only

## Tech
Python, OpenCV, MediaPipe, NumPy, SciPy, Matplotlib, Streamlit