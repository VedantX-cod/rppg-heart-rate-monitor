# Contactless Heart Rate Monitor (rPPG)

Estimates heart rate from a normal webcam by measuring tiny skin colour changes on the face (remote photoplethysmography, rPPG). Compares three signal extraction methods: Green channel, CHROM and POS.

> **Disclaimer:** Educational project only. This is not a medical device and must not be used for diagnosis or treatment decisions.

## How it works
1. Webcam frames are captured with OpenCV
2. MediaPipe Face Mesh locates the face; the forehead and both cheeks are used as regions of interest
3. The mean RGB of these regions is recorded for every frame
4. The signal is resampled to 30 Hz, detrended and bandpass filtered (0.7 to 4 Hz, i.e. 42 to 240 BPM)
5. The FFT peak gives the heart rate
6. Three extraction methods are compared: Green channel, CHROM and POS

## Results
Reference device:  smartwatch / manual pulse count

| Method | MAE (BPM) | RMSE (BPM) |
|--------|-----------|------------|
| Green  | 4.82      | 6.17       |
| CHROM  | 3.96      | 5.28       |
| POS    | 3.41      | 4.67       |

POS gave the lowest error of the three methods in this evaluation.

## Setup
Python 3.11 is recommended. MediaPipe 0.10.21 does not provide wheels for Python 3.13, so use 3.9 to 3.12.

```
py -3.11 -m venv rppg_env
rppg_env\Scripts\activate
pip install -r requirements.txt
```

## Files
| File | Purpose |
|------|---------|
| `_roi.py` | Face and ROI (forehead + cheeks) detection demo |
| `_signal.py` | Record 30 s of RGB signal (saves `rppg_times.npy`, `rppg_rgb.npy`) |
| `_bpm.py` | Filter the recording and estimate BPM with FFT |
| `_live.py` | Live BPM display with pulse waveform (green channel) |
| `_pos.py` | Offline comparison of Green, CHROM and POS |
| `_live_eval.py` | Live BPM with method switching (keys 1/2/3) and MAE/RMSE evaluation against a reference device (saves `eval_log.csv`) |
| `app.py` | Streamlit dashboard for the results |

## Usage
```
python _signal.py        # press 's' to record 30 s, sit still, face the light
python _bpm.py           # BPM from the saved recording
python _live.py          # live BPM, 'q' to quit
python _live_eval.py     # 'r' saves a sample, 'q' quits and asks for reference BPM
streamlit run app.py     # results dashboard
```

## Limitations
- Sensitive to head movement and lighting changes
- Skin tone and camera quality affect accuracy
- Small sample size, results are indicative only
- The reference device is itself not perfectly exact

## Tech
Python, OpenCV, MediaPipe, NumPy, SciPy, Matplotlib, Pandas, Streamlit
