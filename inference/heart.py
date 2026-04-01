"""
Heart Arrhythmia Model Inference
Loads the pretrained ECGNet (Multi-Scale Attention CNN) and runs inference on ECG data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, find_peaks

# ─── Model Architecture (exact reproduction from training notebook) ───

class ChannelAttention(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv5 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.conv7 = nn.Conv1d(in_ch, out_ch, 7, padding=3)
        self.bn = nn.BatchNorm1d(out_ch * 3)
        self.att = ChannelAttention(out_ch * 3)

    def forward(self, x):
        x = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        x = F.relu(self.bn(x))
        return self.att(x)


class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = MultiScaleBlock(1, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = MultiScaleBlock(96, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = MultiScaleBlock(192, 128)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(384, 5)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ─── Signal Processing ───

SAMPLE_RATE = 360
WL = int(0.3 * SAMPLE_RATE)   # 108
WR = int(0.4 * SAMPLE_RATE)   # 144
BEAT_LEN = WL + WR             # 252

LABEL_MAP = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}


def bandpass(sig, low=0.5, high=40, fs=360, order=4):
    """Apply bandpass filter to ECG signal."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def extract_beats(signal, fs=360):
    """Extract individual heartbeat windows from a continuous ECG signal."""
    # Apply bandpass filter
    filtered = bandpass(signal, fs=fs)

    # Simple R-peak detection using scipy
    distance = int(0.5 * fs)  # minimum 0.5s between peaks
    height = np.mean(filtered) + 0.5 * np.std(filtered)
    peaks, _ = find_peaks(filtered, distance=distance, height=height)

    # If no peaks found, try with relaxed parameters
    if len(peaks) == 0:
        peaks, _ = find_peaks(filtered, distance=distance)

    # If still no peaks, use the middle of the signal
    if len(peaks) == 0:
        peaks = [len(filtered) // 2]

    beats = []
    for r in peaks:
        if r - WL < 0 or r + WR >= len(filtered):
            continue
        beat = filtered[r - WL:r + WR]
        beats.append(beat)

    # If no valid beats extracted, try padding
    if len(beats) == 0 and len(filtered) > 0:
        # Take a segment from center and pad/truncate to BEAT_LEN
        center = len(filtered) // 2
        start = max(0, center - WL)
        end = min(len(filtered), center + WR)
        segment = filtered[start:end]
        if len(segment) < BEAT_LEN:
            segment = np.pad(segment, (0, BEAT_LEN - len(segment)), mode='edge')
        else:
            segment = segment[:BEAT_LEN]
        beats.append(segment)

    return np.array(beats)


def resample_signal(signal, orig_fs, target_fs=360):
    """Resample signal to target sampling frequency."""
    if orig_fs == target_fs:
        return signal
    from scipy.signal import resample
    num_samples = int(len(signal) * target_fs / orig_fs)
    return resample(signal, num_samples)


# ─── Model Loading & Prediction ───

_model = None
_device = None


def _get_model(model_path=None):
    """Load and cache the Heart model."""
    global _model, _device

    if _model is not None:
        return _model, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Heart-model.pt")

    _model = ECGNet()
    state_dict = torch.load(model_path, map_location=_device, weights_only=True)
    _model.load_state_dict(state_dict)
    _model.to(_device)
    _model.eval()

    print(f"[Heart Model] Loaded from {model_path} on {_device}")
    return _model, _device


def predict(ecg_csv_path: str, model_path: str = None) -> dict:
    """
    Run heart arrhythmia prediction on an ECG CSV file.

    Args:
        ecg_csv_path: Path to CSV file containing ECG waveform data.
                      Expected format: single column of amplitude values,
                      or multiple columns (first column used).
        model_path: Optional path to model weights file.

    Returns:
        dict with keys: probability, risk_level, class_distribution, details
    """
    import pandas as pd

    model, device = _get_model(model_path)

    # Read ECG data from CSV
    try:
        df = pd.read_csv(ecg_csv_path, header=None)
        if df.shape[1] > 1:
            # If multiple columns, try to use the first numeric column
            signal = df.iloc[:, 0].values.astype(float)
        else:
            signal = df.iloc[:, 0].values.astype(float)
    except Exception as e:
        return {
            "probability": 0.5,
            "risk_level": "moderate",
            "error": f"Could not read ECG file: {str(e)}"
        }

    # Handle potential sampling rate differences
    # Assume 360 Hz by default (MIT-BIH standard)
    fs = 360

    # Extract heartbeat windows
    beats = extract_beats(signal, fs=fs)

    if len(beats) == 0:
        return {
            "probability": 0.5,
            "risk_level": "moderate",
            "error": "Could not extract heartbeat segments from ECG"
        }

    # Run inference on all beats
    with torch.no_grad():
        beats_tensor = torch.tensor(beats, dtype=torch.float32).unsqueeze(1).to(device)
        logits = model(beats_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Aggregate across all beats
    avg_probs = probs.mean(axis=0)

    # Class distribution
    class_dist = {LABEL_MAP[i]: float(avg_probs[i]) for i in range(5)}

    # Abnormality probability = 1 - P(Normal)
    abnormal_prob = 1.0 - float(avg_probs[0])

    # Risk level
    if abnormal_prob < 0.35:
        risk_level = "low"
    elif abnormal_prob < 0.70:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {
        "probability": round(abnormal_prob, 4),
        "risk_level": risk_level,
        "class_distribution": class_dist,
        "num_beats_analyzed": len(beats),
        "details": f"Analyzed {len(beats)} heartbeat(s). Abnormality probability: {abnormal_prob:.2%}"
    }
