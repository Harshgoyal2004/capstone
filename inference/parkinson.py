"""
Parkinson's Voice Risk Model Inference
Loads the pretrained ParkinsonNet (tabular NN) and runs inference on voice features.
Extracts 22 acoustic features from .wav files matching the UCI Parkinson dataset schema.
"""

import os
import numpy as np
import torch
import torch.nn as nn

# ─── Model Architecture (exact reproduction from training notebook) ───

class ParkinsonNet(nn.Module):
    def __init__(self, input_dim=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ─── Feature Names ───
# The 22 features from the UCI Parkinson dataset (excluding 'name' and 'status'):
FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
    'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

# ─── Normalization Constants ───
# Approximate mean and std from the UCI Parkinson dataset
FEATURE_MEANS = np.array([
    154.2287, 197.1049, 116.3246,
    0.00622, 0.00004, 0.00331, 0.00345, 0.00993,
    0.02971, 0.28211, 0.01567, 0.01809,
    0.02412, 0.04700,
    0.02484, 21.8860,
    0.49884, 0.71822,
    -5.6844, 0.22693, 2.38162, 0.20682
])

FEATURE_STDS = np.array([
    41.3901, 91.4916, 43.5210,
    0.00484, 0.00004, 0.00263, 0.00275, 0.00789,
    0.01886, 0.17127, 0.01098, 0.01276,
    0.01676, 0.03529,
    0.04009, 4.42521,
    0.10392, 0.05547,
    1.09068, 0.08354, 0.38275, 0.06555
])


# ─── Audio Feature Extraction ───

def extract_voice_features(wav_path: str) -> np.ndarray:
    """
    Extract 22 acoustic features from a .wav file to match the UCI Parkinson dataset schema.

    Uses librosa for fundamental frequency analysis and custom calculations
    for jitter, shimmer, and other voice quality metrics.
    """
    import librosa
    import soundfile as sf

    # Load audio
    y, sr = librosa.load(wav_path, sr=22050)

    # Ensure sufficient length
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)), mode='constant')

    # ─── Fundamental Frequency (F0) Analysis ───
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=600, sr=sr,
        frame_length=2048, hop_length=512
    )
    f0_voiced = f0[~np.isnan(f0)]

    if len(f0_voiced) < 2:
        f0_voiced = np.array([150.0, 155.0])  # fallback

    fo_mean = np.mean(f0_voiced)
    fo_max = np.max(f0_voiced)
    fo_min = np.min(f0_voiced)

    # ─── Jitter Measures ───
    # Period perturbation measures
    periods = 1.0 / (f0_voiced + 1e-8)
    period_diffs = np.abs(np.diff(periods))

    jitter_percent = (np.mean(period_diffs) / (np.mean(periods) + 1e-8)) * 100
    jitter_abs = np.mean(period_diffs)

    # RAP (Relative Average Perturbation) - 3-point
    if len(periods) >= 3:
        rap_vals = []
        for i in range(1, len(periods) - 1):
            avg3 = (periods[i-1] + periods[i] + periods[i+1]) / 3
            rap_vals.append(abs(periods[i] - avg3))
        rap = np.mean(rap_vals) / (np.mean(periods) + 1e-8)
    else:
        rap = jitter_percent / 100 * 0.5

    # PPQ (Period Perturbation Quotient) - 5-point
    if len(periods) >= 5:
        ppq_vals = []
        for i in range(2, len(periods) - 2):
            avg5 = np.mean(periods[i-2:i+3])
            ppq_vals.append(abs(periods[i] - avg5))
        ppq = np.mean(ppq_vals) / (np.mean(periods) + 1e-8)
    else:
        ppq = rap * 1.05

    ddp = rap * 3  # DDP = 3 * RAP

    # ─── Shimmer Measures ───
    # Amplitude perturbation
    rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if len(rms_frames) < 2:
        rms_frames = np.array([0.1, 0.11])

    amp_diffs = np.abs(np.diff(rms_frames))
    shimmer = np.mean(amp_diffs) / (np.mean(rms_frames) + 1e-8)
    shimmer_db = 20 * np.log10(1 + shimmer + 1e-8)

    # APQ3
    if len(rms_frames) >= 3:
        apq3_vals = []
        for i in range(1, len(rms_frames) - 1):
            avg3 = (rms_frames[i-1] + rms_frames[i] + rms_frames[i+1]) / 3
            apq3_vals.append(abs(rms_frames[i] - avg3))
        apq3 = np.mean(apq3_vals) / (np.mean(rms_frames) + 1e-8)
    else:
        apq3 = shimmer * 0.5

    # APQ5
    if len(rms_frames) >= 5:
        apq5_vals = []
        for i in range(2, len(rms_frames) - 2):
            avg5 = np.mean(rms_frames[i-2:i+3])
            apq5_vals.append(abs(rms_frames[i] - avg5))
        apq5 = np.mean(apq5_vals) / (np.mean(rms_frames) + 1e-8)
    else:
        apq5 = apq3 * 1.1

    # MDVP:APQ (11-point)
    if len(rms_frames) >= 11:
        apq11_vals = []
        for i in range(5, len(rms_frames) - 5):
            avg11 = np.mean(rms_frames[i-5:i+6])
            apq11_vals.append(abs(rms_frames[i] - avg11))
        apq = np.mean(apq11_vals) / (np.mean(rms_frames) + 1e-8)
    else:
        apq = apq5 * 1.3

    dda = apq3 * 3  # DDA = 3 * APQ3

    # ─── Noise Measures ───
    # NHR (Noise-to-Harmonics Ratio)
    harmonic = librosa.effects.harmonic(y)
    noise = y[:len(harmonic)] - harmonic
    nhr = np.mean(noise**2) / (np.mean(harmonic**2) + 1e-8)

    # HNR (Harmonics-to-Noise Ratio)
    hnr = 10 * np.log10(np.mean(harmonic**2) / (np.mean(noise**2) + 1e-8) + 1e-8)

    # ─── Nonlinear Dynamical Complexity ───
    # RPDE (Recurrence Period Density Entropy) - approximation
    rpde = _compute_rpde(y, sr)

    # DFA (Detrended Fluctuation Analysis) - approximation
    dfa = _compute_dfa(y)

    # ─── Nonlinear Fundamental Frequency Measures ───
    log_f0 = np.log(f0_voiced + 1e-8)
    spread1 = np.max(log_f0) - np.min(log_f0)  # approximation
    spread2 = np.std(log_f0)

    # D2 (Correlation dimension) - approximation
    d2 = _approx_d2(y, sr)

    # PPE (Pitch Period Entropy)
    f0_bins = np.histogram(f0_voiced, bins=20)[0]
    f0_dist = f0_bins / (f0_bins.sum() + 1e-8)
    ppe = -np.sum(f0_dist * np.log2(f0_dist + 1e-8))
    ppe = ppe / (np.log2(20) + 1e-8)  # normalize

    features = np.array([
        fo_mean, fo_max, fo_min,
        jitter_percent, jitter_abs, rap, ppq, ddp,
        shimmer, shimmer_db, apq3, apq5,
        apq, dda,
        nhr, hnr,
        rpde, dfa,
        spread1, spread2, d2, ppe
    ], dtype=np.float64)

    return features


def _compute_rpde(y, sr, m=3, tau=None):
    """Approximate RPDE using entropy of recurrence periods."""
    # Downsample for computation
    step = max(1, len(y) // 2000)
    y_ds = y[::step]

    if tau is None:
        tau = max(1, int(sr * 0.001 / step))

    n = len(y_ds)
    if n < m * tau + 50:
        return 0.5

    # Build embedding
    N = n - (m - 1) * tau
    embedded = np.array([y_ds[i:i + m * tau:tau] for i in range(N)])

    # Find recurrence periods for a subset
    sample_size = min(200, N)
    indices = np.random.choice(N, sample_size, replace=False)
    periods = []

    for idx in indices:
        dists = np.linalg.norm(embedded - embedded[idx], axis=1)
        threshold = np.percentile(dists, 10)
        recurrence = np.where(dists < threshold)[0]
        if len(recurrence) > 1:
            rp = np.diff(recurrence)
            periods.extend(rp.tolist())

    if len(periods) == 0:
        return 0.5

    # Compute entropy of period distribution
    periods = np.array(periods)
    hist, _ = np.histogram(periods, bins=30)
    dist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(dist * np.log2(dist + 1e-8))
    max_entropy = np.log2(30)

    return float(np.clip(entropy / (max_entropy + 1e-8), 0, 1))


def _compute_dfa(y, min_box=4, max_box=None):
    """Approximate DFA (Detrended Fluctuation Analysis)."""
    # Downsample
    step = max(1, len(y) // 3000)
    y_ds = y[::step]
    N = len(y_ds)

    if N < 50:
        return 0.7

    if max_box is None:
        max_box = N // 4

    # Cumulative sum
    y_mean = np.mean(y_ds)
    profile = np.cumsum(y_ds - y_mean)

    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), num=15
    ).astype(int))
    box_sizes = box_sizes[box_sizes >= min_box]

    fluctuations = []
    valid_boxes = []

    for bs in box_sizes:
        n_boxes = N // bs
        if n_boxes < 2:
            continue

        rms_list = []
        for i in range(n_boxes):
            segment = profile[i * bs:(i + 1) * bs]
            x = np.arange(bs)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms_list.append(np.sqrt(np.mean((segment - trend) ** 2)))

        fluctuations.append(np.mean(rms_list))
        valid_boxes.append(bs)

    if len(valid_boxes) < 2:
        return 0.7

    log_n = np.log(valid_boxes)
    log_f = np.log(np.array(fluctuations) + 1e-8)
    alpha = np.polyfit(log_n, log_f, 1)[0]

    return float(np.clip(alpha, 0.3, 1.5))


def _approx_d2(y, sr):
    """Approximate correlation dimension D2."""
    step = max(1, len(y) // 1500)
    y_ds = y[::step]
    N = len(y_ds)

    m = 3
    tau = max(1, int(sr * 0.001 / step))

    if N < m * tau + 100:
        return 2.3

    embed_N = N - (m - 1) * tau
    embedded = np.array([y_ds[i:i + m * tau:tau] for i in range(embed_N)])

    sample_size = min(300, embed_N)
    indices = np.random.choice(embed_N, sample_size, replace=False)
    sampled = embedded[indices]

    dists = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            d = np.linalg.norm(sampled[i] - sampled[j])
            if d > 0:
                dists.append(d)

    if len(dists) < 10:
        return 2.3

    dists = np.array(dists)
    epsilons = np.percentile(dists, np.linspace(10, 90, 10))

    C = []
    for eps in epsilons:
        C.append(np.mean(dists < eps))

    valid = [(e, c) for e, c in zip(epsilons, C) if c > 0 and e > 0]
    if len(valid) < 2:
        return 2.3

    log_e = np.log([v[0] for v in valid])
    log_c = np.log([v[1] for v in valid])
    d2 = np.polyfit(log_e, log_c, 1)[0]

    return float(np.clip(d2, 1.0, 4.0))


# ─── Model Loading & Prediction ───

_model = None
_device = None


def _get_model(model_path=None):
    """Load and cache the Parkinson model."""
    global _model, _device

    if _model is not None:
        return _model, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Parkinson-model.pt")

    _model = ParkinsonNet(input_dim=22)
    state_dict = torch.load(model_path, map_location=_device, weights_only=True)
    _model.load_state_dict(state_dict)
    _model.to(_device)
    _model.eval()

    print(f"[Parkinson Model] Loaded from {model_path} on {_device}")
    return _model, _device


def predict(voice_file_path: str, model_path: str = None) -> dict:
    """
    Run Parkinson's risk prediction on a voice recording.

    Args:
        voice_file_path: Path to .wav file containing voice recording.
        model_path: Optional path to model weights file.

    Returns:
        dict with keys: probability, risk_level, details
    """
    model, device = _get_model(model_path)

    # Extract acoustic features
    try:
        features = extract_voice_features(voice_file_path)
    except Exception as e:
        return {
            "probability": 0.5,
            "risk_level": "mild",
            "error": f"Could not process voice file: {str(e)}"
        }

    # Apply StandardScaler normalization
    normalized = (features - FEATURE_MEANS) / (FEATURE_STDS + 1e-8)

    # Run inference
    with torch.no_grad():
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
        logit = model(x)
        prob = torch.sigmoid(logit).cpu().item()

    # Risk level
    if prob < 0.35:
        risk_level = "stable"
    elif prob < 0.70:
        risk_level = "mild"
    else:
        risk_level = "high"

    return {
        "probability": round(prob, 4),
        "risk_level": risk_level,
        "details": f"Motor risk probability: {prob:.2%} ({risk_level})"
    }
