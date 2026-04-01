"""
Diabetes Risk Model Inference
Loads the pretrained DiabetesNet (tabular NN) and runs inference on patient features.
"""

import os
import numpy as np
import torch
import torch.nn as nn

# ─── Model Architecture (exact reproduction from training notebook) ───

class DiabetesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# ─── Normalization Constants ───
# Approximate mean and std from the Pima Indians Diabetes Dataset
# Columns: pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
FEATURE_MEANS = np.array([3.8451, 120.8946, 69.1055, 20.5365, 79.7995, 31.9926, 0.4719, 33.2409])
FEATURE_STDS  = np.array([3.3696,  31.9726, 19.3558, 15.9522, 115.244,  7.8842, 0.3314, 11.7602])

FEATURE_NAMES = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'dpf', 'age'
]


# ─── Model Loading & Prediction ───

_model = None
_device = None


def _get_model(model_path=None):
    """Load and cache the Diabetes model."""
    global _model, _device

    if _model is not None:
        return _model, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Diabetes-model.pt")

    _model = DiabetesNet()
    state_dict = torch.load(model_path, map_location=_device, weights_only=True)
    _model.load_state_dict(state_dict)
    _model.to(_device)
    _model.eval()

    print(f"[Diabetes Model] Loaded from {model_path} on {_device}")
    return _model, _device


def predict(features: dict, model_path: str = None) -> dict:
    """
    Run diabetes risk prediction on patient features.

    Args:
        features: dict with keys matching FEATURE_NAMES:
                  pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age
        model_path: Optional path to model weights file.

    Returns:
        dict with keys: probability, risk_level, details
    """
    model, device = _get_model(model_path)

    # Build feature vector in correct order
    raw_values = []
    for fname in FEATURE_NAMES:
        val = features.get(fname, 0.0)
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = 0.0
        raw_values.append(val)

    raw_arr = np.array(raw_values, dtype=np.float32)

    # Apply StandardScaler normalization
    normalized = (raw_arr - FEATURE_MEANS) / (FEATURE_STDS + 1e-8)

    # Run inference
    with torch.no_grad():
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
        logit = model(x)
        prob = torch.sigmoid(logit).cpu().item()

    # Risk level
    if prob < 0.35:
        risk_level = "low"
    elif prob < 0.70:
        risk_level = "borderline"
    else:
        risk_level = "elevated"

    return {
        "probability": round(prob, 4),
        "risk_level": risk_level,
        "input_features": {fname: float(raw_values[i]) for i, fname in enumerate(FEATURE_NAMES)},
        "details": f"Diabetes risk probability: {prob:.2%} ({risk_level})"
    }
