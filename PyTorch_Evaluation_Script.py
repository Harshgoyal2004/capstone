import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Helper to plot and save Confusion Matrix
def plot_cm(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# =====================================================================
# 1. DIABETESNET (Tabular NN)
# =====================================================================
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

def run_diabetes_eval():
    print(f"\n{'='*50}\n🔹 MODULE 1: DIABETES (PyTorch)\n{'='*50}")
    # Simulating the Pima Indians Dataset
    np.random.seed(42)
    # 768 samples, 8 features
    X = np.random.randn(768, 8).astype(np.float32)
    # Bias outcome based on some features to make it learnable
    y = ((X[:, 1] * 0.5 + X[:, 5] * 0.3 + np.random.randn(768) * 0.5) > 0).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using SMOTE for balancing training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model = DiabetesNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tr_t = torch.tensor(X_train_res)
    y_tr_t = torch.tensor(y_train_res).unsqueeze(1)
    
    # Train
    for epoch in range(80):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        out_test = model(torch.tensor(X_test))
        probs = torch.sigmoid(out_test).numpy()
        preds = (probs > 0.5).astype(int)
    
    print(f"Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    print(classification_report(y_test, preds, target_names=['Healthy', 'Diabetic']))
    plot_cm(y_test, preds, ['Healthy', 'Diabetic'], "DiabetesNet Confusion Matrix", "diabetes_cm.png")

# =====================================================================
# 2. ECGNET (Multi-Scale Attention CNN)
# =====================================================================
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

def run_ecg_eval():
    print(f"\n{'='*50}\n🔹 MODULE 2: ECGNet (PyTorch)\n{'='*50}")
    # Simulating MIT-BIH extracted beats (batch, 1, 252)
    N_SAMPLES = 1000
    X = np.random.randn(N_SAMPLES, 1, 252).astype(np.float32)
    y = np.random.randint(0, 5, N_SAMPLES) # 5 classes
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = ECGNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    X_tr_t = torch.tensor(X_train)
    y_tr_t = torch.tensor(y_train)
    
    # Train
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        out_test = model(torch.tensor(X_test))
        preds = torch.argmax(out_test, dim=1).numpy()
    
    print(f"Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    classes = ['N', 'S', 'V', 'F', 'Q']
    print(classification_report(y_test, preds, target_names=classes))
    plot_cm(y_test, preds, classes, "ECGNet Confusion Matrix", "ecg_cm.png")

# =====================================================================
# 3. PARKINSONNET (Tabular NN 22 Features)
# =====================================================================
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

def run_parkinson_eval():
    print(f"\n{'='*50}\n🔹 MODULE 3: PARKINSON (PyTorch)\n{'='*50}")
    # Simulating UCI Parkinson's voice dataset (195 samples, 22 features)
    X = np.random.randn(195, 22).astype(np.float32)
    y = (np.random.randn(195) > 0).astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = ParkinsonNet(input_dim=22)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tr_t = torch.tensor(X_train)
    y_tr_t = torch.tensor(y_train).unsqueeze(1)
    
    # Train
    for epoch in range(120):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        out_test = model(torch.tensor(X_test))
        probs = torch.sigmoid(out_test).numpy()
        preds = (probs > 0.5).astype(int)
    
    print(f"Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    print(classification_report(y_test, preds, target_names=['Healthy', 'Parkinson']))
    plot_cm(y_test, preds, ['Healthy', 'Parkinson'], "ParkinsonNet Confusion Matrix", "parkinson_cm.png")

if __name__ == "__main__":
    print("🚀 Starting PyTorch Models Training & Evaluation locally or in Colab 🚀")
    print("NOTE: Replace mock dataset generators with your CSV actual data loaders for perfect real accuracy.")
    run_diabetes_eval()
    run_ecg_eval()
    run_parkinson_eval()
    print("\n✅ All diagrams saved! run complete.")
