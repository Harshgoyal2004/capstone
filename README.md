<![CDATA[<div align="center">

# 🏥 AI Health Screening Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![License: Educational](https://img.shields.io/badge/License-Educational%20%26%20Research-blue)](#license)

**A conversational health screening platform integrating pretrained PyTorch deep learning models with a Google Gemini LLM chat interface for cardiac, metabolic, and motor risk assessment.**

[Getting Started](#-quick-start) · [Model Details](#-ml-model-specifications) · [API Reference](#-api-reference) · [Demo](#-demo--sample-test-files)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [ML Model Specifications](#-ml-model-specifications)
  - [Heart Arrhythmia Model (ECGNet)](#1-heart-arrhythmia-model-ecgnet)
  - [Diabetes Risk Model (DiabetesNet)](#2-diabetes-risk-model-diabetesnet)
  - [Parkinson's Voice Risk Model (ParkinsonNet)](#3-parkinsons-voice-risk-model-parkinsonnet)
- [LLM Integration & Prompt Engineering](#-llm-integration--prompt-engineering)
- [System Flow](#system-flow)
- [Project Structure](#project-structure)
- [Quick Start](#-quick-start)
- [Demo & Sample Test Files](#-demo--sample-test-files)
- [Usage Guide](#usage-guide)
- [API Reference](#-api-reference)
- [Data Collection Protocol](#data-collection-protocol)
- [Risk Assessment & Triage](#risk-assessment--triage)
- [Signal Processing Deep Dive](#-signal-processing-deep-dive)
- [Research Papers](#-research-papers)
- [Future Improvements](#-future-improvements)
- [Limitations & Disclaimer](#limitations--disclaimer)
- [Acknowledgments](#acknowledgments)

---

## Overview

This application provides a preliminary health screening experience through natural conversation. A user chats with an AI assistant powered by Google Gemini, which conducts a structured medical intake interview. Once all required data is collected, three pretrained PyTorch models simultaneously analyze:

| Screening Domain | Model | Input Data |
|:---|:---|:---|
| **💓 Cardiac Risk** | ECGNet (Multi-Scale Attention CNN) | Uploaded ECG waveform (.csv) |
| **🩸 Metabolic Risk** | DiabetesNet (Tabular DNN) | 8 clinical biomarkers collected via conversation |
| **🧠 Motor Risk** | ParkinsonNet (Tabular DNN) | Uploaded voice recording (.wav) |

The LLM then explains the screening results in clear, empathetic language — **without diagnosing or claiming any medical conditions**.

> 💡 **How it works**: Gemini collects data conversationally → the backend intercepts a structured `<MODEL_INPUT>` tag → runs all 3 PyTorch models → feeds a `<MODEL_OUTPUT>` back to Gemini → Gemini explains results to the patient.

---

## ✨ Key Features

- **🗨️ Conversational Medical Intake** — Natural language health data collection via Gemini LLM, asking 2–3 questions at a time
- **🧠 Multi-Model Inference** — Three independent PyTorch neural networks running simultaneously for cardiac, metabolic, and motor assessment
- **📊 Automated Triage** — Rule-based triage classification (routine / recommended check / priority review) based on combined risk levels
- **📁 File-Based Analysis** — Direct upload and inference on ECG waveforms (.csv) and voice recordings (.wav)
- **🔄 LLM Fallback Chain** — Automatic model rotation (gemini-1.5-flash → gemini-1.5-pro → gemini-2.0-flash) on rate limits
- **⚕️ Medical Safety Guardrails** — System prompt enforces screening-only language; never diagnoses, always recommends professional consultation
- **🎨 Polished UI** — Gradient header, risk badges, real-time screening status tracking, and disclaimer overlay

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  Chat UI     │  │  File Uploads    │  │  Results Display     │  │
│  │  (st.chat)   │  │  ECG (.csv)      │  │  Risk badges         │  │
│  │              │  │  Voice (.wav)    │  │  Triage level        │  │
│  └──────┬───────┘  └────────┬─────────┘  └──────────────────────┘  │
│         │                   │                                       │
└─────────┼───────────────────┼───────────────────────────────────────┘
          │  HTTP             │  HTTP POST /upload
          │  POST /chat       │
┌─────────▼───────────────────▼───────────────────────────────────────┐
│                        FASTAPI BACKEND                              │
│                                                                     │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐   │
│  │  Gemini LLM │◄──►│        Orchestration Engine               │   │
│  │  (Chat)     │    │  1. Forward msgs to LLM                  │   │
│  └─────────────┘    │  2. Detect <MODEL_INPUT> in response     │   │
│                     │  3. Parse features                        │   │
│                     │  4. Run 3 PyTorch models                  │   │
│                     │  5. Build <MODEL_OUTPUT>                   │   │
│                     │  6. Send results to LLM for explanation   │   │
│                     └──────┬──────────┬──────────┬──────────────┘   │
│                            │          │          │                   │
│  ┌─────────────────┐ ┌────▼────┐ ┌───▼─────┐ ┌─▼───────────┐      │
│  │  Parser Module  │ │ ECGNet  │ │Diabetes │ │ ParkinsonNet│      │
│  │  MODEL_INPUT/   │ │  Heart  │ │   Net   │ │   Voice     │      │
│  │  MODEL_OUTPUT   │ │  Model  │ │  Model  │ │   Model     │      │
│  └─────────────────┘ └─────────┘ └─────────┘ └─────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Two-Pass LLM Architecture

The backend uses a unique **double-pass LLM pattern**:

| Pass | Purpose | Trigger |
|:----:|---------|---------|
| **1st** | Gemini collects patient data and outputs `<MODEL_INPUT>` | User provides all required information |
| — | Backend intercepts, parses, runs PyTorch inference | `<MODEL_INPUT>` detected in LLM output |
| **2nd** | Gemini receives `<MODEL_OUTPUT>` with real probabilities and explains results | Inference complete |

This ensures the **LLM never fabricates the risk assessments** — it only interprets the actual PyTorch model predictions.

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------| 
| **Frontend** | Streamlit 1.38 | Chat interface, file uploads, real-time results display |
| **Backend** | FastAPI 0.115 + Uvicorn | REST API, orchestration, model inference, file handling |
| **LLM** | Google Gemini (1.5-flash / 1.5-pro / 2.0-flash) | Conversational medical intake & result explanation |
| **ML Framework** | PyTorch ≥ 2.0 | Deep learning model inference (CPU/CUDA) |
| **Signal Processing** | SciPy 1.10+ | ECG bandpass filtering (Butterworth), R-peak detection |
| **Audio Analysis** | librosa 0.10+ / soundfile | F0 estimation (pYIN), RMS, harmonic separation, feature extraction |
| **Data Handling** | pandas, NumPy | CSV parsing, feature normalization, array operations |
| **API Client** | google-generativeai | Gemini API interactions with retry logic |
| **Language** | Python 3.10+ | All components |

---

## 🔬 ML Model Specifications

### 1. Heart Arrhythmia Model (ECGNet)

#### Dataset
- **Source**: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet)
- **Records**: 48 half-hour dual-lead ambulatory ECG recordings from 47 subjects
- **Sampling Rate**: 360 Hz
- **Split**: 28 records for training, 19 records for testing
- **Annotation Standard**: AAMI (Association for the Advancement of Medical Instrumentation)

#### Preprocessing Pipeline
1. **Bandpass filtering**: Butterworth 4th-order, 0.5–40 Hz passband (removes baseline wander + high-freq noise)
2. **R-peak detection**: `scipy.signal.find_peaks` with minimum distance 0.5s and adaptive height threshold (`mean + 0.5 × std`)
3. **Beat segmentation**: R-peak centered windows (**108 samples pre-R + 144 samples post-R = 252 samples per beat**)
4. **AAMI beat classification mapping**:

| AAMI Class | Original Symbols | Description |
|:----------:|:----------------:|-------------|
| **N** | N, L, R, e, j | Normal / Bundle branch block |
| **S** | A, a, J, S | Supraventricular ectopic |
| **V** | V, E | Ventricular ectopic |
| **F** | F | Fusion beat |
| **Q** | /, f, Q | Paced / Unknown |

5. **Class balancing**: WeightedRandomSampler for handling severe class imbalance (~72% N-class dominance)

#### Architecture — Multi-Scale Attention CNN

```
Input: (batch, 1, 252) — single-channel ECG beat
  │
  ├─► MultiScaleBlock 1
  │     ├─► Conv1d(1→32, k=3, pad=1)  ─┐
  │     ├─► Conv1d(1→32, k=5, pad=2)  ─┼─► Concat → BN(96) → ReLU → ChannelAttention(96)
  │     └─► Conv1d(1→32, k=7, pad=3)  ─┘
  │   → MaxPool1d(2)                                                  → (batch, 96, 126)
  │
  ├─► MultiScaleBlock 2
  │     ├─► Conv1d(96→64, k=3, pad=1)  ─┐
  │     ├─► Conv1d(96→64, k=5, pad=2)  ─┼─► Concat → BN(192) → ReLU → ChannelAttention(192)
  │     └─► Conv1d(96→64, k=7, pad=3)  ─┘
  │   → MaxPool1d(2)                                                  → (batch, 192, 63)
  │
  ├─► MultiScaleBlock 3
  │     ├─► Conv1d(192→128, k=3, pad=1) ─┐
  │     ├─► Conv1d(192→128, k=5, pad=2) ─┼─► Concat → BN(384) → ReLU → ChannelAttention(384)
  │     └─► Conv1d(192→128, k=7, pad=3) ─┘
  │                                                                    → (batch, 384, 63)
  ├─► AdaptiveAvgPool1d(1) → squeeze                                  → (batch, 384)
  └─► Linear(384 → 5)                                                 → 5-class logits
```

**ChannelAttention(C)** — Squeeze-and-Excitation style:
```
Input: (batch, C, T)
  → AdaptiveAvgPool1d(1)        → (batch, C, 1) → squeeze → (batch, C)
  → Linear(C → C/8) → ReLU
  → Linear(C/8 → C) → Sigmoid  → channel weights
  → element-wise multiply with input
```

**Design rationale**: The multi-scale convolutions (k=3,5,7) capture different morphological features — k=3 for sharp QRS peaks, k=5 for P/T wave shapes, k=7 for broader waveform context. Channel attention lets the model learn which scales matter most for each beat type.

#### Model Parameters

| Component | Parameters |
|:----------|:----------|
| MultiScaleBlock 1 | 3 Conv1d + BN + Attention = ~10K |
| MultiScaleBlock 2 | 3 Conv1d + BN + Attention = ~56K |
| MultiScaleBlock 3 | 3 Conv1d + BN + Attention = ~222K |
| Classifier (FC) | 384 × 5 + 5 = 1,925 |
| **Total** | **~487K trainable parameters** |

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (weight decay regularization) |
| Learning Rate | 3×10⁻⁴ |
| Batch Size | 256 |
| Epochs | 25 |
| Loss Function | Focal Loss (γ=2) — down-weights easy examples, focuses on hard/rare classes |
| Sampler | WeightedRandomSampler |
| Device | CUDA (T4 GPU) |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Final Train Accuracy** | 98.55% |
| **Test Accuracy** | 60.0% |
| **Weighted F1-Score** | 0.68 |
| **Macro AUROC** | 0.7562 |

**Per-class performance:**

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| N | 0.82 | 0.62 | 0.71 | 31,964 |
| S | 0.06 | 0.06 | 0.06 | 1,777 |
| V | 0.72 | 0.83 | 0.77 | 2,458 |
| F | 0.02 | 0.55 | 0.04 | 390 |
| Q | 0.80 | 0.59 | 0.68 | 7,445 |

> **Note**: The gap between train (98.55%) and test (60.0%) accuracy indicates overfitting. The S and F classes have very low F1 due to severe class imbalance (only 1,777 and 390 samples respectively vs. 31,964 for N). For screening purposes, the model effectively identifies abnormal rhythms (V-class F1 = 0.77) which are the most clinically significant.

#### Inference Output

| Risk Level | Abnormality Probability (1 − P(Normal)) |
|:----------:|:----------------------------------------:|
| **Low** | < 0.35 |
| **Moderate** | 0.35 – 0.69 |
| **High** | ≥ 0.70 |

#### Model File
- **File**: `Heart-model.pt` | **Size**: ~2.08 MB | **Format**: PyTorch `state_dict`

---

### 2. Diabetes Risk Model (DiabetesNet)

#### Dataset
- **Source**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (UCI / NIDDK)
- **Records**: 768 female patients (≥21 years, Pima Indian heritage)
- **Positive Rate**: 34.9% diabetic (268/768)
- **Split**: 80/20 train-test, stratified, `random_state=42`
- **Preprocessing**: StandardScaler normalization on all 8 features

#### Input Features

| # | Feature | Description | Unit | Dataset Mean ± Std |
|:-:|---------|-------------|------|--------------------|
| 1 | Pregnancies | Number of pregnancies | count | 3.85 ± 3.37 |
| 2 | Glucose | Fasting plasma glucose (2hr OGTT) | mg/dL | 120.89 ± 31.97 |
| 3 | Blood Pressure | Diastolic blood pressure | mm Hg | 69.11 ± 19.36 |
| 4 | Skin Thickness | Triceps skin fold thickness | mm | 20.54 ± 15.95 |
| 5 | Insulin | 2-hour serum insulin | μU/mL | 79.80 ± 115.24 |
| 6 | BMI | Body mass index | kg/m² | 31.99 ± 7.88 |
| 7 | DPF | Diabetes pedigree function (genetic score) | — | 0.47 ± 0.33 |
| 8 | Age | Age of patient | years | 33.24 ± 11.76 |

#### Architecture — Tabular Deep Neural Network

```
Input: (batch, 8) — 8 StandardScaler-normalized features
  │
  ├─► Linear(8 → 32) → BatchNorm1d(32) → ReLU → Dropout(0.3)
  ├─► Linear(32 → 16) → ReLU → Dropout(0.2)
  └─► Linear(16 → 1)  → Logit output

Inference: σ(logit) → probability ∈ [0, 1]
```

**Total parameters**: 8×32 + 32 + 32 + 32 + 32×16 + 16 + 16×1 + 1 = **833**

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1×10⁻³ |
| Batch Size | 32 |
| Epochs | 80 |
| Loss Function | BCEWithLogitsLoss (numerically stable binary cross-entropy) |
| Device | CUDA (T4 GPU) |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 73.4% |
| **AUROC** | 0.8170 |

**Per-class performance:**

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| Non-diabetic (0) | 0.78 | 0.83 | 0.80 | 100 |
| Diabetic (1) | 0.64 | 0.56 | 0.59 | 54 |

> **Note**: AUROC of 0.817 indicates good discriminative ability. Performance is competitive with literature results on this dataset (typically 74–79% accuracy).

#### Inference Output

| Risk Level | Probability |
|:----------:|:-----------:|
| **Low** | < 0.35 |
| **Borderline** | 0.35 – 0.69 |
| **Elevated** | ≥ 0.70 |

#### Model File
- **File**: `Diabetes-model.pt` | **Size**: ~8.5 KB | **Format**: PyTorch `state_dict`

---

### 3. Parkinson's Voice Risk Model (ParkinsonNet)

#### Dataset
- **Source**: [UCI Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) (University of Oxford)
- **Records**: 195 voice recordings from 31 subjects (23 PD, 8 healthy)
- **Positive Rate**: 75.4% Parkinson's (147/195)
- **Split**: 80/20 train-test, stratified, `random_state=42`
- **Preprocessing**: StandardScaler normalization on all 22 features

#### Input Features (22 Voice Biomarkers)

| Group | Features | Description |
|:------|:---------|:------------|
| **Fundamental Frequency** | Fo(Hz), Fhi(Hz), Flo(Hz) | Average, max, min pitch |
| **Jitter (frequency perturbation)** | Jitter(%), Jitter(Abs), RAP, PPQ, DDP | Cycle-to-cycle pitch variation |
| **Shimmer (amplitude perturbation)** | Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA | Cycle-to-cycle amplitude variation |
| **Noise** | NHR, HNR | Noise-to-harmonics ratio, harmonics-to-noise ratio |
| **Nonlinear Dynamics** | RPDE, DFA, D2 | Recurrence entropy, fluctuation analysis, correlation dimension |
| **Pitch Entropy** | spread1, spread2, PPE | Fundamental frequency variation measures |

#### Architecture — Tabular Deep Neural Network

```
Input: (batch, 22) — 22 StandardScaler-normalized voice features
  │
  ├─► Linear(22 → 64) → BatchNorm1d(64) → ReLU → Dropout(0.4)
  ├─► Linear(64 → 32) → ReLU → Dropout(0.3)
  └─► Linear(32 → 1)  → Logit output

Inference: σ(logit) → probability ∈ [0, 1]
```

**Total parameters**: 22×64 + 64 + 64 + 64 + 64×32 + 32 + 32×1 + 1 = **3,617**

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1×10⁻³ |
| Batch Size | 16 |
| Epochs | 120 |
| Loss Function | BCEWithLogitsLoss |
| Device | CUDA (T4 GPU) |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.4% |
| **AUROC** | 0.9931 |

**Per-class performance:**

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| Healthy (0) | 0.91 | 1.00 | 0.95 | 10 |
| Parkinson's (1) | 1.00 | 0.97 | 0.98 | 29 |

> **Note**: Outstanding performance but evaluated on only 39 test samples. The small test set means confidence intervals are wide. Results should be validated on larger, independent datasets before any clinical consideration.

#### Audio Feature Extraction Pipeline

For uploaded `.wav` files, **librosa** extracts all 22 UCI-compatible features in real-time:

```
WAV file (any sample rate)
  │
  ├─► librosa.load(sr=22050)           — resample to standard rate
  │
  ├─► librosa.pyin(fmin=50, fmax=600)  — fundamental frequency (F0) estimation
  │     └─► Fo(Hz), Fhi(Hz), Flo(Hz)
  │
  ├─► Period analysis (1/F0)            — pitch perturbation
  │     └─► Jitter(%), Jitter(Abs), RAP (3-pt), PPQ (5-pt), DDP
  │
  ├─► librosa.feature.rms()            — amplitude envelope
  │     └─► Shimmer, Shimmer(dB), APQ3 (3-pt), APQ5 (5-pt), APQ (11-pt), DDA
  │
  ├─► librosa.effects.harmonic()       — harmonic/noise separation
  │     └─► NHR, HNR
  │
  ├─► Nonlinear dynamics               — complexity measures
  │     ├─► RPDE (recurrence period density entropy)
  │     ├─► DFA (detrended fluctuation analysis)
  │     └─► D2 (correlation dimension approximation)
  │
  └─► Pitch entropy                    — F0 distribution analysis
        └─► spread1, spread2, PPE
```

#### Inference Output

| Risk Level | Probability |
|:----------:|:-----------:|
| **Stable** | < 0.35 |
| **Mild** | 0.35 – 0.69 |
| **High** | ≥ 0.70 |

#### Model File
- **File**: `Parkinson-model.pt` | **Size**: ~20 KB | **Format**: PyTorch `state_dict`

---

## 🤖 LLM Integration & Prompt Engineering

### System Prompt Design

The Gemini LLM operates under a carefully crafted system prompt with strict medical safety guardrails:

| Prompt Section | Purpose |
|:---------------|:--------|
| **Role Definition** | "You are an AI Health Screening Assistant" — establishes screening-only context |
| **Data Collection Protocol** | Asked to collect in small groups (2–3 questions) with explanations of WHY each value is needed |
| **MODEL_INPUT Format** | Exactly specifies the structured output format the backend parser expects |
| **Result Explanation** | Rules for presenting MODEL_OUTPUT: empathetic, clear, non-diagnostic language |
| **Safety Rules** | Never diagnose, never claim certainty, always recommend professional consultation |

### Model Fallback Chain

The system automatically cycles through 3 Gemini models to handle API rate limits:

```
gemini-1.5-flash (fastest, lowest cost)
    │ 429 error?
    ▼
gemini-1.5-pro (highest quality)
    │ 429 error?
    ▼
gemini-2.0-flash (latest generation)
    │ 429 error?
    ▼
Graceful error message to user
```

Each retry includes a 2-second delay. Non-rate-limit errors are returned immediately.

---

## System Flow

```
User opens Streamlit UI (localhost:8501)
          │
          ▼
Assistant greets user, begins medical intake
          │
          ▼ (conversational loop — 2-3 questions per turn)
User provides: age, gender, symptoms,
  diabetes biomarkers, uploads ECG/voice files
          │
          ▼
LLM outputs <MODEL_INPUT> block with all collected values
          │
          ▼
Backend parses MODEL_INPUT → runs 3 PyTorch models:
  ├── ECGNet      → cardiac_risk   (low / moderate / high)
  ├── DiabetesNet → metabolic_risk (low / borderline / elevated)
  └── ParkinsonNet → motor_risk    (stable / mild / high)
          │
          ▼
Backend computes triage level + builds <MODEL_OUTPUT>
          │
          ▼
LLM receives <MODEL_OUTPUT> → explains findings empathetically
          │
          ▼
User sees screening report with actionable recommendations
```

---

## Project Structure

```
capstone/
├── app.py                     # FastAPI backend — orchestration, endpoints, inference pipeline
├── requirements.txt           # Python dependencies (14 packages)
├── .env                       # API keys (GEMINI_API_KEY) — NOT committed to git
├── .gitignore                 # Excludes .env, venv, __pycache__, .DS_Store
│
├── Heart-model.pt             # Pretrained ECGNet weights (~2.08 MB, ~487K params)
├── Diabetes-model.pt          # Pretrained DiabetesNet weights (~8.5 KB, 833 params)
├── Parkinson-model.pt         # Pretrained ParkinsonNet weights (~20 KB, 3,617 params)
├── models.ipynb               # Jupyter notebook — training code for all 3 models
│
├── inference/                 # Model inference modules
│   ├── __init__.py
│   ├── heart.py               # ECGNet architecture + ECG signal processing (238 lines)
│   ├── diabetes.py            # DiabetesNet architecture + feature normalization (123 lines)
│   └── parkinson.py           # ParkinsonNet architecture + WAV feature extraction (431 lines)
│
├── chat/                      # LLM chat modules
│   ├── __init__.py
│   ├── groq_client.py         # Gemini LLM client + system prompt + fallback chain (169 lines)
│   └── parser.py              # MODEL_INPUT/OUTPUT parsing + triage logic (141 lines)
│
├── ui/                        # Frontend
│   └── streamlit_app.py       # Streamlit chat interface + file uploads + results (338 lines)
│
├── sample_ecg.csv             # Synthetic ECG test file (10s, 360Hz, normal sinus rhythm)
├── sample_voice.wav           # Synthetic voice test file (5s, 22050Hz, sustained vowel)
│
├── mit-bih-arrhythmia-database-1.0.0.zip  # MIT-BIH raw database (~73 MB)
├── Research papers/           # 49 reference papers covering ECG, diabetes, and Parkinson's
└── *.pdf                      # Additional individual research papers
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key — [Get one free](https://aistudio.google.com/apikey)

### Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/Harshgoyal2004/capstone.git
cd capstone

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env

# 5. Start the backend (Terminal 1)
uvicorn app:app --reload --port 8000

# 6. Start the frontend (Terminal 2)
source venv/bin/activate
streamlit run ui/streamlit_app.py --server.port 8501

# 7. Open browser
# Navigate to http://localhost:8501
```

**Expected backend output:**
```
============================================================
Health Screening API - Loading Models...
============================================================
[Heart Model] Loaded from Heart-model.pt on cpu
[Diabetes Model] Loaded from Diabetes-model.pt on cpu
[Parkinson Model] Loaded from Parkinson-model.pt on cpu
============================================================
All models loaded. API ready.
============================================================
```

---

## 🧪 Demo & Sample Test Files

The repository includes ready-to-use synthetic test files for a quick demo:

| File | Description | Format |
|:-----|:------------|:-------|
| `sample_ecg.csv` | 10-second synthetic ECG with normal sinus rhythm (72 bpm) | Single-column CSV, 3600 samples at 360 Hz |
| `sample_voice.wav` | 5-second sustained vowel phonation (150 Hz fundamental + harmonics) | 16-bit mono WAV at 22050 Hz |

### Running the Demo

1. Open **http://localhost:8501**
2. **Upload files** in the sidebar:
   - `sample_ecg.csv` → ECG Recording section
   - `sample_voice.wav` → Voice Recording section
3. **Chat** with the assistant, providing info like:
   > *"I'm a 45-year-old male. Glucose 148, blood pressure 85, skin thickness 33, insulin 150, BMI 33.6, DPF 0.627, no pregnancies. I've uploaded both ECG and voice files."*
4. **View results** — all 3 models will run and the assistant will explain findings

### Expected Results with Sample Data

| Model | Synthetic Input | Expected Risk |
|:------|:----------------|:--------------|
| 💓 Heart | Normal sinus rhythm ECG | **Low** |
| 🩸 Diabetes | Glucose 148 + BMI 33.6 | **Borderline** |
| 🧠 Parkinson | Clean synthetic vowel | **Varies** (depends on extracted features) |

---

## Usage Guide

### 1. Start the Conversation
Type a greeting in the chat. The assistant will begin the health screening intake.

### 2. Answer Questions
The assistant will ask for:
- Basic info (age, gender, symptoms)
- Diabetes biomarkers (glucose, blood pressure, BMI, etc.)
- Whether you have an ECG file or voice recording to upload

### 3. Upload Files (Optional)
Use the sidebar to upload:
- **ECG file** (.csv) — single-column or multi-column waveform data at any sample rate
- **Voice file** (.wav, .mp3, .flac, .ogg) — sustained vowel phonation recording

### 4. Receive Results
Once all data is collected, the system will:
1. Run all three PyTorch models simultaneously
2. Generate individual risk assessments with probabilities
3. Compute automated triage level
4. Provide an empathetic explanation with personalized recommendations

---

## 📡 API Reference

### `POST /chat`

Send a conversation and receive the assistant's response with optional model results.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Welcome!"},
    {"role": "user", "content": "I'm 45 years old, glucose 148..."}
  ],
  "ecg_file_path": "/tmp/health_screening_uploads/ecg.csv",
  "voice_file_path": "/tmp/health_screening_uploads/voice.wav"
}
```

**Response (without model results):**
```json
{
  "response": "Thank you for sharing! Can you also tell me your BMI?",
  "model_results": null
}
```

**Response (with model results — when `<MODEL_INPUT>` is detected):**
```json
{
  "response": "Here are your screening results...",
  "model_results": {
    "cardiac_risk": "low",
    "metabolic_risk": "borderline",
    "motor_risk": "stable",
    "triage": "routine",
    "heart": {"probability": 0.1234, "risk_level": "low", "num_beats_analyzed": 8},
    "diabetes": {"probability": 0.5621, "risk_level": "borderline"},
    "parkinson": {"probability": 0.2103, "risk_level": "stable"}
  }
}
```

### `POST /upload`

Upload ECG (.csv) or voice (.wav/.mp3/.flac/.ogg) files.

**Request**: `multipart/form-data` with `file` field

**Response:**
```json
{
  "file_path": "/tmp/health_screening_uploads/ecg.csv",
  "filename": "ecg.csv",
  "file_type": "ecg"
}
```

### `GET /health`

Health check endpoint.

**Response:** `{"status": "healthy", "models_loaded": true}`

---

## Data Collection Protocol

The LLM collects data through natural conversation and outputs a structured block when all information is gathered:

```xml
<MODEL_INPUT>
age: 45
gender: male
symptoms: occasional chest pain, fatigue

pregnancies: 0
glucose: 120
blood_pressure: 80
skin_thickness: 20
insulin: 85
bmi: 28.5
dpf: 0.45
age_diabetes: 45

ecg_file: provided
voice_file: provided
</MODEL_INPUT>
```

The parser (`chat/parser.py`) uses regex to extract the block, splits on newlines, and builds a dict from `key: value` pairs. Missing/unknown values default to `0.0`.

---

## Risk Assessment & Triage

### Individual Risk Levels

| Model | Low Risk | Moderate Risk | High Risk |
|:------|:---------|:-------------|:---------|
| **Heart (ECGNet)** | < 0.35 | 0.35 – 0.69 | ≥ 0.70 |
| **Diabetes (DiabetesNet)** | < 0.35 (low) | 0.35 – 0.69 (borderline) | ≥ 0.70 (elevated) |
| **Parkinson (ParkinsonNet)** | < 0.35 (stable) | 0.35 – 0.69 (mild) | ≥ 0.70 (high) |

### Triage Determination

| Triage Level | Condition | Action |
|:------------:|-----------|--------|
| 🔴 **priority_review** | Any single risk is **high/elevated** | Immediate professional consultation recommended |
| 🟡 **recommended_check** | Two or more risks are **moderate/borderline/mild** | Schedule follow-up appointment |
| 🟢 **routine** | All other cases | Continue regular health monitoring |

### MODEL_OUTPUT Format

```xml
<MODEL_OUTPUT>
cardiac_risk: moderate (prob: 0.4521)
metabolic_risk: low (prob: 0.3376)
motor_risk: stable (prob: 0.1203)
triage: routine
</MODEL_OUTPUT>
```

---

## 🔊 Signal Processing Deep Dive

### ECG Processing Pipeline

| Step | Method | Parameters |
|:-----|:-------|:-----------|
| **Bandpass Filter** | Butterworth (4th order) | Passband: 0.5–40 Hz |
| **R-Peak Detection** | `scipy.signal.find_peaks` | Min distance: 180 samples (0.5s), height: mean + 0.5σ |
| **Beat Segmentation** | Fixed window around R-peak | 108 pre + 144 post = 252 samples |
| **Fallback** | Center-padding | If no peaks detected, uses middle of signal |

### Voice Feature Extraction

| Algorithm | Library | Output Features |
|:----------|:--------|:----------------|
| **pYIN F0 estimation** | librosa | Fo, Fhi, Flo |
| **Period perturbation** | custom NumPy | Jitter(%), Jitter(Abs), RAP, PPQ, DDP |
| **RMS amplitude analysis** | librosa | Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA |
| **Harmonic separation** | librosa | NHR, HNR |
| **Recurrence analysis** | custom NumPy | RPDE (entropy of recurrence periods) |
| **Detrended fluctuation** | custom NumPy | DFA (scaling exponent α) |
| **Correlation dimension** | custom NumPy | D2 (Grassberger-Procaccia algorithm) |
| **Pitch entropy** | custom NumPy | spread1, spread2, PPE |

---

## 📚 Research Papers

The `Research papers/` directory contains **49 reference papers** spanning three disease domains:

| Domain | Count | Key Topics |
|:-------|:------|:-----------|
| **Cardiac / ECG** | ~18 | Arrhythmia detection, transformer models, attention mechanisms, CNN architectures |
| **Diabetes** | ~14 | Feature selection, ensemble methods, deep learning, explainable AI |
| **Parkinson's** | ~17 | Voice analysis, wearable sensors, multimodal diagnosis, deep learning detection |

---

## 🔮 Future Improvements

| Area | Enhancement |
|:-----|:------------|
| **Heart Model** | Address overfitting gap (98% train → 60% test) with stronger regularization, data augmentation, or transfer learning |
| **Heart Model** | Improve S-class and F-class detection with SMOTE or focused sampling |
| **Diabetes Model** | Validate on larger, multi-ethnic datasets (not just Pima Indian heritage) |
| **Parkinson Model** | Validate on larger independent voice datasets; current test set is only 39 samples |
| **Feature Extraction** | Replace approximated RPDE/DFA/D2 with clinical-grade implementations |
| **LLM** | Migrate to Google `google.genai` SDK (current `google.generativeai` is deprecated) |
| **Deployment** | Dockerize backend + frontend for one-command deployment |
| **Security** | Add authentication, rate limiting, and input sanitization |
| **Testing** | Add unit tests for inference modules and integration tests for the API |

---

## Limitations & Disclaimer

> ⚠️ **This is an AI screening tool, NOT a medical device.**

- **Not a diagnosis**: Results are preliminary screening indicators only
- **Not FDA-approved**: This system has not been validated for clinical use
- **Dataset limitations**: Models were trained on specific populations (MIT-BIH for heart, Pima Indians for diabetes, UCI dataset for Parkinson's) and may not generalize to all demographics
- **Small test sets**: The Parkinson's model was evaluated on only 39 samples
- **Class imbalance**: The heart model shows poor performance on rare classes (S-class F1 = 0.06, F-class F1 = 0.04)
- **Feature extraction**: Voice analysis features (RPDE, DFA, D2) are computational approximations, not clinical-grade measurements
- **LLM dependency**: Result explanations depend on Gemini API availability and quality
- **Always consult healthcare professionals** for medical advice, diagnosis, and treatment

---

## License

This project is for **educational and research purposes only**. Not intended for clinical deployment.

---

## Acknowledgments

- **[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)** — PhysioNet (Moody GB, Mark RG)
- **[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)** — UCI / NIDDK
- **[UCI Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)** — University of Oxford (Max Little)
- **[Google Gemini API](https://ai.google.dev)** — Large Language Model
- **[PyTorch](https://pytorch.org)** — Deep Learning Framework
- **[librosa](https://librosa.org)** — Audio Analysis Library
- **[Streamlit](https://streamlit.io)** — Frontend Framework
- **[FastAPI](https://fastapi.tiangolo.com)** — Backend Framework
]]>
