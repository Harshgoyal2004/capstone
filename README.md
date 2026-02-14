# 🏥 AI Health Screening Assistant

> A conversational health screening platform integrating **pretrained PyTorch deep learning models** with a **Google Gemini LLM** chat interface for cardiac, metabolic, and motor risk assessment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [ML Model Specifications](#ml-model-specifications)
  - [Heart Arrhythmia Model (ECGNet)](#1-heart-arrhythmia-model-ecgnet)
  - [Diabetes Risk Model (DiabetesNet)](#2-diabetes-risk-model-diabetesnet)
  - [Parkinson's Voice Risk Model (ParkinsonNet)](#3-parkinsons-voice-risk-model-parkinsonnet)
- [System Flow](#system-flow)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Data Collection Protocol](#data-collection-protocol)
- [Risk Assessment & Triage](#risk-assessment--triage)
- [Limitations & Disclaimer](#limitations--disclaimer)

---

## Overview

This application provides a preliminary health screening experience through natural conversation. A user chats with an AI assistant powered by Google Gemini, which conducts a structured medical intake interview. Once all required data is collected, three pretrained PyTorch models simultaneously analyze:

- **Cardiac risk** — from uploaded ECG waveform data
- **Metabolic risk** — from diabetes screening biomarkers
- **Motor risk** — from uploaded voice recordings

The LLM then explains the screening results in clear, empathetic language — without diagnosing or claiming any medical conditions.

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

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.38 | Chat interface, file uploads, results display |
| **Backend** | FastAPI + Uvicorn | REST API, orchestration, model inference |
| **LLM** | Google Gemini (1.5-flash / 1.5-pro / 2.0-flash) | Conversational medical intake & result explanation |
| **ML Framework** | PyTorch ≥ 2.0 | Deep learning model inference |
| **Signal Processing** | SciPy, librosa | ECG filtering, audio feature extraction |
| **Language** | Python 3.10+ | All components |

---

## ML Model Specifications

### 1. Heart Arrhythmia Model (ECGNet)

#### Dataset
- **Source**: MIT-BIH Arrhythmia Database (PhysioNet)
- **Records**: 48 half-hour dual-lead ambulatory ECG recordings from 47 subjects
- **Sampling Rate**: 360 Hz
- **Split**: 28 records for training, 19 records for testing
- **Annotation Standard**: AAMI (Association for the Advancement of Medical Instrumentation)

#### Preprocessing Pipeline
1. **Bandpass filtering**: Butterworth 4th-order, 0.5–40 Hz passband
2. **Beat segmentation**: R-peak centered windows (108 pre + 144 post = **252 samples**)
3. **AAMI beat classification mapping**:

| AAMI Class | Original Symbols | Description |
|:----------:|:----------------:|-------------|
| **N** | N, L, R, e, j | Normal / Bundle branch block |
| **S** | A, a, J, S | Supraventricular ectopic |
| **V** | V, E | Ventricular ectopic |
| **F** | F | Fusion beat |
| **Q** | /, f, Q | Paced / Unknown |

4. **Class balancing**: WeightedRandomSampler for handling severe class imbalance

#### Architecture — Multi-Scale Attention CNN

```
Input: (batch, 1, 252) — single-channel ECG beat
  │
  ├─► MultiScaleBlock 1
  │     ├─► Conv1d(1→32, k=3)  ─┐
  │     ├─► Conv1d(1→32, k=5)  ─┼─► Concat → BN(96) → ReLU → ChannelAttention(96)
  │     └─► Conv1d(1→32, k=7)  ─┘
  │   → MaxPool1d(2)
  │
  ├─► MultiScaleBlock 2
  │     ├─► Conv1d(96→64, k=3)  ─┐
  │     ├─► Conv1d(96→64, k=5)  ─┼─► Concat → BN(192) → ReLU → ChannelAttention(192)
  │     └─► Conv1d(96→64, k=7)  ─┘
  │   → MaxPool1d(2)
  │
  ├─► MultiScaleBlock 3
  │     ├─► Conv1d(192→128, k=3) ─┐
  │     ├─► Conv1d(192→128, k=5) ─┼─► Concat → BN(384) → ReLU → ChannelAttention(384)
  │     └─► Conv1d(192→128, k=7) ─┘
  │
  ├─► AdaptiveAvgPool1d(1) → squeeze
  └─► Linear(384 → 5)       → 5-class logits

ChannelAttention(C):
  AdaptiveAvgPool1d(1) → Linear(C, C/8) → ReLU → Linear(C/8, C) → Sigmoid → scale
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3×10⁻⁴ |
| Batch Size | 256 |
| Epochs | 25 |
| Loss Function | Focal Loss (γ=2) |
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

#### Inference Output

| Risk Level | Abnormality Probability (1 − P(Normal)) |
|:----------:|:----------------------------------------:|
| **Low** | < 0.35 |
| **Moderate** | 0.35 – 0.69 |
| **High** | ≥ 0.70 |

#### Model File
- **File**: `Heart-model.pt`
- **Size**: ~2.08 MB
- **Format**: PyTorch `state_dict`
- **Parameters**: ~487K trainable parameters

---

### 2. Diabetes Risk Model (DiabetesNet)

#### Dataset
- **Source**: Pima Indians Diabetes Dataset (UCI / NIDDK)
- **URL**: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
- **Records**: 768 female patients (≥21 years, Pima Indian heritage)
- **Split**: 80/20 train-test, stratified, random_state=42
- **Preprocessing**: StandardScaler normalization on all 8 features

#### Input Features

| # | Feature | Description | Unit |
|:-:|---------|-------------|------|
| 1 | Pregnancies | Number of pregnancies | count |
| 2 | Glucose | Fasting plasma glucose (2hr OGTT) | mg/dL |
| 3 | Blood Pressure | Diastolic blood pressure | mm Hg |
| 4 | Skin Thickness | Triceps skin fold thickness | mm |
| 5 | Insulin | 2-hour serum insulin | μU/mL |
| 6 | BMI | Body mass index | kg/m² |
| 7 | DPF | Diabetes pedigree function (genetic score) | — |
| 8 | Age | Age of patient | years |

#### Architecture — Tabular Deep Neural Network

```
Input: (batch, 8) — 8 normalized features
  │
  ├─► Linear(8 → 32) → BatchNorm1d(32) → ReLU → Dropout(0.3)
  ├─► Linear(32 → 16) → ReLU → Dropout(0.2)
  └─► Linear(16 → 1)  → Logit output (BCEWithLogitsLoss)

Inference: Sigmoid(logit) → probability
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1×10⁻³ |
| Batch Size | 32 |
| Epochs | 80 |
| Loss Function | BCEWithLogitsLoss |
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

#### Inference Output

| Risk Level | Probability |
|:----------:|:-----------:|
| **Low** | < 0.35 |
| **Borderline** | 0.35 – 0.69 |
| **Elevated** | ≥ 0.70 |

#### Model File
- **File**: `Diabetes-model.pt`
- **Size**: ~8.5 KB
- **Format**: PyTorch `state_dict`
- **Parameters**: ~833 trainable parameters

---

### 3. Parkinson's Voice Risk Model (ParkinsonNet)

#### Dataset
- **Source**: UCI Parkinson's Disease Dataset
- **URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data`
- **Records**: 195 voice recordings from 31 subjects (23 PD, 8 healthy)
- **Split**: 80/20 train-test, stratified, random_state=42
- **Preprocessing**: StandardScaler normalization on all 22 features

#### Input Features (22 Voice Biomarkers)

| # | Feature | Description |
|:-:|---------|-------------|
| 1 | MDVP:Fo(Hz) | Average fundamental frequency |
| 2 | MDVP:Fhi(Hz) | Maximum fundamental frequency |
| 3 | MDVP:Flo(Hz) | Minimum fundamental frequency |
| 4 | MDVP:Jitter(%) | Frequency perturbation (%) |
| 5 | MDVP:Jitter(Abs) | Absolute jitter (μs) |
| 6 | MDVP:RAP | Relative average perturbation |
| 7 | MDVP:PPQ | 5-point period perturbation quotient |
| 8 | Jitter:DDP | Average absolute difference of differences |
| 9 | MDVP:Shimmer | Amplitude perturbation |
| 10 | MDVP:Shimmer(dB) | Shimmer in dB |
| 11 | Shimmer:APQ3 | 3-point amplitude perturbation quotient |
| 12 | Shimmer:APQ5 | 5-point amplitude perturbation quotient |
| 13 | MDVP:APQ | 11-point amplitude perturbation quotient |
| 14 | Shimmer:DDA | Average absolute differences between amplitudes |
| 15 | NHR | Noise-to-harmonics ratio |
| 16 | HNR | Harmonics-to-noise ratio (dB) |
| 17 | RPDE | Recurrence period density entropy |
| 18 | DFA | Detrended fluctuation analysis exponent |
| 19 | spread1 | Nonlinear fundamental frequency variation |
| 20 | spread2 | Nonlinear fundamental frequency variation |
| 21 | D2 | Correlation dimension |
| 22 | PPE | Pitch period entropy |

#### Architecture — Tabular Deep Neural Network

```
Input: (batch, 22) — 22 normalized voice features
  │
  ├─► Linear(22 → 64) → BatchNorm1d(64) → ReLU → Dropout(0.4)
  ├─► Linear(64 → 32) → ReLU → Dropout(0.3)
  └─► Linear(32 → 1)  → Logit output (BCEWithLogitsLoss)

Inference: Sigmoid(logit) → probability
```

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

#### Audio Feature Extraction Pipeline
For uploaded `.wav` files, **librosa** extracts the 22 UCI-compatible features:
1. **Fundamental frequency** (F0) via `librosa.pyin` → Fo, Fhi, Flo
2. **Jitter measures** from period perturbation analysis → Jitter%, Jitter(Abs), RAP, PPQ, DDP
3. **Shimmer measures** from RMS amplitude analysis → Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA
4. **Noise ratios** via harmonic/percussive separation → NHR, HNR
5. **Nonlinear dynamics** → RPDE, DFA, D2 (correlation dimension)
6. **Pitch entropy** → spread1, spread2, PPE

#### Inference Output

| Risk Level | Probability |
|:----------:|:-----------:|
| **Stable** | < 0.35 |
| **Mild** | 0.35 – 0.69 |
| **High** | ≥ 0.70 |

#### Model File
- **File**: `Parkinson-model.pt`
- **Size**: ~20 KB
- **Format**: PyTorch `state_dict`
- **Parameters**: ~3,617 trainable parameters

---

## System Flow

```
User opens Streamlit UI (localhost:8501)
          │
          ▼
Assistant greets user, begins medical intake
          │
          ▼ (conversational loop)
User provides: age, gender, symptoms,
  diabetes biomarkers, uploads ECG/voice files
          │
          ▼
LLM outputs <MODEL_INPUT> block with all values
          │
          ▼
Backend parses MODEL_INPUT → runs 3 PyTorch models:
  ├── ECGNet      → cardiac_risk
  ├── DiabetesNet → metabolic_risk
  └── ParkinsonNet → motor_risk
          │
          ▼
Backend builds <MODEL_OUTPUT> with risks + triage
          │
          ▼
LLM receives results → explains findings empathetically
          │
          ▼
User sees screening report with recommendations
```

---

## Project Structure

```
capstone/
├── app.py                     # FastAPI backend (orchestration, endpoints)
├── requirements.txt           # Python dependencies
├── .env                       # API keys (GEMINI_API_KEY)
│
├── Heart-model.pt             # Pretrained ECGNet weights (~2 MB)
├── Diabetes-model.pt          # Pretrained DiabetesNet weights (~8.5 KB)
├── Parkinson-model.pt         # Pretrained ParkinsonNet weights (~20 KB)
├── models.ipynb               # Training notebook (all 3 models)
│
├── inference/
│   ├── __init__.py
│   ├── heart.py               # ECGNet architecture + ECG signal processing
│   ├── diabetes.py            # DiabetesNet architecture + feature normalization
│   └── parkinson.py           # ParkinsonNet architecture + WAV feature extraction
│
├── chat/
│   ├── __init__.py
│   ├── groq_client.py         # Gemini LLM client + system prompt
│   └── parser.py              # MODEL_INPUT/OUTPUT parsing + triage logic
│
└── ui/
    └── streamlit_app.py       # Streamlit chat interface + file uploads
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Google Gemini API key ([Get one free](https://aistudio.google.com/apikey))

### Step 1: Clone & Setup Environment

```bash
cd capstone
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 3: Start Backend

```bash
source venv/bin/activate
uvicorn app:app --reload --port 8000
```

You should see:
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

### Step 4: Start Frontend (new terminal)

```bash
source venv/bin/activate
streamlit run ui/streamlit_app.py --server.port 8501
```

### Step 5: Open Browser

Navigate to **http://localhost:8501**

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
- **ECG file** (.csv) — single-column or multi-column waveform data
- **Voice file** (.wav) — sustained vowel phonation recording

### 4. Receive Results
Once all data is collected, the system will:
1. Run all three PyTorch models
2. Generate risk assessments
3. Provide an empathetic explanation with recommendations

---

## API Reference

### POST `/chat`

Send a conversation and receive the assistant's response.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Welcome!"},
    {"role": "user", "content": "I'm 45 years old"}
  ],
  "ecg_file_path": "/path/to/ecg.csv",
  "voice_file_path": "/path/to/voice.wav"
}
```

**Response:**
```json
{
  "response": "Thank you for sharing...",
  "model_results": {
    "cardiac_risk": "low",
    "metabolic_risk": "borderline",
    "motor_risk": "stable",
    "triage": "routine"
  }
}
```

### POST `/upload`

Upload ECG or voice files.

**Request**: Multipart form data with file
**Response:**
```json
{
  "file_path": "/tmp/health_screening_uploads/ecg.csv",
  "filename": "ecg.csv",
  "file_type": "ecg"
}
```

### GET `/health`

Health check endpoint. Returns `{"status": "healthy", "models_loaded": true}`.

---

## Data Collection Protocol

The LLM collects data through natural conversation and outputs a structured block:

```
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

---

## Risk Assessment & Triage

### Individual Risk Levels

| Model | Low | Moderate | High |
|-------|-----|----------|------|
| **Heart (ECGNet)** | < 0.35 | 0.35 – 0.69 | ≥ 0.70 |
| **Diabetes (DiabetesNet)** | < 0.35 (low) | 0.35 – 0.69 (borderline) | ≥ 0.70 (elevated) |
| **Parkinson (ParkinsonNet)** | < 0.35 (stable) | 0.35 – 0.69 (mild) | ≥ 0.70 (high) |

### Triage Determination

| Triage Level | Condition |
|:------------:|-----------|
| 🔴 **priority_review** | Any single risk is **high/elevated** |
| 🟡 **recommended_check** | Two or more risks are **moderate/borderline/mild** |
| 🟢 **routine** | All other cases |

### Model Output Format

```
<MODEL_OUTPUT>
cardiac_risk: moderate (prob: 0.4521)
metabolic_risk: low (prob: 0.3376)
motor_risk: stable (prob: 0.1203)
triage: routine
</MODEL_OUTPUT>
```

---

## Limitations & Disclaimer

> ⚠️ **This is an AI screening tool, NOT a medical device.**

- **Not a diagnosis**: Results are preliminary screening indicators only
- **Not FDA-approved**: This system has not been validated for clinical use
- **Dataset limitations**: Models were trained on specific populations (MIT-BIH for heart, Pima Indians for diabetes, UCI dataset for Parkinson's) and may not generalize to all demographics
- **Small test sets**: The Parkinson's model was evaluated on only 39 samples
- **Class imbalance**: The heart model shows poor performance on rare classes (S, F)
- **Feature extraction**: Voice features are approximations of clinical-grade measurements
- **Always consult healthcare professionals** for medical advice, diagnosis, and treatment

---

## License

This project is for **educational and research purposes only**. Not intended for clinical deployment.

---

## Acknowledgments

- **MIT-BIH Arrhythmia Database** — PhysioNet
- **Pima Indians Diabetes Dataset** — UCI / NIDDK
- **UCI Parkinson's Disease Dataset** — University of Oxford
- **Google Gemini API** — Large Language Model
- **PyTorch** — Deep Learning Framework
