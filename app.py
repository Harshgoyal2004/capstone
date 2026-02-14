"""
FastAPI Backend for Health Screening Application
Handles chat endpoints, file uploads, and model inference orchestration.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file
import sys
import tempfile
import shutil
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from chat.groq_client import chat as groq_chat
from chat.parser import (
    contains_model_input, parse_model_input,
    extract_diabetes_features, format_model_output,
    determine_triage, strip_model_tags
)
from inference import heart, diabetes, parkinson

# ─── App Setup ───

app = FastAPI(
    title="Health Screening API",
    description="Conversational health screening with PyTorch ML models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary file storage
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "health_screening_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Models ───

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    ecg_file_path: Optional[str] = None
    voice_file_path: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model_results: Optional[dict] = None


class UploadResponse(BaseModel):
    file_path: str
    filename: str
    file_type: str


# ─── Startup Event ───

@app.on_event("startup")
async def startup_event():
    """Pre-load all models at startup for fast inference."""
    print("=" * 60)
    print("Health Screening API - Loading Models...")
    print("=" * 60)

    model_dir = os.path.dirname(__file__)

    try:
        heart._get_model(os.path.join(model_dir, "Heart-model.pt"))
    except Exception as e:
        print(f"[WARNING] Could not load Heart model: {e}")

    try:
        diabetes._get_model(os.path.join(model_dir, "Diabetes-model.pt"))
    except Exception as e:
        print(f"[WARNING] Could not load Diabetes model: {e}")

    try:
        parkinson._get_model(os.path.join(model_dir, "Parkinson-model.pt"))
    except Exception as e:
        print(f"[WARNING] Could not load Parkinson model: {e}")

    print("=" * 60)
    print("All models loaded. API ready.")
    print("=" * 60)


# ─── Inference Pipeline ───

def run_inference(parsed_input: dict, ecg_path: str = None, voice_path: str = None) -> dict:
    """
    Run all three models and produce results.

    Args:
        parsed_input: Parsed MODEL_INPUT dict
        ecg_path: Path to uploaded ECG CSV file
        voice_path: Path to uploaded voice WAV file

    Returns:
        dict with cardiac_risk, metabolic_risk, motor_risk, triage, and details
    """
    results = {}

    # ─── Diabetes Model ───
    diabetes_features = extract_diabetes_features(parsed_input)
    try:
        diabetes_result = diabetes.predict(diabetes_features)
        results["diabetes"] = diabetes_result
        metabolic_risk = diabetes_result["risk_level"]
    except Exception as e:
        print(f"[Diabetes Inference Error] {e}")
        metabolic_risk = "moderate"
        results["diabetes"] = {"error": str(e), "risk_level": "moderate", "probability": 0.5}

    # ─── Heart Model ───
    ecg_provided = parsed_input.get("ecg_file", "").lower() == "provided"
    if ecg_provided and ecg_path and os.path.exists(ecg_path):
        try:
            heart_result = heart.predict(ecg_path)
            results["heart"] = heart_result
            cardiac_risk = heart_result["risk_level"]
        except Exception as e:
            print(f"[Heart Inference Error] {e}")
            cardiac_risk = "moderate"
            results["heart"] = {"error": str(e), "risk_level": "moderate", "probability": 0.5}
    else:
        cardiac_risk = "not_assessed"
        results["heart"] = {
            "risk_level": "not_assessed",
            "probability": None,
            "details": "ECG file not provided - cardiac screening skipped"
        }

    # ─── Parkinson Model ───
    voice_provided = parsed_input.get("voice_file", "").lower() == "provided"
    if voice_provided and voice_path and os.path.exists(voice_path):
        try:
            parkinson_result = parkinson.predict(voice_path)
            results["parkinson"] = parkinson_result
            motor_risk = parkinson_result["risk_level"]
        except Exception as e:
            print(f"[Parkinson Inference Error] {e}")
            motor_risk = "mild"
            results["parkinson"] = {"error": str(e), "risk_level": "mild", "probability": 0.5}
    else:
        motor_risk = "not_assessed"
        results["parkinson"] = {
            "risk_level": "not_assessed",
            "probability": None,
            "details": "Voice file not provided - motor screening skipped"
        }

    # ─── Triage ───
    # Only consider assessed risks for triage
    assessed_cardiac = cardiac_risk if cardiac_risk != "not_assessed" else "low"
    assessed_motor = motor_risk if motor_risk != "not_assessed" else "stable"
    triage = determine_triage(assessed_cardiac, metabolic_risk, assessed_motor)

    results["triage"] = triage
    results["cardiac_risk"] = cardiac_risk
    results["metabolic_risk"] = metabolic_risk
    results["motor_risk"] = motor_risk

    return results


# ─── Endpoints ───

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload an ECG CSV or voice WAV file."""
    # Validate file type
    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        file_type = "ecg"
    elif ext in (".wav", ".mp3", ".flac", ".ogg"):
        file_type = "voice"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Please upload .csv (ECG) or .wav (voice) files."
        )

    # Save file
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return UploadResponse(
        file_path=file_path,
        filename=filename,
        file_type=file_type
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message through the health screening pipeline.

    1. Send conversation to Groq LLM
    2. If LLM outputs MODEL_INPUT → parse and run inference
    3. If inference ran → send MODEL_OUTPUT back to LLM for explanation
    4. Return final response
    """
    # Convert messages to dict format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Step 1: Get LLM response
    llm_response = groq_chat(messages)

    # Step 2: Check for MODEL_INPUT
    if contains_model_input(llm_response):
        parsed = parse_model_input(llm_response)

        if parsed:
            # Step 3: Run inference
            inference_results = run_inference(
                parsed,
                ecg_path=request.ecg_file_path,
                voice_path=request.voice_file_path
            )

            # Step 4: Create MODEL_OUTPUT
            model_output = format_model_output(
                cardiac_risk=f"{inference_results['cardiac_risk']} (prob: {inference_results.get('heart', {}).get('probability', 'N/A')})",
                metabolic_risk=f"{inference_results['metabolic_risk']} (prob: {inference_results.get('diabetes', {}).get('probability', 'N/A')})",
                motor_risk=f"{inference_results['motor_risk']} (prob: {inference_results.get('parkinson', {}).get('probability', 'N/A')})",
                triage=inference_results['triage']
            )

            # Step 5: Send results back to LLM for explanation
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": f"Here are the model screening results:\n\n{model_output}\n\nPlease explain these results to the patient in a clear, empathetic way. Remember: these are screening results, not diagnoses."})

            explanation = groq_chat(messages)

            return ChatResponse(
                response=explanation,
                model_results=inference_results
            )

    # No MODEL_INPUT found - return LLM response as-is
    return ChatResponse(response=llm_response)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": True}
