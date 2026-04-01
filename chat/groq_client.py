"""
LLM Client for Health Screening Chat
Manages conversation with Gemini API (primary) and HuggingFace (fallback)
for medical intake and result explanation.
"""

import os
import time
import requests as http_requests
import google.generativeai as genai

# ─── System Prompt ───

SYSTEM_PROMPT = """You are an AI Health Screening Assistant. Your role is to conduct a medical intake interview through friendly, professional conversation.

## YOUR RESPONSIBILITIES:
1. Greet the user warmly and explain that you'll be conducting a health screening
2. Collect ALL required information through natural conversation
3. When all data is collected, output a structured <MODEL_INPUT> block
4. After receiving model results in a <MODEL_OUTPUT> block, explain findings empathetically

## DATA TO COLLECT:

### Basic Information:
- Age (number)
- Gender (male/female/other)
- Current symptoms or health concerns (text description)

### Diabetes Screening Features:
- Number of pregnancies (0 if male or not applicable)
- Glucose level (mg/dL) - fasting blood glucose
- Blood pressure (mm Hg) - diastolic
- Skin thickness (mm) - triceps skin fold thickness
- Insulin level (mu U/ml) - 2-hour serum insulin
- BMI (kg/m²) - body mass index
- Diabetes Pedigree Function (DPF) - a score indicating genetic predisposition (0.0 to 2.5, typical ~0.5)
- Age for diabetes assessment

### Heart Screening:
- Ask if they have an ECG file (.csv waveform) to upload
- Note: The user will upload this through the interface

### Parkinson Screening:
- Ask if they have a voice recording (.wav) to upload
- Note: The user will upload this through the interface

## CONVERSATION GUIDELINES:
- Ask questions in small groups (2-3 at a time), not all at once
- Explain WHY you need each piece of information in simple terms
- If the user doesn't know a value, help them estimate or use a reasonable default
- Be patient and supportive
- Use clear, non-medical language when possible
- If a value seems unusual, gently confirm it

## WHEN ALL DATA IS COLLECTED:
Output the data in this exact format (replace ... with actual values):

<MODEL_INPUT>
age: ...
gender: ...
symptoms: ...

pregnancies: ...
glucose: ...
blood_pressure: ...
skin_thickness: ...
insulin: ...
bmi: ...
dpf: ...
age_diabetes: ...

ecg_file: provided/not_provided
voice_file: provided/not_provided
</MODEL_INPUT>

## AFTER RECEIVING MODEL RESULTS:
When you receive a <MODEL_OUTPUT> block, explain the results to the user:
- Use empathetic, clear language
- DO NOT diagnose - you are a screening tool only
- Explain what each risk level means
- Recommend appropriate follow-up based on triage level
- Remind them these are screening results, not medical diagnoses
- Encourage consulting with healthcare professionals

## IMPORTANT RULES:
- NEVER make medical predictions yourself
- NEVER claim any condition with certainty
- ALWAYS recommend professional medical consultation
- Be honest about the limitations of AI screening
- If the user seems distressed, provide reassurance and recommend speaking with a doctor
"""

# ─── Gemini Client ───

GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
_gemini_configured = False


def _configure_gemini():
    """Configure the Gemini API key once."""
    global _gemini_configured
    if not _gemini_configured:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return False
        genai.configure(api_key=api_key)
        _gemini_configured = True
    return True


def _make_gemini_model(model_name: str):
    """Create a GenerativeModel instance."""
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=2048,
            top_p=0.9,
        ),
    )


def _chat_gemini(messages: list) -> str:
    """Try Gemini models. Returns response text or None if all fail."""
    if not _configure_gemini():
        print("[Gemini] No API key found, skipping Gemini...")
        return None

    gemini_history = []
    for msg in messages[:-1]:
        role = msg["role"]
        if role == "assistant":
            role = "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    last_msg = messages[-1]["content"] if messages else ""

    for model_name in GEMINI_MODELS:
        try:
            model = _make_gemini_model(model_name)
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(last_msg)
            print(f"[Gemini] Response from {model_name}")
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                print(f"[Gemini] {model_name} quota exceeded, trying next...")
                time.sleep(2)
                continue
            else:
                print(f"[Gemini] {model_name} error: {e}")
                continue

    print("[Gemini] All models failed.")
    return None


# ─── HuggingFace Fallback Client ───

HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"


def _chat_huggingface(messages: list) -> str:
    """Fallback: Use HuggingFace Inference API with Mistral."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("[HuggingFace] No HF_TOKEN found, skipping fallback...")
        return None

    # Build messages with system prompt
    hf_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        role = msg["role"]
        if role == "model":
            role = "assistant"
        hf_messages.append({"role": role, "content": msg["content"]})

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL,
        "messages": hf_messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        response = http_requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            print(f"[HuggingFace] Response from {HF_MODEL}")
            return result
        else:
            print(f"[HuggingFace] Error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"[HuggingFace] Request failed: {e}")
        return None


# ─── Main Chat Function ───

def chat(messages: list) -> str:
    """
    Send conversation and return the assistant's response.
    Tries Gemini first, falls back to HuggingFace if Gemini fails.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        The assistant's response text.
    """
    # Primary: Try Gemini
    response = _chat_gemini(messages)
    if response:
        return response

    # Fallback: Try HuggingFace
    print("[Chat] Gemini unavailable, falling back to HuggingFace...")
    response = _chat_huggingface(messages)
    if response:
        return response

    # All providers failed
    return (
        "I apologize, but I'm currently unable to process your request. "
        "Both our primary (Gemini) and backup (HuggingFace) AI services are unavailable. "
        "Please try again in a few minutes."
    )
