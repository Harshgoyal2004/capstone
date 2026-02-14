"""
Google Gemini LLM Client for Health Screening Chat
Manages conversation with Gemini API for medical intake and result explanation.
"""

import os
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

# ─── Client ───

# Models to try in order of preference
MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]

_configured = False


def _configure():
    """Configure the Gemini API key once."""
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set. "
                "Please set it with: export GEMINI_API_KEY=your_key_here"
            )
        genai.configure(api_key=api_key)
        _configured = True


def _make_model(model_name: str):
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


def chat(messages: list) -> str:
    """
    Send conversation to Gemini and return the assistant's response.
    Tries multiple models if quota is exceeded.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Roles: 'user', 'assistant' (mapped to 'model' for Gemini).

    Returns:
        The assistant's response text.
    """
    import time

    _configure()

    # Convert message format: Gemini uses 'user' and 'model' roles
    gemini_history = []
    for msg in messages[:-1]:  # all but the last message go into history
        role = msg["role"]
        if role == "assistant":
            role = "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    last_msg = messages[-1]["content"] if messages else ""
    last_error = None

    for model_name in MODELS:
        try:
            model = _make_model(model_name)
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(last_msg)
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                print(f"[Gemini] {model_name} quota exceeded, trying next model...")
                time.sleep(2)
                continue
            else:
                return f"I apologize, but I'm experiencing a technical issue: {str(e)}. Please try again."

    return f"I apologize, but all models are currently rate-limited. Please wait a minute and try again. Last error: {str(last_error)}"
