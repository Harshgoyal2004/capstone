"""
Streamlit Chat Interface for Health Screening Application
Provides a conversational UI for interacting with the health screening system.
"""

import os
import requests
import streamlit as st
import json

# ─── Configuration ───

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


# ─── Page Config ───

st.set_page_config(
    page_title="🏥 AI Health Screening Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .header-container h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .header-container p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .sidebar-section h3 {
        margin-top: 0;
        color: #495057;
        font-size: 0.95rem;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .risk-low { background: #d4edda; color: #155724; }
    .risk-moderate { background: #fff3cd; color: #856404; }
    .risk-high { background: #f8d7da; color: #721c24; }
    .risk-na { background: #e2e3e5; color: #383d41; }

    /* Results card */
    .results-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #856404;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───

st.markdown("""
<div class="header-container">
    <h1>🏥 AI Health Screening Assistant</h1>
    <p>Powered by PyTorch ML Models & Groq LLM</p>
</div>
""", unsafe_allow_html=True)


# ─── Session State ───

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ecg_file_path" not in st.session_state:
    st.session_state.ecg_file_path = None

if "voice_file_path" not in st.session_state:
    st.session_state.voice_file_path = None

if "model_results" not in st.session_state:
    st.session_state.model_results = None


# ─── Sidebar ───

with st.sidebar:
    st.markdown("### 📋 File Uploads")
    st.markdown("Upload your medical files for comprehensive screening.")

    # ECG Upload
    st.markdown("---")
    st.markdown("#### 💓 ECG Recording")
    st.caption("Upload a .csv file containing ECG waveform data")
    ecg_file = st.file_uploader(
        "ECG File (.csv)",
        type=["csv"],
        key="ecg_upload",
        label_visibility="collapsed"
    )

    if ecg_file is not None and st.session_state.ecg_file_path is None:
        try:
            files = {"file": (ecg_file.name, ecg_file.getvalue(), "text/csv")}
            resp = requests.post(f"{API_BASE_URL}/upload", files=files)
            if resp.status_code == 200:
                st.session_state.ecg_file_path = resp.json()["file_path"]
                st.success(f"✅ ECG file uploaded: {ecg_file.name}")
            else:
                st.error(f"Upload failed: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to backend. Is the API running?")

    if st.session_state.ecg_file_path:
        st.info(f"📄 ECG file ready")

    # Voice Upload
    st.markdown("---")
    st.markdown("#### 🎤 Voice Recording")
    st.caption("Upload a .wav file of sustained vowel phonation")
    voice_file = st.file_uploader(
        "Voice File (.wav)",
        type=["wav", "mp3", "flac", "ogg"],
        key="voice_upload",
        label_visibility="collapsed"
    )

    if voice_file is not None and st.session_state.voice_file_path is None:
        try:
            files = {"file": (voice_file.name, voice_file.getvalue(), "audio/wav")}
            resp = requests.post(f"{API_BASE_URL}/upload", files=files)
            if resp.status_code == 200:
                st.session_state.voice_file_path = resp.json()["file_path"]
                st.success(f"✅ Voice file uploaded: {voice_file.name}")
            else:
                st.error(f"Upload failed: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to backend. Is the API running?")

    if st.session_state.voice_file_path:
        st.info(f"🎵 Voice file ready")

    # Status
    st.markdown("---")
    st.markdown("### 📊 Screening Status")

    files_status = []
    if st.session_state.ecg_file_path:
        files_status.append("✅ ECG")
    else:
        files_status.append("⬜ ECG")

    if st.session_state.voice_file_path:
        files_status.append("✅ Voice")
    else:
        files_status.append("⬜ Voice")

    for s in files_status:
        st.markdown(s)

    # Model Results
    if st.session_state.model_results:
        st.markdown("---")
        st.markdown("### 🔬 Model Results")
        results = st.session_state.model_results

        # Cardiac
        cardiac = results.get("cardiac_risk", "N/A")
        st.markdown(f"**💓 Cardiac:** {cardiac}")

        # Metabolic
        metabolic = results.get("metabolic_risk", "N/A")
        st.markdown(f"**🩸 Metabolic:** {metabolic}")

        # Motor
        motor = results.get("motor_risk", "N/A")
        st.markdown(f"**🧠 Motor:** {motor}")

        # Triage
        triage = results.get("triage", "N/A")
        triage_colors = {
            "routine": "🟢",
            "recommended_check": "🟡",
            "priority_review": "🔴"
        }
        st.markdown(f"**Triage:** {triage_colors.get(triage, '⚪')} {triage}")

    # Reset button
    st.markdown("---")
    if st.button("🔄 New Screening", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ecg_file_path = None
        st.session_state.voice_file_path = None
        st.session_state.model_results = None
        st.rerun()

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> This is an AI screening tool,
        not a medical device. Results are for informational purposes only.
        Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ─── Chat Display ───

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ─── Chat Input ───

if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "messages": st.session_state.messages,
                    "ecg_file_path": st.session_state.ecg_file_path,
                    "voice_file_path": st.session_state.voice_file_path,
                }

                resp = requests.post(
                    f"{API_BASE_URL}/chat",
                    json=payload,
                    timeout=120
                )

                if resp.status_code == 200:
                    data = resp.json()
                    assistant_response = data["response"]

                    # Store model results if present
                    if data.get("model_results"):
                        st.session_state.model_results = data["model_results"]

                    st.markdown(assistant_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                else:
                    error_msg = f"⚠️ Error from server: {resp.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

            except requests.exceptions.ConnectionError:
                error_msg = "⚠️ Cannot connect to the backend server. Please make sure it's running with:\n\n```\nuvicorn app:app --reload\n```"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except requests.exceptions.Timeout:
                error_msg = "⚠️ Request timed out. The server may be processing. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


# ─── Welcome Message ───

if not st.session_state.messages:
    with st.chat_message("assistant"):
        welcome = "👋 Welcome to the AI Health Screening Assistant! I'm here to help you with a preliminary health screening. I'll ask you some questions about your health, and if you have any medical files (ECG recordings, voice samples), you can upload them using the sidebar.\n\nLet's get started! **What's your name and how are you feeling today?**"
        st.markdown(welcome)
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome
        })
