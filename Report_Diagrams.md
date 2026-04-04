# Capstone Project Diagrams

Here are the Mermaid LaTeX compatibility diagrams you can directly embed into your LaTeX report using the `\usepackage{mermaid}` or by pasting the mermaid code into a Mermaid-to-PDF online generator (like Mermaid Live Editor) and using `\includegraphics`.

## 1. System Architecture Pipeline (Flowchart)
This is a detailed layout of the backend processes and model routing you described in Chapter 4.

```mermaid
graph TD
    %% Styling
    classDef frontend fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px,color:#000
    classDef backend fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#000
    classDef ai fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000
    classDef model fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#000

    %% Nodes
    User(("👨‍⚕️ User Patient/Doctor")):::frontend
    UI["💻 Streamlit UI\n(Frontend)"]:::frontend
    API["⚙️ FastAPI Backend\n(Orchestrator)"]:::backend
    Gemini1{"🧠 Gemini LLM\n(Pass 1: Intake)"}:::ai
    Gemini2{"🗣️ Gemini LLM\n(Pass 2: Explain)"}:::ai
    Parser["🛠️ Input Parser\n(< MODEL_INPUT >)"]:::backend
    
    ECG_Model["❤️ ECGNet\nPyTorch Multi-Scale CNN"]:::model
    DIA_Model["🩸 DiabetesNet\nPyTorch Tabular NN"]:::model
    PAR_Model["🧠 ParkinsonNet\nPyTorch Tabular NN"]:::model

    Triage["🗂️ Triage Logic\n(Rule-based Classifier)"]:::backend

    %% Flow
    User -- Chat / Upload Files --> UI
    UI -- POST /chat --> API
    API -- Prompt + Context --> Gemini1
    Gemini1 -- Returns XML Block --> Parser
    
    Parser -- Extracts 252-sample beat --> ECG_Model
    Parser -- Extracts 8 features --> DIA_Model
    Parser -- Extracts 22 voice features --> PAR_Model
    
    ECG_Model -- Cardiac Risk --> Triage
    DIA_Model -- Metabolic Risk --> Triage
    PAR_Model -- Motor Risk --> Triage
    
    Triage -- < MODEL_OUTPUT > --> Gemini2
    Gemini2 -- Empathetic Explanation --> API
    API -- Triage + Report --> UI
    UI -- Final Screening Result --> User
```

## 2. Sequence Diagram (Chat Endpoint Execution Flow)
Tracks the exact lifecycle of the FastAPI `/chat` endpoint logic.

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit App
    participant FA as FastAPI Back-End
    participant LLM as Gemini Generative AI
    participant PY as PyTorch Inference Engines

    U->>UI: Uploads ECG CSV & Voice WAV
    U->>UI: Submits Symptoms text
    UI->>FA: POST /chat (Messages + FilePaths)
    
    rect rgb(240, 240, 240)
        Note right of FA: Pass 1 Pipeline
        FA->>LLM: Send Conversation + Upload Status
        LLM-->>FA: Returns text with <MODEL_INPUT>
    end

    FA->>FA: parse_model_input()

    rect rgb(240, 255, 240)
        Note right of FA: Inference Pipeline
        FA->>PY: Diabetes Extractor (8 features)
        PY-->>FA: Metabolic Risk Score
        FA->>PY: Heart Feature Extractor (R-Peak 252 window)
        PY-->>FA: Cardiac Risk Score
        FA->>PY: Parkinson Extractor (librosa 22 features)
        PY-->>FA: Motor Risk Score
    end

    FA->>FA: determine_triage() -> priority
    FA->>FA: format_model_output()

    rect rgb(255, 245, 235)
        Note right of FA: Pass 2 Pipeline
        FA->>LLM: Send <MODEL_OUTPUT> + Ask for patient explanation
        LLM-->>FA: Medical safely-phrased explanation
    end

    FA-->>UI: Return Final JSON (Results + Explanation)
    UI-->>U: Renders beautiful Triage Badges & AI Text
```

## 3. Detailed Inference Workflow (Triage Subsystem)
```mermaid
stateDiagram-v2
    [*] --> ParseInput: < MODEL_INPUT > Received

    state ParseInput {
        [*] --> ExtractTabular
        [*] --> ECGSignalProcess: Bandpass Filtering
        [*] --> VoiceSignalProcess: Extract 22 Acoustic Features (librosa)
        
        ECGSignalProcess --> FindRPeaks: scipy.find_peaks
        FindRPeaks --> SliceBeatWindows
        
        ExtractTabular --> NormalizeStats
        VoiceSignalProcess --> NormalizeStats
    }

    ParseInput --> RunPyTorchModels: Batched Normalized Audio/Tabular

    state RunPyTorchModels {
        ECGNet --> CardiacRisk
        DiabetesNet --> MetabolicRisk 
        ParkinsonNet --> MotorRisk
    }

    RunPyTorchModels --> TriageDecision

    state TriageDecision {
        if_high <<choice>>
        if_two_moderate <<choice>>
        
         CardiacRisk --> if_high
         MetabolicRisk --> if_high
         MotorRisk --> if_high

        if_high --> PriorityReview : If any == High / Elevated
        if_high --> if_two_moderate : Else
        
        if_two_moderate --> RecommendedCheck : If >= 2 Moderate / Borderline
        if_two_moderate --> Routine : Else
    }

    TriageDecision --> [*] : Formats < MODEL_OUTPUT >
```
