"""
Parser for MODEL_INPUT and MODEL_OUTPUT blocks in LLM conversation.
"""

import re
from typing import Optional


def contains_model_input(text: str) -> bool:
    """Check if text contains a MODEL_INPUT block."""
    return "<MODEL_INPUT>" in text and "</MODEL_INPUT>" in text


def contains_model_output(text: str) -> bool:
    """Check if text contains a MODEL_OUTPUT block."""
    return "<MODEL_OUTPUT>" in text and "</MODEL_OUTPUT>" in text


def parse_model_input(text: str) -> Optional[dict]:
    """
    Extract and parse content between <MODEL_INPUT> and </MODEL_INPUT> tags.

    Returns a dict of key-value pairs, or None if parsing fails.
    """
    pattern = r'<MODEL_INPUT>(.*?)</MODEL_INPUT>'
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    content = match.group(1).strip()
    result = {}

    for line in content.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue

        # Split on first colon only
        key, _, value = line.partition(':')
        key = key.strip().lower().replace(' ', '_')
        value = value.strip()

        # Skip empty values
        if not value or value.lower() in ('...', 'none', 'n/a', 'not_provided'):
            continue

        result[key] = value

    return result if result else None


def extract_diabetes_features(parsed: dict) -> dict:
    """Extract the 8 diabetes features from parsed MODEL_INPUT."""
    feature_mapping = {
        'pregnancies': 'pregnancies',
        'glucose': 'glucose',
        'blood_pressure': 'blood_pressure',
        'skin_thickness': 'skin_thickness',
        'insulin': 'insulin',
        'bmi': 'bmi',
        'dpf': 'dpf',
        'age_diabetes': 'age',
        'age': 'age',  # fallback
    }

    features = {}
    for input_key, feature_name in feature_mapping.items():
        if input_key in parsed:
            try:
                features[feature_name] = float(parsed[input_key])
            except (ValueError, TypeError):
                features[feature_name] = 0.0

    # Use general age if age_diabetes not provided
    if 'age' not in features and 'age' in parsed:
        try:
            features['age'] = float(parsed['age'])
        except (ValueError, TypeError):
            features['age'] = 0.0

    return features


def format_model_output(cardiac_risk: str, metabolic_risk: str,
                        motor_risk: str, triage: str,
                        details: dict = None) -> str:
    """
    Format prediction results into a MODEL_OUTPUT block.

    Args:
        cardiac_risk: Heart model risk level and probability
        metabolic_risk: Diabetes model risk level and probability
        motor_risk: Parkinson model risk level and probability
        triage: Overall triage level
        details: Optional dict with additional details

    Returns:
        Formatted MODEL_OUTPUT string
    """
    output = f"""<MODEL_OUTPUT>
cardiac_risk: {cardiac_risk}
metabolic_risk: {metabolic_risk}
motor_risk: {motor_risk}
triage: {triage}
</MODEL_OUTPUT>"""

    return output


def determine_triage(cardiac_risk: str, metabolic_risk: str, motor_risk: str) -> str:
    """
    Determine triage level based on individual risk assessments.

    priority_review: if any risk is high/elevated
    recommended_check: if two or more are moderate/borderline/mild
    routine: otherwise
    """
    high_levels = {'high', 'elevated'}
    moderate_levels = {'moderate', 'borderline', 'mild'}

    risks = [cardiac_risk.lower(), metabolic_risk.lower(), motor_risk.lower()]

    # Check for any high risk
    if any(r in high_levels for r in risks):
        return "priority_review"

    # Check for two or more moderate risks
    moderate_count = sum(1 for r in risks if r in moderate_levels)
    if moderate_count >= 2:
        return "recommended_check"

    return "routine"


def strip_model_tags(text: str) -> str:
    """Remove MODEL_INPUT and MODEL_OUTPUT blocks from text for display."""
    text = re.sub(r'<MODEL_INPUT>.*?</MODEL_INPUT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<MODEL_OUTPUT>.*?</MODEL_OUTPUT>', '', text, flags=re.DOTALL)
    return text.strip()
