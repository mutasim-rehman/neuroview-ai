"""
Safety prompts and medical disclaimers.

Ensures all responses include appropriate warnings.
"""

from typing import Optional


MEDICAL_DISCLAIMER = """
---
**Medical Disclaimer**: This information is provided for educational purposes only and should not be considered medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns. AI predictions are not diagnoses.
---
"""

SHORT_DISCLAIMER = "Note: This is educational information only. Please consult a healthcare professional for medical advice."

PREDICTION_DISCLAIMER = """
**Important**: This AI prediction is NOT a medical diagnosis. Only qualified healthcare professionals (radiologists, neurologists, oncologists) can diagnose medical conditions. This information is for educational purposes only.

**Recommended Actions**:
1. Discuss these findings with your healthcare provider
2. Seek evaluation from appropriate specialists
3. Do not make medical decisions based solely on AI predictions
"""


def wrap_with_disclaimer(
    content: str,
    disclaimer_type: str = "standard"
) -> str:
    """
    Wrap content with appropriate disclaimer.
    
    Args:
        content: Response content
        disclaimer_type: 'standard', 'short', or 'prediction'
        
    Returns:
        Content with disclaimer
    """
    disclaimers = {
        "standard": MEDICAL_DISCLAIMER,
        "short": f"\n\n{SHORT_DISCLAIMER}",
        "prediction": f"\n\n{PREDICTION_DISCLAIMER}"
    }
    
    disclaimer = disclaimers.get(disclaimer_type, MEDICAL_DISCLAIMER)
    return f"{content}{disclaimer}"


def get_safety_reminder() -> str:
    """Get a safety reminder for the agent."""
    return """
SAFETY REMINDER:
- Never provide specific diagnoses
- Never recommend specific treatments
- Always recommend professional consultation
- Include disclaimers in all medical responses
- Be clear about limitations of AI predictions
"""


def check_response_safety(response: str) -> dict:
    """
    Check if response follows safety guidelines.
    
    Args:
        response: Response to check
        
    Returns:
        Dict with check results
    """
    checks = {
        "has_disclaimer": False,
        "no_diagnosis": True,
        "no_treatment_recommendation": True,
        "recommends_professional": False
    }
    
    response_lower = response.lower()
    
    # Check for disclaimer
    disclaimer_keywords = ["educational", "not medical advice", "consult", "healthcare professional"]
    checks["has_disclaimer"] = any(kw in response_lower for kw in disclaimer_keywords)
    
    # Check for diagnosis language
    diagnosis_phrases = ["you have", "you are diagnosed", "this confirms", "definitely"]
    checks["no_diagnosis"] = not any(phrase in response_lower for phrase in diagnosis_phrases)
    
    # Check for treatment recommendations
    treatment_phrases = ["you should take", "prescribe", "recommended dosage", "start taking"]
    checks["no_treatment_recommendation"] = not any(phrase in response_lower for phrase in treatment_phrases)
    
    # Check for professional referral
    referral_keywords = ["doctor", "physician", "specialist", "healthcare provider", "medical professional"]
    checks["recommends_professional"] = any(kw in response_lower for kw in referral_keywords)
    
    return checks


def get_source_citation_format() -> str:
    """Get format for citing sources."""
    return """
When citing sources, use:
- PubMed: [PMID: 12345678]
- Wikipedia: [Wikipedia: Article Name]
- MedlinePlus: [MedlinePlus: Topic]
- Web: [Source: URL or site name]
"""

