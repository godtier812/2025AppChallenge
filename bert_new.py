from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
import platform
import pytesseract

# Set up Tesseract based on the operating system
if platform.system() == "Darwin":  # <- this is macOS!
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  
else:
   print("Unsupported OS for Tesseract. Please configure manually.")


# Load Bio_ClinicalBERT for NER
#model_name = "emilyalsentzer/Bio_ClinicalBERT"
model_name = "d4data/biomedical-ner-all"
#model_name="pritamdeka/BioBERT-NER-diseases"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)  # NER-optimized

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def analyze_with_clinicalBert(ocr_text: str):
    entities = ner_pipeline(ocr_text)

    findings = []
    seen_diseases = set()

    for entity in entities:
        label = entity.get("entity_group", "")
        text = entity.get("word", "").strip()
        print(f"DEBUG: Label: {label}, Text: {text}")
        # Simple filter: only diseases/conditions
        if label.lower() in ["disease", "condition", "disease_disorder"] and text.lower() not in seen_diseases:
            seen_diseases.add(text.lower())

            # Heuristic severity (can replace with ML classifier)
            if any(x in ocr_text.lower() for x in ["critical", "severe", "acute"]):
                severity = "CRITICAL"
            elif any(x in ocr_text.lower() for x in ["moderate", "elevated"]):
                severity = "SEVERE"
            else:
                severity = "MILD"

            findings.append({
                "findings": text,
                "severity": severity,
                "recommendations": [
                    "Follow up with a medical professional.",
                    "Consider diagnostic tests if symptoms persist."
                ],
                "treatment_suggestions": [
                    "Lifestyle changes",
                    "Medication as prescribed"
                ],
                "home_care_guidance": [
                    "Maintain hydration",
                    "Get adequate rest"
                ]
            })

    return findings if findings else [{"findings": "No diseases detected", "severity": "MILD"}]
# Example
text = "The patient is diagnosed with asthma and severe pneumonia."
print(analyze_with_clinicalBert(text))
