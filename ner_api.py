from fastapi import FastAPI
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer only once at startup
model_name = "d4data/biomedical-ner-all"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def extract_severity_or_value(text, entity_start, entity_end):
    window_start = max(0, entity_start - 10)
    window_end = min(len(text), entity_end + 10)
    context = text[window_start:window_end].lower()

    # Check for severity keywords
    for severity in ["mild", "moderate", "severe", "chronic"]:
        if severity in context:
            return severity

    # Regex to find decimal or integer numbers
    numbers = re.findall(r'\b\d+\.?\d*\b', context)
    if numbers:
        return f"value: {numbers[0]}"

    return "unspecified"


@app.post("/ner/")
def run_ner(text: str):
    entities = nlp(text)
    results = []
    for entity in entities:
        word = entity['word']
        start = entity['start']
        end = entity['end']
        sev_or_val = extract_severity_or_value(text, start, end)
        results.append({
            "entity": word,
            "type": entity['entity_group'],
            "severity_or_value": sev_or_val
        })
    return {"entities": results}


# Optional: test locally with `python your_file.py`
if __name__ == "__main__":
    test_text = "The patient's HbA1c is 10.0 and the condition is chronic"
    print(f"Running NER on: {test_text}")
    entities = nlp(test_text)
    for entity in entities:
        word = entity['word']
        start = entity['start']
        end = entity['end']
        sev_or_val = extract_severity_or_value(test_text, start, end)
        print(f"Entity: {word}, Type: {entity['entity_group']}, Severity/Value: {sev_or_val}")
