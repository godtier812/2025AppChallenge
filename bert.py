import streamlit as st
import pytesseract
import cv2
import numpy as np
from transformers import  BertTokenizer, BertForSequenceClassification
from PIL import Image
import platform
import torch
from disease_links import diseases

# Set up Tesseract based on the operating system
if platform.system() == "Darwin":  
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  
else:
    st.error("Unsupported OS for Tesseract. Please configure manually.")



# Load ClinicalBERT model and tokenizer
clinical_bert_model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinical_bert_tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def analyze_with_clinicalBert(extracted_text: str) -> str:
    num_chars, num_words, description, medical_content_found, detected_diseases = analyze_text_and_describe(extracted_text)
    severity_label,disease_label = classify_disease_and_severity(extracted_text)
    if medical_content_found:
        response = f"Detected medical content: {description}. "
        response += f"Severity: {severity_label}. Disease: {disease_label}. "
        if detected_diseases:
            response += "Detected diseases: " + ", ".join(detected_diseases) + ". "
    else:
        response = "No significant medical content detected."    
    return response


# Function to extract text using Tesseract OCR
def extract_text_from_image(image):
    if len(image.shape) == 2:  # If grayscale
        gray_img = image
    elif len(image.shape) == 3:  # Convert to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format. Please provide a valid image.")
    text = pytesseract.image_to_string(gray_img)
    return text

# Function to analyze text for medical relevance
def analyze_text_and_describe(text):
    num_chars = len(text)
    num_words = len(text.split())
    description = "The text contains: "
    
    medical_content_found = False
    detected_diseases = []

    for disease, meaning in diseases.items():
        if disease.lower() in text.lower():
            description += f"{meaning}, "
            medical_content_found = True
            detected_diseases.append(disease)

    description = description.rstrip(", ")
    if description == "The text contains: ":
        description += "uncertain content."
    return num_chars, num_words, description, medical_content_found, detected_diseases

# Function to classify disease and severity using ClinicalBERT
def classify_disease_and_severity(text):
    inputs = clinical_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = clinical_bert_model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    print(f"Bert model response: {predicted_class}")  # Debugging line

    # Modify for more advanced classes if necessary (Assuming binary classification: 0: disease, 1: severity level)
    severity_label = "Mild" if predicted_class == 0 else "Severe"
    
    # For simplicity, use keywords in the text to classify disease
    disease_keywords = [
        "heart", "cancer", "diabetes", "asthma", "arthritis", "stroke", "allergy", "hypertension",
        "dengue", "malaria", "tuberculosis", "bronchitis", "pneumonia", "obesity", "epilepsy",
        "dementia", "autism", "parkinson", "leukemia", "glaucoma", "hepatitis", "kidney",
        "thyroid", "hiv", "aids", "anemia", "migraine", "psoriasis", "eczema", "vitiligo",
        "cholera", "typhoid", "meningitis", "insomnia", "sleep apnea", "fibromyalgia",
        "lupus", "sclerosis", "shingles", "chickenpox", "covid", "corona"
    ]

    detected_labels = []
    lowered_text = text.lower()
    
    for keyword in disease_keywords:
        if keyword in lowered_text:
            formatted_name = keyword.title()
            if keyword == "hiv" or keyword == "aids":
                formatted_name = "HIV/AIDS"
            elif keyword == "parkinson":
                formatted_name = "Parkinson's Disease"
            elif keyword == "covid" or keyword == "corona":
                formatted_name = "COVID-19"
            elif keyword == "sleep apnea":
                formatted_name = "Sleep Apnea"
            elif keyword == "high blood pressure":
                formatted_name = "Hypertension"
            detected_labels.append(formatted_name)

    if not detected_labels:
        detected_labels.append("Unknown")

    return severity_label, list(set(detected_labels))
if __name__ == '__main__':
    print("ClinicalBERT model and tokenizer loaded successfully.")
    