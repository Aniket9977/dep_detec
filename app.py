import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import cv2
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)  # Modify num_labels for levels of depression

# Function to preprocess text and get model predictions
def predict_depression_level(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probs = F.softmax(outputs.logits, dim=1)
    
    # Map to depression levels (0: No Depression, 1: Mild Depression, 2: Severe Depression)
    depression_level = torch.argmax(probs, dim=1).item()
    
    return depression_level, probs

# Function to access camera and capture image
def capture_camera_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return frame
    else:
        st.write("Failed to access the camera.")
        cap.release()
        return None

# Streamlit App
def main():
    st.title("Depression Detection App")
    st.write("This app predicts the level of depression based on the text you provide and analyzes your facial expressions.")

    # Text-based depression detection
    user_input = st.text_area("Enter your text here:")
    
    if st.button("Predict Depression Level"):
        if user_input:
            level, probabilities = predict_depression_level(user_input)
            
            # Depression level names
            depression_levels = ["No Depression", "Mild Depression", "Severe Depression"]
            
            # Show the results
            st.write(f"**Predicted Depression Level:** {depression_levels[level]}")
            st.write(f"Confidence scores: {probabilities.numpy()}")
        else:
            st.write("Please enter some text to analyze.")
    
    st.write("## Facial Analysis")
    
    # Camera capture for facial analysis
    if st.button("Access Camera"):
        frame = capture_camera_frame()
        if frame is not None:
            st.image(frame, caption="Captured from Camera", use_column_width=True)
            # Future Work: You can add facial emotion detection here
            
if __name__ == '__main__':
    main()
