import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import google.generativeai as genai

# Define the model
class BertClassifier(nn.Module):
    def __init__(self, num_vul_classes, num_danger_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.vul_classifier = nn.Linear(self.bert.config.hidden_size, num_vul_classes)  # Multi-label output
        self.danger_classifier = nn.Linear(self.bert.config.hidden_size, num_danger_classes)  # Single-label output

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token output
        vul_logits = self.vul_classifier(cls_output)
        danger_logits = self.danger_classifier(cls_output)
        return vul_logits, danger_logits


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_vul_classes = 8  # Replace with your actual number of vulnerability classes
num_danger_classes = 4  # Replace with your actual number of danger level classes
model = BertClassifier(num_vul_classes, num_danger_classes)
model.to(device)

# Load the saved model state
checkpoint = torch.load("final_bert_classifier.pkl", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Load tokenizer and encoders
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mlb = joblib.load("mlb.pkl")  # Path to saved MultiLabelBinarizer
le_danger = joblib.load("le_danger.pkl")  # Path to saved LabelEncoder

# Load dataset
df = pd.read_json("data_big.json")  # Replace with your dataset path

# Prediction function
def predict_with_details(text, model, tokenizer, dataset, threshold=0.5):
    model.eval()
    tokens = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        vul_logits, danger_logits = model(tokens['input_ids'], tokens['attention_mask'])
        vul_probs = torch.sigmoid(vul_logits).cpu().numpy()[0]
        danger_probs = torch.softmax(danger_logits, dim=1).cpu().numpy()[0]
        vul_labels = [mlb.classes_[i] for i, prob in enumerate(vul_probs) if prob > threshold]
        danger_label = le_danger.classes_[np.argmax(danger_probs)]

    predictions = {
        "vulnerability_type": vul_labels,
        "danger_level": danger_label,
        "description": [],
        "fix_suggestions": []
    }

    for vul in vul_labels:
        match = dataset[dataset['vulnerability_type'].str.contains(vul, regex=False)]
        if not match.empty:
            predictions["description"].append(match.iloc[0]['description'])
            predictions["fix_suggestions"].extend(match.iloc[0]['fix_suggestions'])

    predictions["fix_suggestions"] = list(set(predictions["fix_suggestions"]))
    return predictions




# Configure the Google GenAI API
genai.configure(api_key="AIzaSyDlgqI_8LTujBMP1AC_XlpCaEfXx9Sxurg")

def get_vulnerable_code_snippets(geminput, genai_model_name="gemini-1.5-flash"):
    """
    Use the Gemini API to identify and return all code snippets causing vulnerabilities.
    
    Args:
        gemapi_code (str): The code to analyze for vulnerabilities.
        genai_model_name (str): The Gemini model name to use for analysis.
        
    Returns:
        list: A list of code snippets causing vulnerabilities, or an error message.
    """
    query = f"""
        You are provided with a JavaScript code snippet. Your task is to identify any vulnerabilities from the following list of known vulnerabilities:

        SQL Injection
        Cross-Site Request Forgery (CSRF)
        Insecure Deserialization
        Improper Error Handling
        Improper Input Validation
        Reflected XSS in URL Parameters
        DOM-Based XSS
        Stored XSS
        
        Please analyze the code and identify any vulnerabilities that match the ones listed above. If a matching vulnerability is found, return the exact code snippet or part of the code that contains the vulnerability. Only the exact code snippet that contains the vulnerability should be returned. If there are multiple vulnerabilities present, return them all separately. If no vulnerabilities are found, return an empty result.

        Here is the JavaScript code snippet:
        {geminput}
    """
    
    try:
        # Create the model instance
        genai_model = genai.GenerativeModel(genai_model_name)
        
        # Generate response
        response = genai_model.generate_content(query)
        
        # Extract and store all snippets in a list
        if response and hasattr(response, "text"):
            vulnerabilities = response.text.strip().split("\n")  # Assuming each snippet is separated by a newline
            return [v.strip() for v in vulnerabilities if v.strip()]  # Clean up whitespace
        else:
            return ["The Gemini API could not provide a valid response. Please try again."]
    except Exception as e:
        return [f"An error occurred while using the Gemini API: {e}"]

# Call the function and store vulnerabilities in a variable
#vulnerabilities = get_vulnerable_code_snippets(javascript_code)


# Streamlit UI
st.title("Vulnerability Detection Dashboard")

# User input
st.header("Enter Code Snippet")
input_text = st.text_area("Paste your code snippet below:")

if st.button("Analyze"):
    if input_text.strip():
        # Get vulnerable code snippets using Gemini API
        st.subheader("Extracting Vulnerable Snippets")
        with st.spinner("Analyzing code for vulnerabilities..."):
            vulnerable_snippets = get_vulnerable_code_snippets(input_text)

        # Check if vulnerable snippets were detected
        if vulnerable_snippets and isinstance(vulnerable_snippets, list):
            st.write(f"**Identified {len(vulnerable_snippets)} vulnerable snippet(s):**")
            for idx, snippet in enumerate(vulnerable_snippets, start=1):
                st.code(snippet, language="javascript")
        else:
            st.error("No vulnerable snippets identified or an error occurred.")
            vulnerable_snippets = []

        # Pass each vulnerable snippet to predict_with_details
        st.subheader("Analyzing Vulnerable Snippets")
        for idx, snippet in enumerate(vulnerable_snippets, start=1):
            st.write(f"### Analysis for Snippet {idx}")
            result = predict_with_details(snippet, model, tokenizer, df, threshold=0.1)

            # Display results
            st.write(f"**Vulnerability Type:** {', '.join(result['vulnerability_type']) if result['vulnerability_type'] else 'None'}")
            st.write(f"**Danger Level:** {result['danger_level']}")

            st.subheader("Descriptions of Vulnerabilities")
            if result["description"]:
                for desc in result["description"]:
                    st.write(f"- {desc}")
            else:
                st.write("No descriptions available.")

            st.subheader("Fix Suggestions")
            if result["fix_suggestions"]:
                for suggestion in result["fix_suggestions"]:
                    st.write(f"- {suggestion}")
            else:
                st.write("No suggestions available.")
    else:
        st.error("Please enter a code snippet to analyze.")
