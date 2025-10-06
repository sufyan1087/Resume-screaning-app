import streamlit as st
import os
import zipfile
import joblib
import PyPDF2
import re
import io

# -------------------------------
# Define Resume Cleaning Function
# -------------------------------
def clean_resume_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# -------------------------------
# Load Model + Vectorizer + Encoder
# -------------------------------
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf.pkl")
label_encoder = joblib.load("encoder.pkl")

st.title("Resume Screening App")

uploaded_files = st.file_uploader("Upload resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

if st.button("Classify Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        # Create an in-memory ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for uploaded_file in uploaded_files:
                # Extract text from PDF
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # Clean text
                cleaned_text = clean_resume_text(text)

                # Transform & predict
                features = vectorizer.transform([cleaned_text])
                prediction = model.predict(features)
                category = label_encoder.inverse_transform(prediction)[0]

                # Add to category folder inside zip
                folder_name = f"{category}_CVs"
                zipf.writestr(f"{folder_name}/{uploaded_file.name}", uploaded_file.getbuffer())

        # Finish zip and provide download
        zip_buffer.seek(0)
        st.success("Classification complete!")
        st.download_button(
            "Download Classified CVs",
            data=zip_buffer,
            file_name="classified_cvs.zip",
            mime="application/zip"
        )
