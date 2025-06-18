import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import re

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🧠 Resume Screening Tool (v2)")
st.write("Upload your resume (PDF) and paste job description to see how well you match.")

# Upload resume file
resume_file = st.file_uploader("📄 Upload your Resume (PDF only)", type=["pdf"])


job_description = st.text_area("📋 Paste Job Description")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# Clean text
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

# Analyze
if st.button("🔍 Analyze"):
    if resume_file and job_description:
        resume_text = extract_text_from_pdf(resume_file)
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_description)

        # Encode with BERT
        resume_emb = model.encode(resume_clean, convert_to_tensor=True)
        job_emb = model.encode(job_clean, convert_to_tensor=True)

        similarity = util.cos_sim(resume_emb, job_emb).item()
        score = round(similarity * 100, 2)
        st.success(f"✅ Match Score: {score}%")

        # Keyword-level comparison
        job_keywords = set(job_clean.lower().split())
        resume_words = set(resume_clean.lower().split())

        matched = job_keywords.intersection(resume_words)
        unmatched = job_keywords - resume_words

        st.markdown("### 🟢 Matched Keywords")
        st.write(", ".join(list(matched)[:20]))

        st.markdown("### 🔴 Missing Keywords (Consider adding these)")
        st.write(", ".join(list(unmatched)[:20]))

    else:
        st.warning("Please upload a PDF resume and input job description.")
