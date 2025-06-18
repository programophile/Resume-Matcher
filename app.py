import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.title("🧠 Resume Matcher")

uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
job_description = st.text_area("Paste the job description here")

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def extract_keywords(text):
    # Lowercase and extract words with 3+ letters
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    # Remove common English stopwords
    keywords = set(word for word in words if word not in ENGLISH_STOP_WORDS)
    return keywords

if uploaded_file is not None and job_description.strip() != "":
    resume_text = extract_text_from_pdf(uploaded_file)

    # Encode texts to embeddings
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(job_description, convert_to_tensor=True)

    # Calculate cosine similarity score
    score = cosine_similarity(
        [resume_embedding.cpu().numpy()],
        [jd_embedding.cpu().numpy()]
    )[0][0]

    st.success(f"✅ Similarity Score: {score:.2f}")

    # Extract and compare keywords
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    common_keywords = resume_keywords.intersection(jd_keywords)

    # Show match explanation
    if common_keywords:
        st.markdown("### 🔍 Match Explanation:")
        st.write(f"The resume and job description share these key terms:")
        st.write(", ".join(sorted(common_keywords)))
    else:
        st.write("No direct keyword overlap found, but semantic similarity still exists!")
else:
    st.info("Please upload a PDF resume and enter a job description to see the match.")
