import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import requests
from PIL import Image
import pytesseract
import io

# -------------------------------
# CONFIG
# -------------------------------
HF_API_TOKEN = st.secrets["hf_token"]

model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# PDF / Image Text Extraction
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text.strip()

# -------------------------------
# Hugging Face Suggestions
# -------------------------------
def get_hf_resume_suggestions(resume_text, target_job):
    API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    prompt = (
        f"Here is a resume text:\n{resume_text}\n\n"
        f"I want to apply for the job: {target_job}\n"
        f"Please suggest improvements to the resume to better match this job."
    )

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150, "temperature": 0.7},
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        generated_text = result[0].get('generated_text', '')
        return generated_text.strip()
    else:
        return f"Error from Hugging Face API: {response.status_code} - {response.text}"

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("📄 Resume Matcher & Career Helper")

uploaded_file = st.file_uploader("Upload your resume (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])
job_description = st.text_input("Enter your target job title (e.g. Data Scientist)")

if uploaded_file and job_description.strip():
    file_type = uploaded_file.type

    if "pdf" in file_type:
        resume_text = extract_text_from_pdf(uploaded_file)
    elif "image" in file_type:
        resume_text = extract_text_from_image(uploaded_file.read())
    else:
        st.error("Unsupported file format.")
        st.stop()

    if not resume_text:
        st.warning("No text could be extracted from the file.")
        st.stop()

    # Display extracted resume
    with st.expander("📝 Extracted Resume Text"):
        st.write(resume_text)

    # Similarity score
    embedding_resume = model.encode(resume_text, convert_to_tensor=True)
    embedding_job = model.encode(job_description, convert_to_tensor=True)
    score = util.cos_sim(embedding_resume, embedding_job).item()

    st.success(f"🧠 Match Score: {score:.2f} (closer to 1 is better)")

    if score > 0.7:
        st.info("✅ Your resume is a good match for the target job!")
    elif score > 0.4:
        st.info("🛠️ Your resume somewhat matches, but can be improved.")
    else:
        st.info("⚠️ Your resume doesn't match well. Consider revising it.")

    # Suggestions via Hugging Face
    with st.spinner("Getting suggestions..."):
        st.markdown("### 💡 Suggestions to Improve Your Resume")
        suggestions = get_hf_resume_suggestions(resume_text, job_description)
        st.write(suggestions)

else:
    st.info("Please upload your resume and enter a job title to continue.")
