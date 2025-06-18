

---

# 📄 Resume Matcher & Career Helper

An AI-powered Streamlit web app that helps job seekers improve their resumes by analyzing semantic similarity with their target job title and providing personalized improvement suggestions using Large Language Models (LLMs).

---

## 🚀 Features

* 📤 **Upload Resume (PDF or Image)**
  Supports `.pdf`, `.png`, `.jpg`, and `.jpeg` formats with built-in OCR using `PyMuPDF` and `pytesseract`.

* 🧠 **Semantic Matching**
  Uses `sentence-transformers/all-MiniLM-L6-v2` to compute a **semantic similarity score** between the resume content and the target job title.

* 💡 **AI Suggestions for Improvement**
  Connects to Hugging Face’s Inference API via `huggingface_hub.InferenceClient` (powered by `featherless-ai`) to get actionable feedback using the `mistralai/Magistral-Small-2506` LLM.

* 🔐 **Secure API Integration**
  Authentication via Streamlit secrets (`st.secrets["hf_token"]`), with no hardcoded tokens.

---

## 🧰 Tech Stack

| Component                       | Description                                 |
| ------------------------------- | ------------------------------------------- |
| 🐍 Python                       | Core programming language                   |
| 🔷 Streamlit                    | UI/UX and app deployment                    |
| 🧩 Sentence-Transformers        | Semantic similarity computation             |
| 📚 PyMuPDF + pytesseract        | Resume text extraction from PDFs and images |
| 🤖 Hugging Face InferenceClient | Model querying for smart resume suggestions |

---

## 🌐 Live Demo

> 🔗 https://resume-matcher-programophile.streamlit.app/

---

## 🔧 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/resume-matcher.git
cd resume-matcher

# Create virtual environment & install dependencies
pip install -r requirements.txt

# Set your Hugging Face token securely
echo '[hf_token]' > .streamlit/secrets.toml

# Run the app
streamlit run app.py
```

---

## 🧪 Example Use Case

1. Upload your resume (as a PDF or image).
2. Enter your dream job title (e.g., *Data Scientist*).
3. View your **semantic match score**.
4. Get **instant AI-generated suggestions** to enhance your resume.

---

## 🤝 Contributing

Pull requests are welcome! If you have ideas for improving LLM prompts, better OCR handling, or UI enhancements, feel free to fork the repo and create a PR.

---

## 📄 License

MIT License – see [`LICENSE`](./LICENSE) for details.

