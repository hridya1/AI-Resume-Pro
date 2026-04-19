import streamlit as st
import pickle
import re
import nltk
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from docx import Document
import PyPDF2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Pro",
    page_icon="📄",
    layout="wide"
)

# ---------------- NLTK ----------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------- LOAD FILES ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# ---------------- SESSION STORAGE ----------------
if "resumes" not in st.session_state:
    st.session_state.resumes = []

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #0a66c2;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}
.stButton>button {
    background: linear-gradient(90deg,#0a66c2,#00a6ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PREPROCESS ----------------
def preprocess_text(text):
    text = str(text).lower()

    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\s\-]{8,}\d', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words and len(w) > 2
    ]

    return " ".join(words)

# ---------------- FILE EXTRACTION ----------------
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except:
        return ""

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 AI Resume Pro")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📁 Stored Resumes", "🎯 JD Matcher"]
)

# ======================================================
# HOME PAGE
# ======================================================
if page == "🏠 Home":

    st.markdown('<p class="main-title">📄 AI Resume Screening System</p>', unsafe_allow_html=True)
    st.write("Upload multiple resumes in PDF or DOCX format.")

    files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if files:
        for file in files:

            text = ""

            if file.name.endswith(".pdf"):
                text = extract_text_from_pdf(file)

            elif file.name.endswith(".docx"):
                text = extract_text_from_docx(file)

            if text:
                cleaned = preprocess_text(text)

                st.session_state.resumes.append({
                    "name": file.name,
                    "raw_text": text,
                    "cleaned_text": cleaned
                })

        st.success(f"{len(files)} resumes uploaded successfully!")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Dashboard Stats")
    st.write("Stored Resumes:", len(st.session_state.resumes))
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# STORED RESUMES PAGE
# ======================================================
elif page == "📁 Stored Resumes":

    st.markdown('<p class="main-title">📁 Stored Resume Classifier</p>', unsafe_allow_html=True)

    if len(st.session_state.resumes) == 0:
        st.warning("No resumes uploaded yet.")

    else:
        selected = st.selectbox(
            "Select Resume",
            [r["name"] for r in st.session_state.resumes]
        )

        resume = next(r for r in st.session_state.resumes if r["name"] == selected)

        if st.button("🔍 Predict Category"):

            X = vectorizer.transform([resume["cleaned_text"]])

            prediction = model.predict(X)[0]

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success(f"🎯 Predicted Category: {prediction}")
            st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# JD MATCHER PAGE
# ======================================================
elif page == "🎯 JD Matcher":

    st.markdown('<p class="main-title">🎯 Job Description Matcher</p>', unsafe_allow_html=True)

    jd_text = st.text_area("Paste Job Description", height=220)

    if st.button("🚀 Find Best Candidates"):

        if len(st.session_state.resumes) == 0:
            st.warning("Upload resumes first.")

        elif jd_text.strip() == "":
            st.warning("Paste job description.")

        else:
            cleaned_jd = preprocess_text(jd_text)

            jd_vec = vectorizer.transform([cleaned_jd])

            results = []

            for r in st.session_state.resumes:

                resume_vec = vectorizer.transform([r["cleaned_text"]])

                score = cosine_similarity(resume_vec, jd_vec)[0][0] * 100

                pred = model.predict(resume_vec)[0]

                results.append((r["name"], pred, score))

            results = sorted(results, key=lambda x: x[2], reverse=True)

            st.subheader("🏆 Top Matches")

            for name, pred, score in results[:5]:

                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.write(f"📄 Resume: {name}")
                st.write(f"🎯 Predicted Role: {pred}")
                st.progress(min(int(score), 100))
                st.write(f"📊 Match Score: {round(score,2)}%")

                st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using NLP + Machine Learning + Streamlit")