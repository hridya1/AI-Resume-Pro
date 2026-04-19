import streamlit as st
import pickle
import re
import nltk
import os
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from docx import Document
import PyPDF2

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Talent Screening Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# LOAD NLTK
# ---------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "resumes" not in st.session_state:
    st.session_state.resumes = []

# ---------------------------------------------------
# CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Main App */
.stApp{
    background:#f5f5f7;
    color:#1d1d1f;
    font-family:-apple-system,BlinkMacSystemFont,sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#111827,#1f2937);
    border-right:1px solid #e5e5e7;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Buttons */
.stButton>button{
    background:#0071e3;
    color:white;
    border:none;
    border-radius:14px;
    height:50px;
    width:100%;
    font-weight:600;
    font-size:16px;
}

.stButton>button:hover{
    background:#0077ED;
}

/* Inputs */
textarea, input, .stSelectbox > div > div{
    border-radius:14px !important;
    border:1px solid #d2d2d7 !important;
}

/* Cards */
.card{
    background:white;
    padding:28px;
    border-radius:22px;
    box-shadow:0 8px 20px rgba(0,0,0,0.05);
    margin-bottom:18px;
}

/* Role Output */
.role-box{
    background:#f5f5f7;
    padding:25px;
    border-radius:18px;
    text-align:center;
    font-size:28px;
    font-weight:700;
}

/* Results */
.result-box{
    background:white;
    padding:20px;
    border-radius:18px;
    box-shadow:0 8px 20px rgba(0,0,0,0.05);
    margin-bottom:15px;
}

/* Footer */
.footer{
    text-align:center;
    color:#86868b;
    font-size:13px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# PREPROCESS
# ---------------------------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\s\-]{8,}\d', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)

# ---------------------------------------------------
# FILE READERS
# ---------------------------------------------------
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

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Talent Screening Portal")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Candidate Classification", "Role Matching"]
)

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------
if page == "Dashboard":

    st.markdown("""
    <h1 style='font-size:34px;font-weight:700;color:#1d1d1f;margin-bottom:6px;'>
    Talent Screening Dashboard
    </h1>
    <p style='font-size:16px;color:#6e6e73;margin-bottom:22px;'>
    Upload resumes and manage candidate screening workflow.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(f"""
        <div style="
            background:white;
            padding:34px;
            border-radius:22px;
            text-align:center;
            box-shadow:0 8px 20px rgba(0,0,0,0.05);
        ">
            <div style="font-size:17px;color:#6e6e73;margin-bottom:10px;">
                Stored Resumes
            </div>
            <div style="font-size:42px;font-weight:700;">
                {len(st.session_state.resumes)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background:white;
            padding:34px;
            border-radius:22px;
            text-align:center;
            box-shadow:0 8px 20px rgba(0,0,0,0.05);
        ">
            <div style="font-size:17px;color:#6e6e73;margin-bottom:10px;">
                Ready for Screening
            </div>
            <div style="font-size:42px;font-weight:700;">
                {len(st.session_state.resumes)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='font-size:24px;font-weight:700;color:#1d1d1f;margin-bottom:12px;'>
    Upload Resumes
    </h2>
    """, unsafe_allow_html=True)

    files = st.file_uploader(
        "Upload PDF or DOCX Files",
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
                    "cleaned_text": cleaned
                })

        st.success(f"{len(files)} resumes uploaded successfully.")

# ---------------------------------------------------
# CANDIDATE CLASSIFICATION
# ---------------------------------------------------
elif page == "Candidate Classification":

    st.markdown("""
    <h1 style='font-size:34px;font-weight:700;color:#1d1d1f;'>
    Candidate Classification
    </h1>
    <p style='font-size:16px;color:#6e6e73;margin-bottom:22px;'>
    Predict the most relevant category for uploaded resumes.
    </p>
    """, unsafe_allow_html=True)

    if len(st.session_state.resumes) == 0:
        st.warning("Please upload resumes first.")

    else:
        selected = st.selectbox(
            "Select Resume",
            [r["name"] for r in st.session_state.resumes]
        )

        resume = next(r for r in st.session_state.resumes if r["name"] == selected)

        if st.button("Run Classification"):

            X = vectorizer.transform([resume["cleaned_text"]])
            prediction = model.predict(X)[0]

            st.markdown(f"""
            <div class="card">
                <div class="role-box">{prediction}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------
# ROLE MATCHING
# ---------------------------------------------------
elif page == "Role Matching":

    st.markdown("""
    <h1 style='font-size:34px;font-weight:700;color:#1d1d1f;'>
    Role Matching
    </h1>
    <p style='font-size:16px;color:#6e6e73;margin-bottom:22px;'>
    Rank uploaded resumes against hiring requirements.
    </p>
    """, unsafe_allow_html=True)

    jd_text = st.text_area("Enter Job Description", height=220)

    if st.button("Generate Match Results"):

        if len(st.session_state.resumes) == 0:
            st.warning("Please upload resumes first.")

        elif jd_text.strip() == "":
            st.warning("Please enter a job description.")

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

            for name, pred, score in results[:5]:

                st.markdown(f"""
                <div class="result-box">
                    <b>Candidate Resume:</b> {name}<br><br>
                    <b>Predicted Category:</b> {pred}<br><br>
                    <b>Match Score:</b> {round(score,2)}%
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(int(score),100))

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<div class="footer">
Talent Screening Portal
</div>
""", unsafe_allow_html=True)
