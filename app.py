import streamlit as st
import pdfplumber
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up Streamlit page
st.set_page_config(page_title="📄 ATS Resume Score Checker", layout="centered")
st.title("📄 ATS Resume Score Checker")
st.caption("Upload your resume and paste the job description to get your ATS match score.")

# Upload resume
resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

# Input job description (text only)
jd_text_input = st.text_area("Paste the Job Description Here", height=300, placeholder="Paste the job description here...")

# Extract text from resume PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text.strip()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
     ("system", 
    """You are an AI ATS Resume Evaluator. Your task is to compare a candidate's resume with a job description.

1. Extract important keywords (skills, tools, technologies, qualifications) from the job description.
2. Compare with the candidate’s resume.
3. Identify which important keywords are missing in the resume.
4. Score the resume from 0 to 100.
5. Provide reasoning and suggestions for improvement.

Respond in this exact format:

---
**ATS Score:** <score>/100

**Reasoning:** <summary of analysis>

**Suggestions to Improve ATS Score:**
- Tip 1
- Tip 2

**Missing Keywords (Not Found in Resume):**
- keyword1
- keyword2
---
"""),
    ("user", "Resume:\n{resume}\n\nJob Description:\n{jd}")
])

# LangChain model pipeline
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
output_parser = StrOutputParser()
chain = prompt | model | output_parser

# Process and evaluate
# Always show button
evaluate_button = st.button("📊 Evaluate ATS Score")

if evaluate_button:
    if not resume_file or not jd_text_input.strip():
        st.warning("Please upload your resume and paste the job description.")
    else:
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(resume_file)
            jd_text = jd_text_input.strip()

            response = chain.invoke({
                "resume": resume_text,
                "jd": jd_text
            })

            st.success("✅ Analysis Complete")
            st.markdown(response)
            