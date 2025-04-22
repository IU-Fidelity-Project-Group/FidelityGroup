import os
import zipfile
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import streamlit as st
from openai import OpenAI
import tiktoken 

# --------------------
# Configuration & Secrets
# --------------------
# In Streamlit Secrets panel:
# [openai]
# api_key = "<YOUR-SK-KEY>"

openai_client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# --------------------
# Persona descriptions
# --------------------
DESCRIPTIONS = {
    "Vendor Security Specialist": (
        "Vendor security specialists are responsible for assessing and managing the cybersecurity posture of "
        "third‑party vendors as well as the vendor’s products and services. They focus on integrations, data "
        "security practices, SOC 2 and ISO 27001 compliance, vendor audits, and security clauses in contracts."
    ),
    "Network Security Analyst": (
        "Network Security Analysts secure data transmission within an organization’s IT infrastructure, including "
        "cloud and on‑prem environments. Responsibilities include firewalls, IDS/IPS, VPNs, network segmentation, "
        "zero‑trust architecture, traffic monitoring, access controls, and incident response."
    ),
    "Cyber Risk Analyst / CISO / ISO": (
        "Cyber Risk Analysts identify and prioritize risks to the organization’s IT environment. CISOs and ISOs oversee "
        "strategic direction of cybersecurity policies, ensure regulatory compliance (e.g. GDPR), lead incident response, "
        "crisis management, and align security practices with business objectives."
    ),
    "Application Security Analyst": (
        "Application Security Analysts focus on secure coding practices, identify vulnerabilities in development frameworks, "
        "monitor the OWASP Top 10 and zero‑day threats, and integrate SAST/DAST tools into CI/CD pipelines."
    ),
    "Threat Intelligence Analyst": (
        "Threat Intelligence Analysts track evolving threat actor TTPs, especially those targeting financial systems. "
        "They analyze supply chain attacks, use MITRE ATT&CK, monitor initial access and lateral movement, "
        "and collaborate with sources like FS‑ISAC."
    ),
    "DLP / Insider Threat Analyst": (
        "DLP and Insider Threat Analysts monitor internal data misuse, detect policy failures, USB/file transfers, "
        "shadow IT activities, and enforce DLP policies using UEBA platforms."
    ),
    "Malware Analyst": (
        "Malware Analysts reverse‑engineer malware, study payload behavior, track new strains, use YARA and Ghidra, "
        "and analyze IOCs to understand ransomware, trojans, and C2 frameworks."
    ),
}

# --------------------
# PDF text extraction
# --------------------
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)

def extract_text_from_zip(file) -> str:
    texts = []
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            if name.lower().endswith(".pdf"):
                with z.open(name) as pdf_file:
                    try:
                        texts.append(extract_text_from_pdf(pdf_file))
                    except PdfReadError:
                        texts.append(f"[ERROR READING {name}]")
    return "\n\n".join(texts)

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Persona Summarizer", layout="wide")
st.title("Cybersecurity Persona–Based Summarizer")

# Sidebar controls
st.sidebar.header("Controls")
persona = st.sidebar.selectbox("Select Persona", list(DESCRIPTIONS.keys()))
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or ZIP of PDFs",
    type=["pdf", "zip"],
    help="Max size: ~200 MB",
)
max_toks = st.sidebar.slider(
    "Max summary length (tokens)",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
)
generate = st.sidebar.button("Generate Summary")

if generate:
    if not uploaded_file:
        st.sidebar.warning("Please upload a file first.")
    else:
        with st.spinner("Extracting text..."):
            name = uploaded_file.name.lower()
            if name.endswith(".zip"):
                document_text = extract_text_from_zip(uploaded_file)
            else:
                try:
                    document_text = extract_text_from_pdf(uploaded_file)
                except PdfReadError as e:
                    st.sidebar.error(f"PDF parse error: {e}")
                    st.stop()

        # Truncate to fit under token-per-minute limits
        encoder = tiktoken.encoding_for_model("gpt-4o")
        sys_tokens = len(encoder.encode(DESCRIPTIONS[persona]))
        preamble = "Document content:\n\n"
        postamble = f"\n\nPlease provide a detailed summary for a {persona}, using up to {max_toks} tokens."
        overhead_tokens = (
            sys_tokens
            + len(encoder.encode(preamble))
            + len(encoder.encode(postamble))
            + max_toks
            + 50
        )
        tpm_limit = 30000
        allowed_doc_tokens = max(0, tpm_limit - overhead_tokens)

        doc_tokens = encoder.encode(document_text)
        if len(doc_tokens) > allowed_doc_tokens:
            doc_tokens = doc_tokens[:allowed_doc_tokens]
            document_text = encoder.decode(doc_tokens)
            st.warning(
                f"⚠️ Input was too long and has been truncated to ~{allowed_doc_tokens} tokens."
            )

        system_msg = {"role": "system", "content": DESCRIPTIONS[persona]}
        user_msg = {
            "role": "user",
            "content": (
                f"Document content:\n\n{document_text}\n\n"
                f"Please provide a detailed summary for a {persona}, "
                f"using up to {max_toks} tokens."
            ),
        }

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[system_msg, user_msg],
                max_tokens=max_toks,
            )
            summary = resp.choices[0].message.content
            st.subheader("Summary Output")
            st.write(summary)
        except Exception as e:
            st.error(f"OpenAI error: {e}")

# --------------------
# Feedback stub
# --------------------
def send_feedback_to_sheet(rating: int, comment: str):
    """
    TODO: wire this up to Google Sheets (e.g. via gspread) to append each feedback.
    """
    pass

st.sidebar.markdown("---")
st.sidebar.subheader("Rate this summary")
rating = st.sidebar.radio("Stars", [1, 2, 3, 4, 5], index=4)
comment = st.sidebar.text_area("Additional comments")
if st.sidebar.button("Submit Feedback"):
    send_feedback_to_sheet(rating, comment)
    st.success(f"Thanks! You rated this {rating} star(s).")
    if comment:
        st.info(f"Your comment: “{comment}”")
