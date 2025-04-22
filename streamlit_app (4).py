import io
import zipfile
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

# --------------------
# Configuration & Secrets
# --------------------
# In Streamlit Secrets panel:
# [openai]
# api_key = "<YOUR-SK-KEY>"

openai_client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Predefined persona descriptions
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

def extract_text_from_pdf(file) -> str:
    """
    Extracts and concatenates text from all pages of a PDF file-like object.
    """
    reader = PdfReader(file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)

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
    help="Max size per PDF: ~200 MB",
)
max_toks = st.sidebar.slider(
    "Max summary length (tokens)",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
)
generate = st.sidebar.button("Generate Summary")

# Generate summaries
if generate:
    if not uploaded_file:
        st.sidebar.warning("Please upload a file first.")
    else:
        system_msg = {"role": "system", "content": DESCRIPTIONS[persona]}

        # Handle ZIP of PDFs
        if uploaded_file.name.lower().endswith(".zip") or uploaded_file.type == "application/zip":
            data = uploaded_file.read()
            z = zipfile.ZipFile(io.BytesIO(data))
            pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]
            if not pdf_files:
                st.error("No PDFs found in the ZIP.")
            else:
                for pdf_name in pdf_files:
                    with st.spinner(f"Processing {pdf_name}..."):
                        pdf_bytes = z.read(pdf_name)
                        document_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
                    user_msg = {
                        "role": "user",
                        "content": (
                            f"Document content from {pdf_name}:\n\n{document_text}\n\n"
                            f"Please summarize this for a {persona}."
                        ),
                    }
                    try:
                        resp = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[system_msg, user_msg],
                            max_tokens=max_toks,
                        )
                        summary = resp.choices[0].message.content
                        st.subheader(f"Summary for {pdf_name}")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"OpenAI error for {pdf_name}: {e}")

        # Handle single PDF
        else:
            with st.spinner("Extracting text..."):
                pdf_bytes = uploaded_file.read()
                document_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
            user_msg = {
                "role": "user",
                "content": (
                    f"Document content:\n\n{document_text}\n\n"
                    f"Please summarize this for a {persona}."
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

# Feedback section
st.sidebar.markdown("---")
st.sidebar.subheader("Rate this summary")
rating = st.sidebar.radio("Stars", [1, 2, 3, 4, 5], index=4)
comment = st.sidebar.text_area("Additional comments")
if st.sidebar.button("Submit Feedback"):
    st.success(f"Thanks! You rated this {rating} star(s).")
    if comment:
        st.info(f"Your comment: “{comment}”")
