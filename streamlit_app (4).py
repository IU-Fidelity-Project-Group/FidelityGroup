import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import zipfile
import io

# --------------------
# Configuration & Secrets
# --------------------
# In your Streamlit Secrets panel:
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
        "strategic direction of cybersecurity policies, ensure regulatory compliance (e.g., GDPR), lead incident response, "
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
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t)
    return "\n\n".join(texts)

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Persona Summarizer", layout="wide")
st.title("Cybersecurity Persona–Based Summarizer")

# Sidebar controls
st.sidebar.header("Controls")
persona = st.sidebar.selectbox("Select Persona", list(DESCRIPTIONS.keys()))

uploaded = st.sidebar.file_uploader(
    "Upload a PDF or ZIP of PDFs",
    type=["pdf", "zip"],
    help="Max size per file: ~200 MB"
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
    if not uploaded:
        st.sidebar.warning("Please upload a PDF or ZIP first.")
    else:
        docs = []
        # Handle ZIP of PDFs
        if uploaded.type == "application/zip":
            with zipfile.ZipFile(uploaded) as z:
                for fname in z.namelist():
                    if fname.lower().endswith(".pdf"):
                        with z.open(fname) as f:
                            text = extract_text_from_pdf(f)
                            docs.append((fname, text))
        else:
            # Single PDF
            text = extract_text_from_pdf(uploaded)
            docs.append((uploaded.name or "Document", text))

        # Summarize each document
        for title, text in docs:
            system_msg = {"role": "system", "content": DESCRIPTIONS[persona]}
            user_msg = {
                "role": "user",
                "content": (
                    f"Document content:\n\n{text}\n\n"
                    f"Please provide a **detailed** summary for a {persona}, "
                    f"using as close to {max_toks} tokens as possible (but no more)."
                ),
            }
            with st.spinner(f"Summarizing {title}..."):
                try:
                    resp = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[system_msg, user_msg],
                        max_tokens=max_toks,
                    )
                    summary = resp.choices[0].message.content
                    st.subheader(f"Summary: {title}")
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
