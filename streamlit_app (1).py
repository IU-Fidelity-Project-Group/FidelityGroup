import os
import streamlit as st
import openai
from unstructured.partition.pdf import partition_pdf

# --------------------
# Configuration & Secrets
# --------------------
# Set Unstructured API key for PDF parsing
os.environ["UNSTRUCTURED_API_KEY"] = st.secrets["unstructured"]["api_key"]

# Load OpenAI API key securely
openai.api_key = st.secrets["openai"]["api_key"]

# --------------------
# Persona Definitions
# --------------------
personas = {
    "SOC Analyst": (
        "You are a SOC Analyst. Summarize focusing on threat indicators and incident triage."
    ),
    "CISO": (
        "You are a CISO. Summarize focusing on high-level risks and strategic security decisions."
    ),
    "Compliance Officer": (
        "You are a Compliance Officer. Summarize focusing on policy compliance and regulatory gaps."
    ),
    "Privacy Engineer": (
        "You are a Privacy Engineer. Summarize focusing on data handling and privacy implications."
    ),
    "Vendor Security Specialist": (
        "Vendor security specialists are responsible for assessing and managing the cybersecurity posture of "
        "third-party vendors as well as the vendor’s products and services. They identify vulnerabilities, "
        "manage vendor audits, and enforce security clauses in contracts."
    ),
    "Network Security Analyst": (
        "Network Security Analysts secure data transmission within an organization’s IT infrastructure, including "
        "cloud and on-prem environments. They manage firewalls, IDS/IPS, VPNs, network segmentation, and zero-trust architecture."
    ),
    "Cyber Risk Analyst / CISO / ISO": (
        "Cyber Risk Analysts identify and prioritize risks to the organization’s IT environment. "
        "CISOs and ISOs oversee cybersecurity policy, compliance (e.g., GDPR), incident response, and align security with business goals."
    ),
    "Application Security Analyst": (
        "Application Security Analysts focus on secure coding practices, vulnerabilities in development frameworks, and CI/CD pipeline security. "
        "They monitor OWASP Top 10, SSRF, code injection, and buffer overflow threats."
    ),
    "Threat Intelligence Analyst": (
        "Threat Intelligence Analysts track evolving threat actor TTPs, analyze supply chain attacks, MITRE ATT&CK mappings, "
        "and collaborate with intelligence sources."
    ),
    "DLP / Insider Threat Analyst": (
        "DLP and Insider Threat Analysts focus on internal data misuse, policy failures, and behavioral indicators. "
        "They enforce DLP policies, monitor privilege misuse, and use UEBA for insider threat detection."
    ),
    "Malware Analyst": (
        "Malware Analysts reverse engineer malware, study payload behavior, and identify obfuscation techniques. "
        "They use tools like YARA and Ghidra to analyze ransomware, trojans, and C2 structures."
    )
}

# --------------------
# Helper Functions
# --------------------
def extract_text(uploaded_file):
    """
    Extract text from uploaded file. Uses Unstructured for PDFs, falls back to plain text for others.
    """
    file_bytes = uploaded_file.read()
    if uploaded_file.type == "application/pdf":
        elements = partition_pdf(file=uploaded_file)
        return "\n".join(el.text for el in elements)
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return str(file_bytes)

# --------------------
# Streamlit App UI
# --------------------
st.set_page_config(page_title="Persona-Based Summarizer", layout="wide")
st.title("Cybersecurity Persona-Based Summarizer")

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])

# Persona selection dropdown
selected_persona = st.selectbox("Select Persona", list(personas.keys()))

# Generate summary
if uploaded_file and st.button("Generate Summary"):
    with st.spinner("Extracting content..."):
        document_text = extract_text(uploaded_file)

    system_prompt = personas[selected_persona]
    prompt = f"{system_prompt}\n\n{document_text}"

    with st.spinner("Generating summary..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        summary = response.choices[0].message.content

    st.subheader("Summary Output")
    st.text_area("", value=summary, height=300)

    # Rating system
    st.markdown("---")
    st.subheader("Rate This Summary")
    rating = st.slider("How would you rate this summary? (1 = poor, 5 = excellent)", 1, 5, 3)
    feedback = st.text_area("Optional feedback:", placeholder="What did you like or what could be improved?", height=80)
    if st.button("Submit Feedback"):
        st.success(f"Thanks for your feedback! You rated this summary {rating} stars.")
