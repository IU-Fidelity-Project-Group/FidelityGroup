import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader


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

st.title("Cybersecurity Persona–Based Summarizer")

# 1) Persona selector
persona = st.selectbox("Select Persona", list(DESCRIPTIONS.keys()))

# 2) PDF uploader
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Max size: ~200 MB",
)

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text_chunks = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text_chunks.append(txt)
    return "\n\n".join(text_chunks)

# 3) Generate summary
if st.button("Generate Summary"):
    if not uploaded_file:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Extracting text…"):
            document_text = extract_text_from_pdf(uploaded_file)

        # Build messages
        system_msg = {"role": "system", "content": DESCRIPTIONS[persona]}
        user_msg = {
            "role": "user",
            "content": (
                f"Here is the document content:\n\n{document_text}\n\n"
                f"Please summarize it for a {persona}."
            ),
        }

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[system_msg, user_msg],
            )
            summary = resp.choices[0].message.content
            st.subheader("Summary Output")
            st.write(summary)
        except Exception as e:
            st.error(f"OpenAI error: {e}")

# 4) Simple rating UI
st.markdown("---")
st.subheader("Rate this summary")
rating = st.radio("Stars", [1, 2, 3, 4, 5], index=4, key="rating")
comment = st.text_area("Additional comments", key="comment")

if st.button("Submit Feedback"):
    # Placeholder: hook into your storage of choice later
    st.success(f"Thanks! You rated this {rating} star(s).")
    if comment:
        st.info(f"Your comment: “{comment}”")
