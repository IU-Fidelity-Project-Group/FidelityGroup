# Cybersecurity Persona-Based Document Summarizer

This Streamlit app performs cybersecurity-specific summarization of uploaded PDF or ZIP documents based on a selected **persona profile** (e.g., Threat Intelligence Analyst, Malware Analyst). It extracts relevant context, applies glossary and persona alignment, and generates summaries tailored to the selected role using OpenAI models.

## 🧱 Features

- Dynamically loaded persona profiles from AstraDB
- PDF and ZIP (multi-PDF) upload and parsing
- Keyword extraction via OpenAI
- Suitability scoring (document vs persona relevance)
- Persona-aware summary generation with glossary support
- Logging of skipped or irrelevant documents

---

## 🧰 Requirements

- Python 3.10+
- Access to an OpenAI API key (used for embeddings and summarization)
- AstraDB instance (Vector Search enabled)
  - `profile_collection` must be populated with vector-enabled persona documents
  - `glossarycollection` must be populated with embedded cybersecurity terms

---

## 🔐 Not Included

This repo does **not** include:

- OpenAI or Astra API credentials
- Streamlit Cloud deployment configuration
- Sample AstraDB data (profiles/glossary)

---

## 🔧 Setup Instructions

### Step 1: Install Dependencies

1. Create and activate a virtual environment:
   - `python -m venv .venv`
   - On macOS/Linux: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`
2. Install Python packages:
   - `pip install -r requirements.txt`

### Step 2: Add OpenAI and Astra Credentials

1. Inside the project root, create a directory named `.streamlit` if it doesn't exist.
2. In that directory, create a file called `secrets.toml`.
3. Add the following content to `secrets.toml`, replacing the placeholder values:

'[openai] 
api_key = "sk-your-openai-key"

[astra] 
profile_token = "your-profile-db-token" 
glossary_token = "your-glossary-db-token"'


### Step 3: Run the App

1. From the project root, run:
- `streamlit run persona_summarizer_streamlit_dynamic.py`
2. Open the app in your browser when Streamlit provides the URL.

---

## 📁 File Overview

- `persona_summarizer_streamlit_dynamic.py` — Main Streamlit app logic
- `utils.py` — Helper functions (embedding, vector fetch, keyword parsing, etc.)
- `skipped_summaries.csv` — Log of documents skipped due to low relevance
- `/personas` — (optional) Reference JSON profiles for personas (if available)

---

## 📌 Notes

- The app assumes AstraDB collections already exist with valid documents and $vector fields.
- Summaries are generated using the GPT-4o model.
- ZIP file support allows batch summarization of multi-PDF document sets.

---

## 🧭 Next Steps (Optional)

- Enhance personas and glossary collections
- Improve scoring heuristics
- Add support for audit logging, user authentication, or CSV export

