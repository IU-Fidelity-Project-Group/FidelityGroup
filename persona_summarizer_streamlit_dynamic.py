import time
import os
import zipfile
import streamlit as st
import numpy as np
import json
from io import BytesIO
from pdfminer.high_level import extract_text
from openai import OpenAI
from utils import (
    extract_text_from_pdf, extract_text_from_zip,
    get_embedding, safe_cosine_similarity, chunk_text_by_tokens,
    query_astra_vectors_rest, log_skipped_summary,
    fetch_persona_names, fetch_persona_vector,
    extract_keywords_from_text, fetch_glossary_context,
    fetch_persona_metadata
)

# --------------------
# Configuration & Setup
# --------------------
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

profile_token = st.secrets["astra"]["profile_token"]
glossary_token = st.secrets["astra"]["glossary_token"]

profile_endpoint = "https://b897d7d9-a304-411c-abd0-836a9f38cc78-us-east1.apps.astra.datastax.com"
glossary_endpoint = "https://255cbde1-b53f-4dc1-b18b-8f9dbc13d28f-us-east1.apps.astra.datastax.com"

profile_collection = "profile_collection"
glossary_collection = "glossarycollection"

# --------------------
# Streamlit UI Setup
# --------------------
st.set_page_config(page_title="Persona Summarizer", layout="wide")
st.title("Cybersecurity Persona-Based Summarizer")

# Dynamically fetch persona names from the Astra DB profile collection
persona_list = fetch_persona_names(profile_endpoint, profile_token, profile_collection)
persona = st.sidebar.selectbox("Select Persona", persona_list)

# UI widgets for input
uploaded_file = st.sidebar.file_uploader("Upload PDF, ZIP, TXT, or JSON", type=["pdf", "zip", "txt", "json"])
max_toks = st.sidebar.slider("Max Summary Length", 100, 2000, 500, 100)
override_skip = st.sidebar.checkbox("Force summary even if relevance is low")
delay = st.sidebar.slider("Request Delay (seconds)", 0.0, 2.0, 0.2, 0.1)
generate = st.sidebar.button("Generate Summary")

# --------------------
# Summary Generation Logic
# --------------------
if generate:
    if not uploaded_file:
        st.warning("Please upload a file.")
        st.stop()

    if uploaded_file.name.endswith(".zip"):
        raw_text = extract_text_from_zip(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        raw_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".json"):
      try:
        data = json.load(uploaded_file)
        raw_text = json.dumps(data, indent=2)
      except Exception as e:
        st.error(f"Could not parse JSON: {e}")
        st.stop()
    else:
        raw_text = extract_text_from_pdf(uploaded_file)


    keyword_text = extract_keywords_from_text(raw_text, openai_client)

    if not keyword_text.strip():
        fallback_chunks = chunk_text_by_tokens(raw_text, chunk_size=1000, overlap=100)
        for chunk in fallback_chunks:
            keyword_text = extract_keywords_from_text(chunk, openai_client)
            if keyword_text.strip():
                break

    if not keyword_text.strip():
        st.warning("This document appears unrelated to cybersecurity. Summary generation has been skipped.")
        log_skipped_summary({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "persona": persona,
            "score": 0.0,
            "label": "Irrelevant",
            "filename": uploaded_file.name
        })
        st.stop()

    glossary_context = fetch_glossary_context(keyword_text, openai_client, glossary_collection, glossary_endpoint, glossary_token)
    doc_embedding = get_embedding(keyword_text, openai_client)
    persona_vector = fetch_persona_vector(persona, profile_endpoint, profile_token)
    persona_metadata = fetch_persona_metadata(persona, profile_endpoint, profile_token)

    score = safe_cosine_similarity(doc_embedding, persona_vector)

    if score >= 0.8:
        label = "Excellent"
    elif score >= 0.6:
        label = "Good"
    elif score >= 0.5:
        label = "Moderate"
    else:
        label = "Poor"

    score_display = f"Suitability Score: {score:.2f} ({label})"

    if score < 0.4 and not override_skip:
        st.subheader(score_display)
        st.warning("This document has limited relevance to the selected persona. Summary generation has been skipped.")
        log_skipped_summary({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "persona": persona,
            "score": round(score, 3),
            "label": label,
            "filename": uploaded_file.name
        })
        st.stop()
    else:
        color = 'green' if label == 'Good' else 'orange' if label == 'Moderate' else '#d4af37' if label == 'Fair' else 'red'
        st.markdown(f"<h3 style='color:{color}'>{score_display}</h3>", unsafe_allow_html=True)

        chunks = chunk_text_by_tokens(raw_text)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing chunk {i+1}/{len(chunks)}..."):
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": f"You are a {persona}."},
                            {"role": "user", "content": f"{chunk}\n\nSummarize this chunk for a {persona}."}
                        ],
                        max_tokens=max_toks
                    )
                    chunk_summaries.append(response.choices[0].message.content)
                    time.sleep(delay)
                except Exception as e:
                    chunk_summaries.append(f"[Error summarizing chunk {i+1}: {e}]")

        final_prompt = f"""
You are summarizing a cybersecurity document for a specific security persona: {persona} who {persona_metadata.get('description', '')}
Use the following persona profile to guide how the summary is written and what it emphasizes. Do **not** repeat the persona details in the output.
Only provide a **persona-informed summary of the document**, written in the appropriate tone and format.
---
**Tone**: {persona_metadata.get('tone', '')}
**Communication Style**: {persona_metadata.get('communication_style', '')}
**Preferred Format**: {persona_metadata.get('content_format', '')}
**Goals of the Persona**: {persona_metadata.get('goals', '')}
**Common Tasks the Persona Performs**: {persona_metadata.get('common_tasks', '')}
**Frameworks They Typically Use** (optional): {persona_metadata.get('frameworks', '')}
**Cybersecurity Domain this summary should focus on**: {persona_metadata.get('security_domain', '')}
---
**Instructions for Summary Generation**:
- Use the tone and communication style specified by the persona.
- Structure the output using the preferred format (e.g., brief, actionable bullet points, policy summary, technical note).
- Emphasize information that will help the persona achieve their goals.
- Highlight any insights related to their common tasks or workflows.
- If applicable, mention opportunities for integration with tools or policy controls.
- Avoid generic language — focus on what this persona needs to act or escalate.
- In the title of the response, declare what persona this summary is intended for.
The following glossary terms may help you understand the source text:
{glossary_context}
Chunk Summaries:
{'\n\n'.join(chunk_summaries)}
Generate a summary that would be maximally useful to the **{persona}**, written in their tone and aligned with their goals.
"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are a {persona}."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=600
            )
            st.subheader("Final Executive Summary")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"[Error generating final summary: {e}]")
