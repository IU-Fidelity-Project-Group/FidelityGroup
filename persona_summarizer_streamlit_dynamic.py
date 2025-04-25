import time
import os
import zipfile
import streamlit as st
import numpy as np
from io import BytesIO
from pdfminer.high_level import extract_text
from openai import OpenAI
from utils import (
    extract_text_from_pdf, extract_text_from_zip,
    get_embedding, safe_cosine_similarity, chunk_text_by_tokens,
    query_astra_vectors_rest, log_skipped_summary,
    fetch_persona_names, fetch_persona_vector,
    extract_keywords_from_text
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
uploaded_file = st.sidebar.file_uploader("Upload PDF or ZIP", type=["pdf", "zip"])
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

    # Extract raw text from uploaded document
    raw_text = extract_text_from_zip(uploaded_file) if uploaded_file.name.endswith(".zip") else extract_text_from_pdf(uploaded_file)

    # Extract cybersecurity-relevant keywords only
    keyword_text = extract_keywords_from_text(raw_text, openai_client)
    st.write("Extracted keywords:", keyword_text)

    # Guardrail: if keywords are empty, halt due to irrelevance
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

    # Compute vector-based relevance match
    doc_embedding = get_embedding(keyword_text, openai_client)
    persona_vector = fetch_persona_vector(persona, profile_endpoint, profile_token)

    score = safe_cosine_similarity(doc_embedding, persona_vector)

    # Label assignment based on adjusted cosine score
    if score >= 0.85:
        label = "Good"
    elif score >= 0.65:
        label = "Moderate"
    elif score >= 0.45:
        label = "Fair"
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

        # --------------------
        # Document Chunking & Summarization
        # --------------------
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

        # --------------------
        # Final Executive Summary Generation
        # --------------------
        final_prompt = f"""
You are summarizing a technical cybersecurity document for a {persona}.
Your goal is to extract and synthesize only the most relevant, actionable, and persona-specific insights from the chunk summaries provided below.
Focus on findings, issues, techniques, and concepts that align with the responsibilities and decision-making scope of a {persona}. Avoid generic restatements or overly broad summaries.
Chunk Summaries:
{chr(10).join(chunk_summaries)}
Write a final executive summary that would be directly useful to a {persona}.
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
