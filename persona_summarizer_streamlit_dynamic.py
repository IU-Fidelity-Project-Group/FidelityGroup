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
    get_embedding, cosine_similarity, chunk_text_by_tokens,
    query_astra_vectors_rest, log_skipped_summary,
    fetch_persona_names, fetch_persona_description
)

# --------------------
# Config
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
# Streamlit App
# --------------------
st.set_page_config(page_title="Persona Summarizer", layout="wide")
st.title("Cybersecurity Persona-Based Summarizer")

persona_list = ["Malware Analyst", "Application Security Analyst", "Threat Intelligence Analyst", "SOC Analyst", "Cyber Risk Analyst / CISO", "Network Security Analyst", "Vendor Security Specialist", "DLP / Insider Threat Analyst"]
persona = st.sidebar.selectbox("Select Persona", persona_list)

uploaded_file = st.sidebar.file_uploader("Upload PDF or ZIP", type=["pdf", "zip"])
max_toks = st.sidebar.slider("Max tokens per chunk", 100, 2000, 500, 100)
override_skip = st.sidebar.checkbox("Force summary even if relevance is low")
delay = st.sidebar.slider("Request Delay (seconds)", 0.0, 2.0, 0.2, 0.1)
generate = st.sidebar.button("Generate Summary")

if generate:
    if not uploaded_file:
        st.warning("Please upload a file.")
        st.stop()

    raw_text = extract_text_from_zip(uploaded_file) if uploaded_file.name.endswith(".zip") else extract_text_from_pdf(uploaded_file)
    doc_embedding = get_embedding(raw_text, openai_client)

    glossary_hits = query_astra_vectors_rest(glossary_collection, glossary_endpoint, glossary_token, doc_embedding, top_k=5)
    glossary_context = "\n\n".join([d.get("text", "") for d in glossary_hits])

    persona_context = "\n\n".join([d.get("text", "") for d in query_astra_vectors_rest(profile_collection, profile_endpoint, profile_token, doc_embedding, top_k=1)])
    persona_description = fetch_persona_description(persona, profile_endpoint, profile_token)

    persona_embedding = get_embedding(persona_description, openai_client)

    st.write("Embedding lengths:", len(doc_embedding), len(persona_embedding))
    st.write("First 5 values:", doc_embedding[:5], persona_embedding[:5])
    st.write("Persona description:", persona_description[:200])
    st.write("Doc text (truncated):", raw_text[:200])

    score = cosine_similarity(doc_embedding, persona_embedding)
    score = max(0.0, min(1.0, (score + 1) / 2))

    if score >= 0.8:
        label = "Good"
    elif score >= 0.6:
        label = "Moderate"
    elif score >= 0.4:
        label = "Fair"
    else:
        label = "Poor"

    score_display = f"Suitability Score: {score:.2f} ({label})"

    if score < 0.4 and not override_skip:
        st.subheader(score_display)
        st.warning("⚠️ This document has limited relevance to the selected persona. Summary generation has been skipped.")
        log_skipped_summary({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "persona": persona,
            "score": round(score, 3),
            "label": label,
            "filename": uploaded_file.name
        })
    else:
        st.markdown(f"<h3 style='color:{{'green' if label == 'Good' else 'orange' if label == 'Moderate' else '#d4af37' if label == 'Fair' else 'red'}}'>{{score_display}}</h3>", unsafe_allow_html=True)

        chunks = chunk_text_by_tokens(raw_text)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing chunk {{i+1}}/{{len(chunks)}}..."):
                system_msg = {{"role": "system", "content": persona_description}}
                user_msg = {{"role": "user", "content": f"{{persona_context}}\n\n{{glossary_context}}\n\n{{chunk}}\n\nPlease summarize this chunk for a {{persona}}."}}
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[system_msg, user_msg],
                        max_tokens=max_toks
                    )
                    chunk_summaries.append(response.choices[0].message.content)
                    time.sleep(delay)
                except Exception as e:
                    chunk_summaries.append(f"[Error summarizing chunk {{i+1}}: {{e}}]")

        final_prompt = f"""You are summarizing a technical cybersecurity document for a {{persona}}.

Your goal is to extract and synthesize only the most relevant, actionable, and persona-specific insights from the chunk summaries provided below.

Exclude generalities and prioritize insights, findings, issues, or context that align with the responsibilities and focus areas of a {{persona}}.

Chunk Summaries:
{{'\n\n'.join(chunk_summaries)}}

Write a final executive summary that would be directly useful to a {{persona}}."""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[system_msg, {{"role": "user", "content": final_prompt}}],
                max_tokens=600
            )
            st.subheader("Final Executive Summary")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"[Error generating final summary: {{e}}]")
