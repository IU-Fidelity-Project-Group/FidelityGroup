import os
import zipfile
import streamlit as st
import numpy as np
import tiktoken
from io import BytesIO
from pdfminer.high_level import extract_text
from openai import OpenAI
from astrapy import DataAPIClient

# --------------------
# Config
# --------------------
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)
encoder = tiktoken.encoding_for_model("gpt-4o")

# AstraDB Config
profile_client = DataAPIClient(st.secrets["astra"]["profile_token"])
glossary_client = DataAPIClient(st.secrets["astra"]["glossary_token"])

persona_db = profile_client.get_database_by_api_endpoint("https://b897d7d9-a304-411c-abd0-836a9f38cc78-us-east1.apps.astra.datastax.com")
glossary_db = glossary_client.get_database_by_api_endpoint("https://255cbde1-b53f-4dc1-b18b-8f9dbc13d28f-us-east1.apps.astra.datastax.com")

persona_collection = persona_db["profile_collection"]
glossary_collection = glossary_db["glossarycollection"]

# --------------------
# Embedding
# --------------------
def get_embedding(text, max_tokens=8192, max_chars=16000):
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        st.warning(f"⚠️ Truncating embedding input from {len(tokens)} tokens to {max_tokens}")
        tokens = tokens[:max_tokens]
    text = encoder.decode(tokens)
    if len(text) > max_chars:
        st.warning(f"⚠️ Truncating embedding input from {len(text)} characters to {max_chars}")
        text = text[:max_chars]
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(response.data[0].embedding, dtype=np.float32)

# --------------------
# Astra Vector Search
# --------------------
def query_astra_vectors(collection, embedding, top_k):
    result = collection.search(vector=embedding, limit=top_k)
    return result["data"]["documents"]

# --------------------
# PDF Text Extraction using pdfminer
# --------------------
def extract_text_from_pdf(file):
    return extract_text(BytesIO(file.read()))

def extract_text_from_zip(file):
    with zipfile.ZipFile(file) as z:
        return "\n\n".join([
            extract_text(BytesIO(z.read(n))) for n in z.namelist() if n.lower().endswith(".pdf")
        ])

# --------------------
# Text Chunking
# --------------------
def chunk_text_by_tokens(text, chunk_size=3072, overlap=256):
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = encoder.decode(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# --------------------
# Streamlit App
# --------------------
st.set_page_config(page_title="Persona Summarizer", layout="wide")
st.title("Cybersecurity Persona-Based Summarizer")

# Load personas from DB
persona_cursor = persona_collection.find()
persona_list = [
    doc["metadata"]["persona"]
    for doc in persona_cursor
    if "metadata" in doc and "persona" in doc["metadata"]
]
persona = st.sidebar.selectbox("Select Persona", persona_list)

# Fetch full persona description
persona_doc = persona_collection.find_one({"metadata.persona": {"$eq": persona}})
persona_metadata = persona_doc.get("metadata", {}) if persona_doc else {}
persona_description = persona_metadata.get("description", "")

uploaded_file = st.sidebar.file_uploader("Upload PDF or ZIP", type=["pdf", "zip"])
max_toks = st.sidebar.slider("Max tokens per chunk", 100, 2000, 500, 100)
generate = st.sidebar.button("Generate Summary")

if generate:
    if not uploaded_file:
        st.warning("Please upload a file.")
        st.stop()

    raw_text = extract_text_from_zip(uploaded_file) if uploaded_file.name.endswith(".zip") else extract_text_from_pdf(uploaded_file)
    raw_text = raw_text.replace("\\n", " ").replace("\\u200b", "").strip()

    doc_embedding = get_embedding(raw_text)
    glossary_hits = query_astra_vectors(glossary_collection, doc_embedding, top_k=20)
    glossary_context = "\n\n".join([d.get("text", "") for d in glossary_hits])

    persona_hits = query_astra_vectors(persona_collection, doc_embedding, top_k=1)
    persona_context = "\n\n".join([d.get("text", "") for d in persona_hits])

    text_chunks = chunk_text_by_tokens(raw_text, chunk_size=3072, overlap=256)
    chunk_summaries = []

    for i, chunk in enumerate(text_chunks):
        with st.spinner(f"Summarizing chunk {i+1}/{len(text_chunks)}..."):
            system_msg = {"role": "system", "content": persona_description}
            user_msg = {"role": "user", "content": f"{persona_context}\n\n{glossary_context}\n\n{chunk}\n\nPlease summarize this chunk for a {persona}."}
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[system_msg, user_msg],
                    max_tokens=max_toks
                )
                chunk_summaries.append(response.choices[0].message.content)
            except Exception as e:
                chunk_summaries.append(f"[Error summarizing chunk {i+1}: {e}]")

    st.subheader("Combined Summary")
    for i, summary in enumerate(chunk_summaries, 1):
        st.markdown(f"### Chunk {i}")
        st.write(summary)
