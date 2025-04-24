# utils.py

import openai
import numpy as np
import pandas as pd
import requests
import tiktoken
from io import BytesIO
from pdfminer.high_level import extract_text
import zipfile

# Set up token encoder for managing OpenAI token limits
encoder = tiktoken.encoding_for_model("gpt-4o")

# ------------------------------------------------------
# Embeds text using the OpenAI embedding model.
# Truncates to fit token and character limits if needed.
# ------------------------------------------------------
def get_embedding(text, openai_client, max_tokens=8192, max_chars=16000):
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    text = encoder.decode(tokens)
    if len(text) > max_chars:
        text = text[:max_chars]
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(response.data[0].embedding, dtype=np.float32)

# ------------------------------------------------------
# Computes cosine similarity between two vectors.
# Returns a float representing similarity.
# ------------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ------------------------------------------------------
# Splits text into chunks of tokens with optional overlap.
# Useful for breaking up large documents.
# ------------------------------------------------------
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

# ------------------------------------------------------
# Queries an Astra vector collection using a REST API call.
# Returns a list of documents sorted by similarity.
# ------------------------------------------------------
def query_astra_vectors_rest(collection_name, endpoint_url, token, embedding, top_k=5):
    url = f"{endpoint_url}/api/json/v1/{collection_name}/vector-search"
    headers = {
        "x-cassandra-token": token,
        "Content-Type": "application/json"
    }
    payload = {
        "vector": embedding.tolist(),
        "limit": top_k
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("data", {}).get("documents", [])
    else:
        return []

# ------------------------------------------------------
# Appends a skipped summary to a CSV log file.
# Used to track low-relevance documents.
# ------------------------------------------------------
def log_skipped_summary(log_entry):
    log_file = "skipped_summaries.csv"
    try:
        existing = pd.read_csv(log_file)
    except FileNotFoundError:
        existing = pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([log_entry])], ignore_index=True)
    updated.to_csv(log_file, index=False)

# ------------------------------------------------------
# Extracts text from a PDF file.
# ------------------------------------------------------
def extract_text_from_pdf(file):
    return extract_text(BytesIO(file.read()))

# ------------------------------------------------------
# Extracts text from all PDFs inside a ZIP file.
# ------------------------------------------------------
def extract_text_from_zip(file):
    with zipfile.ZipFile(file) as z:
        return "\n\n".join([
            extract_text(BytesIO(z.read(n))) for n in z.namelist() if n.lower().endswith(".pdf")
        ])

# ------------------------------------------------------
# Fetches the list of persona names from the profile collection.
# This powers the dropdown in the Streamlit sidebar.
# ------------------------------------------------------
def fetch_persona_names(endpoint_url, token, collection_name="profile_collection", top_k=50):
    url = f"{endpoint_url}/api/json/v1/{collection_name}/find"
    headers = {
        "x-cassandra-token": token,
        "Content-Type": "application/json"
    }
    payload = {
        "options": {"limit": top_k}
    }
    response = requests.post(url, headers=headers, json=payload)
    docs = response.json().get("data", {}).get("documents", [])
    return sorted({
        doc.get("metadata", {}).get("persona")
        for doc in docs
        if doc.get("metadata", {}).get("persona")
    })

# ------------------------------------------------------
# Retrieves the vector for a given persona by name.
# Assumes the vector is already embedded and stored under "$vector".
# ------------------------------------------------------
def fetch_persona_vector(persona_name, endpoint_url, token, collection_name="profile_collection"):
    url = f"{endpoint_url}/api/json/v1/{collection_name}/find"
    headers = {
        "x-cassandra-token": token,
        "Content-Type": "application/json"
    }
    payload = {
        "filter": {
            "metadata.persona": {"$eq": persona_name}
        },
        "options": {
            "limit": 1
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    doc = response.json().get("data", {}).get("documents", [{}])[0]
    if "$vector" in doc:
        return np.array(doc["$vector"], dtype=np.float32)
    return np.zeros(1536, dtype=np.float32)

# ------------------------------------------------------
# Uses an LLM to extract top 10 cybersecurity keywords from text.
# Used to create a compact, focused vector representation.
# ------------------------------------------------------
def extract_keywords_from_text(text, openai_client):
    system_prompt = "Extract the top 10 technical cybersecurity keywords, concepts, or entities from this document. Return them as a single comma-separated string."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""
