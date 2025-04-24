import openai
import numpy as np
import pandas as pd
import requests
import tiktoken
from io import BytesIO
from pdfminer.high_level import extract_text
import zipfile
from astrapy import DataAPIClient
import streamlit as st

encoder = tiktoken.encoding_for_model("gpt-4o")

def get_embedding(text, openai_client, max_tokens=8192, max_chars=16000):
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    text = encoder.decode(tokens)
    if len(text) > max_chars:
        text = text[:max_chars]
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(response.data[0].embedding, dtype=np.float32)

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

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

def log_skipped_summary(log_entry):
    log_file = "skipped_summaries.csv"
    try:
        existing = pd.read_csv(log_file)
    except FileNotFoundError:
        existing = pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([log_entry])], ignore_index=True)
    updated.to_csv(log_file, index=False)

def extract_text_from_pdf(file):
    return extract_text(BytesIO(file.read()))

def extract_text_from_zip(file):
    with zipfile.ZipFile(file) as z:
        return "\n\n".join([
            extract_text(BytesIO(z.read(n))) for n in z.namelist() if n.lower().endswith(".pdf")
        ])

def fetch_persona_names(endpoint_url, token, collection_name="profile_collection", top_k=50):
    dummy_vector = [0.0] * 1536
    url = f"{endpoint_url}/api/json/v1/{collection_name}/vector-search"
    headers = {
        "x-cassandra-token": token,
        "Content-Type": "application/json"
    }
    payload = {
        "vector": dummy_vector,
        "limit": top_k
    }
    response = requests.post(url, headers=headers, json=payload)
    docs = response.json().get("data", {}).get("documents", [])
    return sorted({
        doc.get("metadata", {}).get("persona")
        for doc in docs
        if doc.get("metadata", {}).get("persona")
    })

def fetch_persona_description(persona_name, endpoint_url, token, collection_name="profile_collection"):
    url = f"{endpoint_url}/api/json/v1/{collection_name}/find"
    headers = {
        "x-cassandra-token": token,
        "Content-Type": "application/json"
    }
    payload = {
        "filter": {
            "metadata.persona": { "$eq": persona_name }
        },
        "options": {
            "limit": 1
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    docs = response.json().get("data", {}).get("documents", [])
    if docs and "metadata" in docs[0]:
        return docs[0]["metadata"].get("description", "")
    return ""

def fetch_persona_vector(persona_name, endpoint_url, token, collection_name="profile_collection"):
    import streamlit as st

    # Construct correct Document API URL
    url = f"{endpoint_url}/api/json/v1/{collection_name}/find"
    headers = {
        "x-cassandra-token": token,
        "Content-Type": "application/json"
    }

    payload = {
        "filter": {
            "metadata.persona": { "$eq": persona_name }
        },
        "options": {
            "limit": 1,
            "includeFields": ["$vector"]
        }
    }

    st.write("🧭 Fetching persona vector from:", url)
    st.json(payload)

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        st.write("📦 Raw persona vector response:", data)

        docs = data.get("data", {}).get("documents", [])
        if docs and "$vector" in docs[0]:
            return np.array(docs[0]["$vector"], dtype=np.float32)

        st.warning("No $vector found in document.")
    except Exception as e:
        st.error(f"Error fetching persona vector: {e}")

    return np.zeros(1536, dtype=np.float32)

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
    except Exception as e:
        return ""
