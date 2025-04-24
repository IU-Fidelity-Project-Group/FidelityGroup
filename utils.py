# utils.py

import zipfile
import openai
import numpy as np
import pandas as pd
import requests
import tiktoken
from io import BytesIO, StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from astrapy import DataAPIClient

# ------------------------------
# Initialize tokenizer for consistent token handling with OpenAI models
# ------------------------------
encoder = tiktoken.encoding_for_model("gpt-4o")

# ------------------------------
# Generate an OpenAI embedding vector for a given text.
# Truncates input to token and character limits for model compatibility.
# ------------------------------
def get_embedding(text, openai_client, max_tokens=8192, max_chars=16000):
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    text = encoder.decode(tokens)
    if len(text) > max_chars:
        text = text[:max_chars]
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(response.data[0].embedding, dtype=np.float32)

# ------------------------------
# Compute cosine similarity between two vectors.
# Measures how aligned the document content is with the persona.
# ------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ------------------------------
# Chunk text based on token count with optional overlap.
# Supports long document splitting for LLM summarization.
# ------------------------------
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

# ------------------------------
# Query Astra DB via REST for vector similarity search.
# Sends vector and retrieves top-k similar documents.
# ------------------------------
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

# ------------------------------
# Log documents skipped from summarization to CSV.
# Used for diagnostics or manual reprocessing.
# ------------------------------
def log_skipped_summary(log_entry):
    log_file = "skipped_summaries.csv"
    try:
        existing = pd.read_csv(log_file)
    except FileNotFoundError:
        existing = pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([log_entry])], ignore_index=True)
    updated.to_csv(log_file, index=False)

# ------------------------------
# Extract structured text from a PDF file.
# Applies LAParams for table and layout preservation.
# ------------------------------
def extract_text_from_pdf(file):
    output_string = StringIO()
    laparams = LAParams(
        line_overlap=0.5,
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        boxes_flow=0.5,
        all_texts=True
    )
    extract_text_to_fp(BytesIO(file.read()), output_string, laparams=laparams)
    return output_string.getvalue()

# ------------------------------
# Extract structured text from each PDF in a ZIP archive.
# Returns the concatenated contents from all PDF files.
# ------------------------------
def extract_text_from_zip(file):
    with zipfile.ZipFile(file) as z:
        return "\n\n".join([
            extract_text_from_pdf(BytesIO(z.read(n)))
            for n in z.namelist() if n.lower().endswith(".pdf")
        ])

# ------------------------------
# Retrieve all persona labels from Astra DB profile collection.
# Dynamically populates dropdown menu in Streamlit UI.
# ------------------------------
def fetch_persona_names(endpoint_url, token, collection_name="profile_collection"):
    client = DataAPIClient(token)
    db = client.get_database_by_api_endpoint(endpoint_url)
    collection = db.get_collection(collection_name)

    try:
        docs = collection.find()
        return sorted({
            doc.get("metadata", {}).get("persona")
            for doc in docs
            if doc.get("metadata", {}).get("persona")
        })
    except Exception as e:
        print(f"⚠️ Error fetching persona names: {e}")
        return []

# ------------------------------
# Fetch the precomputed vector for a given persona.
# Used for cosine similarity with document embeddings.
# ------------------------------
def fetch_persona_vector(persona_name, endpoint_url, token, collection_name="profile_collection"):
    client = DataAPIClient(token)
    db = client.get_database_by_api_endpoint(endpoint_url)
    collection = db.get_collection(collection_name)

    try:
        result = collection.find_one(
            {"metadata.persona": persona_name},
            projection={"$vector": True}
        )
        if result and "$vector" in result:
            return np.array(result["$vector"], dtype=np.float32)
        else:
            print("⚠️ No $vector field found in the document.")
            return np.zeros(1536, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Error fetching persona vector: {e}")
        return np.zeros(1536, dtype=np.float32)

# ------------------------------
# Use OpenAI LLM to extract top 20 technical keywords from a document.
# Keywords are comma-separated for embedding or context generation.
# ------------------------------
def extract_keywords_from_text(text, openai_client):
    system_prompt = (
    "You are an expert cybersecurity analyst. Your task is to extract up to 20 technical keywords, "
    "concepts, or entities strictly related to cybersecurity (e.g., threat actors, vulnerabilities, "
    "network protocols, security tools). Only provide keywords if the content is clearly relevant "
    "to cybersecurity. If the content is unrelated (e.g., games, marketing, legal, finance), return an empty string. "
    "Avoid hallucination or general tech terms not tied to cybersecurity."
    "Return as a single, comma-separated string of unique terms. Avoid general or vague terms."
    )

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
