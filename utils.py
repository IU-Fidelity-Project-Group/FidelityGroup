# utils.py

import openai
import numpy as np
import pandas as pd
import requests
import tiktoken
from io import BytesIO, StringIO
from pdfminer.high_level import extract_text_to_fp, extract_text
from pdfminer.layout import LAParams
import zipfile
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
# Applies a nonlinear squashing to penalize weak similarity and reduce false midrange scores.
# ------------------------------
def safe_cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    raw = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    adjusted = (raw + 1) / 2
    penalized = adjusted ** 1.5
    return round(min(max(penalized, 0.0), 1.0), 4)

# ------------------------------
# Chunk text based on token count with optional overlap.
# Supports long document splitting for LLM summarization.
# ------------------------------
def chunk_text_by_tokens(text, chunk_size=3072, overlap=256):
    try:
        tokens = encoder.encode(text, allowed_special={"<|endoftext|>", "<|startoftext|>"})
    except ValueError:
        # Remove or escape problematic characters if they cause encoding issues
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        tokens = encoder.encode(text, allowed_special={"<|endoftext|>", "<|startoftext|>"})

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
def query_astra_vectors_rest(collection_name, endpoint_url, token, embedding, top_k=10):
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
            extract_text(BytesIO(z.read(n)), laparams=LAParams())
            for n in z.namelist() if n.lower().endswith(".pdf")
        ])

# ------------------------------
# Retrieve all persona labels from Astra DB profile collection using AstraPy client.
# Ensures reliable and consistent access to the persona list.
# ------------------------------
def fetch_persona_names(endpoint_url, token, collection_name="profile_collection"):
    client = DataAPIClient(token)
    db = client.get_database_by_api_endpoint(endpoint_url)
    collection = db.get_collection(collection_name)
    try:
        all_docs = collection.find()
        return sorted({doc["metadata"]["persona"] for doc in all_docs if "metadata" in doc and "persona" in doc["metadata"]})
    except Exception as e:
        print(f"Error fetching persona names: {e}")
        return []

# ------------------------------
# Fetch the precomputed vector for a given persona using AstraPy client.
# Returns the $vector as numpy array or zeros fallback.
# ------------------------------
def fetch_persona_vector(persona_name, endpoint_url, token, collection_name="profile_collection"):
    client = DataAPIClient(token)
    db = client.get_database_by_api_endpoint(endpoint_url)
    collection = db.get_collection(collection_name)
    try:
        result = collection.find_one({"metadata.persona": persona_name}, projection={"$vector": True})
        if result and "$vector" in result:
            return np.array(result["$vector"], dtype=np.float32)
        else:
            print("No $vector field found in the document.")
            return np.zeros(1536, dtype=np.float32)
    except Exception as e:
        print(f"Error fetching persona vector: {e}")
        return np.zeros(1536, dtype=np.float32)

# ------------------------------
# Fetch full metadata for a persona from Astra DB profile collection.
# Used for tailoring the LLM prompt formatting and summary tone.
# ------------------------------
def fetch_persona_metadata(persona_name, endpoint_url, token, collection_name="profile_collection"):
    client = DataAPIClient(token)
    db = client.get_database_by_api_endpoint(endpoint_url)
    collection = db.get_collection(collection_name)
    try:
        result = collection.find_one({"metadata.persona": persona_name}, projection={"metadata": True})
        if result and "metadata" in result:
            return result["metadata"]
    except Exception as e:
        print(f"Error fetching persona metadata: {e}")
    return {}

# ------------------------------
# Use OpenAI LLM to extract up to 20 highly relevant cybersecurity keywords.
# If the document is unrelated to cybersecurity, return an empty string.
# Prevents hallucination and enforces strict contextual filtering.
# ------------------------------
def extract_keywords_from_text(text, openai_client):
    system_prompt = (
        "You are an expert cybersecurity analyst. Your task is to extract up to 20 technical keywords, "
        "concepts, or entities strictly related to cybersecurity (e.g., threat actors, vulnerabilities, "
        "network protocols, security tools). Only provide keywords if the content is clearly relevant "
        "to cybersecurity. If the content is unrelated (e.g., games, marketing, legal, finance), return an empty string. "
        "Avoid hallucination or general tech terms not tied to cybersecurity. "
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
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ""

# ------------------------------
# Query glossary collection with keyword embedding for relevant contextual hits.
# Returns concatenated glossary definitions for final summary prompt enhancement.
# ------------------------------
def fetch_glossary_context(keywords, openai_client, glossary_collection, glossary_endpoint, glossary_token, top_k=10):
    if not keywords.strip():
        return ""
    embedding = get_embedding(keywords, openai_client)
    hits = query_astra_vectors_rest(glossary_collection, glossary_endpoint, glossary_token, embedding, top_k=top_k)
    return "\n\n".join([d.get("text", "") for d in hits])
