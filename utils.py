import openai
import numpy as np
import pandas as pd
import requests
import tiktoken
import time

encoder = tiktoken.encoding_for_model("gpt-4o")

def get_embedding(text, openai_client, max_tokens=8192, max_chars=16000, retries=3):
    if not text or not isinstance(text, str):
        raise ValueError("Input to get_embedding must be a non-empty string.")

    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        text = encoder.decode(tokens[:max_tokens])
    if len(text) > max_chars:
        text = text[:max_chars]

    for attempt in range(retries):
        try:
            response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
            return np.array(response.data[0].embedding, dtype=np.float32)
        except openai.RateLimitError:
            wait_time = (2 ** attempt) * 2
            time.sleep(wait_time)
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {str(e)}")

    raise RuntimeError("OpenAI embedding failed after multiple retries due to rate limits.")

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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
