
import os
import zipfile
import streamlit as st
import numpy as np
import faiss
import tiktoken
from PyPDF2 import PdfReader
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

persona_collection = persona_db["persona_vectors"]
glossary_collection = glossary_db["glossary_vectors"]

# --------------------
# Embedding
# --------------------
def get_embedding(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(response.data[0].embedding, dtype=np.float32)

# --------------------
# FAISS Similarity
# --------------------
def build_faiss_index(texts):
    dim = len(get_embedding("sample"))
    index = faiss.IndexFlatL2(dim)
    vectors = [get_embedding(t) for t in texts]
    index.add(np.array(vectors))
    return index, vectors

def match_persona(doc_text, persona_names, persona_texts):
    index, _ = build_faiss_index(persona_texts)
    doc_vec = get_embedding(doc_text)
    scores, idxs = index.search(np.array([doc_vec]), k=1)
    return persona_names[idxs[0][0]]

# --------------------
# Astra Vector Search
# --------------------
def query_astra_vectors(collection, embedding, top_k):
    result = collection.vector_find(vector=embedding, limit=top_k)
    return result["data"]["documents"]

# --------------------
# PDF Text Extraction
# --------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

def extract_text_from_zip(file):
    with zipfile.ZipFile(file) as z:
        return "\n\n".join([
            extract_text_from_pdf(z.open(n)) for n in z.namelist() if n.lower().endswith(".pdf")
        ])

# --------------------
# Streamlit App
# --------------------
st.set_page_config(page_title="Persona Summarizer", layout="wide")
st.title("Cybersecurity Persona-Based Summarizer")

# Fetch available personas from AstraDB (profile DB)
persona_docs = persona_collection.find()
persona_list = [doc["name"] for doc in persona_docs["data"]["documents"]]
persona = st.sidebar.selectbox("Select Persona", persona_list)

# Fetch full description for selected persona
persona_doc = persona_collection.find_one({"name": {"$eq": persona}})
persona_description = persona_doc["data"]["document"]["text"] if persona_doc["status"]["code"] == 200 else ""
uploaded_file = st.sidebar.file_uploader("Upload PDF or ZIP", type=["pdf", "zip"])
max_toks = st.sidebar.slider("Max tokens", 100, 2000, 500, 100)
generate = st.sidebar.button("Generate Summary")

if generate:
    if not uploaded_file:
        st.warning("Please upload a file.")
        st.stop()

    # Extract text
    raw_text = extract_text_from_zip(uploaded_file) if uploaded_file.name.endswith(".zip") else extract_text_from_pdf(uploaded_file)

    # Truncate for token safety
    def truncate(text, persona_text, max_toks):
        pre = "Document content:\n\n"
        post = f"\n\nPlease summarize for a {persona}."
        overhead = len(encoder.encode(persona_text + pre + post)) + max_toks + 50
        doc_tokens = encoder.encode(text)
        return encoder.decode(doc_tokens[:max(0, 30000 - overhead)])

    doc_text = truncate(raw_text, persona_description, max_toks)
    doc_embedding = get_embedding(doc_text)

    # Vector match glossary & persona content
    glossary_hits = query_astra_vectors(glossary_collection, doc_embedding, top_k=5)
    glossary_context = "\n\n".join([d.get("text", "") for d in glossary_hits])

    persona_hits = query_astra_vectors(persona_collection, doc_embedding, top_k=1)
    persona_context = "\n\n".join([d.get("text", "") for d in persona_hits])

    # Prompt
    system_msg = {"role": "system", "content": persona_description}
    user_msg = {"role": "user", "content": f"{persona_context}\n\n{glossary_context}\n\n{doc_text}\n\nPlease summarize for a {persona}."}

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[system_msg, user_msg],
            max_tokens=max_toks
        )
        st.subheader(uploaded_file.name)
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
