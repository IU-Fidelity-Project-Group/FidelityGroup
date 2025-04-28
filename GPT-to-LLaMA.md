# Migrating from OpenAI to LLaMA-based Model: Roadmap

This document outlines the key considerations and steps required to migrate the current cybersecurity summarization application from OpenAI's GPT API to a LLaMA-based language model, such as LLaMA 2 or 3, hosted locally or via an inference API (e.g., Hugging Face, Ollama).

---

## 1. Replace the OpenAI API Client

### Current Setup
- The app currently uses the `OpenAI` client to access chat completions and embedding models.

### Migration Task
- Replace OpenAI API usage with a LLaMA-compatible inference backend.
- Choose from:
  - Hugging Face Transformers (local or HuggingFace Inference API)
  - Ollama (local inference)
  - LM Studio or other REST-capable inference servers

---

## 2. Modify Prompt Formatting

### Current Format
- OpenAI uses structured messages: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`

### Migration Task
- Convert to plain-text concatenated prompts:
  - Include both system and user roles in a single formatted prompt string.
  - Use newlines and context headers (e.g., `User:`, `System:`) to maintain clarity.

---

## 3. Replace the Embedding Model

### Current Setup
- OpenAI's `text-embedding-3-small` is used for all vector embeddings.

### Migration Task
- Swap out OpenAI embeddings for local embedding models.
- Use `sentence-transformers` such as `all-MiniLM-L6-v2` or a more cybersecurity-specialized model.
- Ensure embeddings return NumPy arrays compatible with the current vector search logic.

---

## 4. Adjust the Summarization Pipeline

### Current Flow
- Summary generation is performed chunk-by-chunk using OpenAI chat.

### Migration Task
- Use local generation via LLaMA with manual prompt construction.
- Ensure that summarization logic still respects token count and response limits.

---

## 5. Update Dependencies

### Migration Task
- Install required packages:
  - `transformers`
  - `sentence-transformers`
  - `torch` (for local GPU inference)
  - Optional: `accelerate` for efficient distributed inference

---

## 6. Update Environment & Hardware Requirements

### Considerations
- LLaMA-based models are large and require GPU memory (min 8–16 GB for small models).
- If deploying locally, ensure a machine with CUDA-compatible GPU and appropriate memory.

---

## 7. Abstract the Model Provider

### Recommendation
- Refactor summarization and embedding functions into a modular `LLMClient` class.
- Switch between providers using a simple config flag, e.g., `USE_OPENAI = False`

---

## 8. Glossary & Persona Context Use

### Migration Task
- Ensure glossary and persona metadata are integrated into the constructed prompt string.
- Reuse current persona formatting with updated LLaMA-compatible instruction format.

---

## 9. Test and Evaluate

### Final Task
- Test with various personas and cybersecurity documents.
- Evaluate:
  - Summary quality
  - Relevance scoring
  - Speed and token cost (if using paid inference)

---

## Optional: Run via Inference APIs
- Hugging Face Inference Endpoints (managed)
- Replicate.com
- Modal or Banana for serverless GPU apps

