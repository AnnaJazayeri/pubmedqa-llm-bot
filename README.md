# PubMedQA RAG Question-Answering Bot

This project is a biomedical question-answering system built on the **PubMedQA** dataset.  
It combines:

- A **dual-encoder retriever** trained on PubMedQA (labeled + unlabeled + artificial),
- A **vector index** over PubMed article contexts,
- A **GPT-4o Mini generator** that produces simple, layperson-friendly answers,
- A small **Streamlit web app** so users can type questions and see answers plus evidence.

The system is designed as a **Retrieval-Augmented Generation (RAG)** pipeline, and it does **not** rely on the original yes/no/maybe labels from PubMedQA during inference.

---

## 1. Problem Statement

The goal is to build a question-answering system that can answer biomedical questions by:
- Retrieving relevant PubMed-style article abstracts/contexts, and
- Generating short, easy-to-understand answers for non-expert users.

A key constraint is **limited labeled data** (only ~1,000 labeled PubMedQA samples), so the project focuses on using the **large unlabeled corpus (60k+)** with a **self-supervised dual-encoder** instead of purely supervised classification.

---

## 2. Methodology Overview

### 2.1 Data

- **Dataset:** `pubmed_qa` (HuggingFace)
  - `pqa_labeled`: 1,000 labeled Q–A triplets
  - `pqa_unlabeled`: ~60k question–context pairs
  - `pqa_artificial`: additional generated pairs

For this RAG system, we use all three splits but **ignore the yes/no/maybe labels** and only keep:

- `question`
- `context.contexts` (merged into a single text field)
- a global integer `id`

These are stored in a single DataFrame (`df_all_pubmedqa.parquet`) and a corresponding embedding matrix (`context_embs_pubmedqa.npy`).

### 2.2 Dual-Encoder Retriever

- Base encoder: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Architecture: shared **dual encoder** (same encoder for questions and contexts).
- Training objective:
  - **Contrastive InfoNCE loss** with in-batch negatives.
  - For each batch, the true question–context pair is a positive, all other contexts in the batch are negatives.
- Evaluation:
  - **Self-retrieval Recall@k and MRR**:
    - Recall@1 ≈ 0.79  
    - Recall@5 ≈ 0.93  
    - Recall@10 ≈ 0.95  
  - This means ~79% of the time, the model retrieves the correct context as the top result, and ~95% within the top 10.

The trained context embeddings are saved as `context_embs_pubmedqa.npy`.

### 2.3 RAG-Style Generator (GPT-4o Mini)

For answer generation, we use **OpenAI GPT-4o Mini** as a small, instruction-following model.  
It is **not fine-tuned** on this project; instead, we prompt it in a **semi-strict grounded mode**:

- The model receives:
  - The user’s question,
  - Top-k retrieved PubMed contexts (snippets),
  - A strict prompt that:
    - Forces the answer to start with one of:
      - `Short answer: Yes.`
      - `Short answer: No.`
      - `Short answer: It leans toward yes.`
      - `Short answer: It leans toward no.`
      - `Short answer: Unclear.`
    - Requires 1–2 simple explanatory sentences.
    - Forbids using outside knowledge beyond the retrieved evidence.

This gives **interpretable, grounded answers with explicit uncertainty** when evidence is weak.

---

## 3. App Structure

Main files:

- `app.py` – Streamlit app entry point (UI + wiring)
- `dual_encoder_pubmedqa.pt` – fine-tuned dual-encoder weights
- `df_all_pubmedqa.parquet` – all question–context pairs with IDs
- `context_embs_pubmedqa.npy` – pre-computed context embeddings
- `requirements.txt` – Python dependencies
- `README.md` – this document

Key components inside `app.py`:

- `DualEncoder` class – loads PubMedBERT and encodes texts into vectors.
- `retrieve_topk_docs(question, k)` – retrieves top-k contexts using cosine similarity in embedding space.
- `generate_plain_answer_gpt4o(question, docs)` – sends question + evidence to GPT-4o Mini and returns a layperson-friendly answer.
- Simple Streamlit UI:
  - Text input for the question,
  - A button to run QA,
  - Display of:
    - Short answer,
    - Evidence snippets from PubMed contexts.

---

## 4. How to Run Locally

### 4.1 Prerequisites

- Python 3.9+ recommended
- An OpenAI API key with access to GPT-4o Mini

### 4.2 Installation

```bash
git clone https://github.com/YOUR_USERNAME/pubmedqa-rag-bot.git
cd pubmedqa-rag-bot

pip install -r requirements.txt
