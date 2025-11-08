# PubMedQA LLM Starter (CIS660)

This starter kit helps you build **Medical Question Answering** using **PubMedQA** and **BioGPT / PubMedBERT**, aligned to Prof. Chung's requirements.

## What you will build
1) **Baseline** (fast): TF–IDF + Logistic Regression (predicts _yes/no/maybe_).  
2) **LLM Pass** (domain model): BioGPT generation on top‐k retrieved context.  
3) **Demo App**: Gradio web UI you can share from Colab.

## Quick start (Google Colab)
1. Open `notebooks/01_load_data_and_baseline.ipynb` in Colab → Run all.  
2. Open `notebooks/02_biogpt_prompting.ipynb` → Run all (uses `microsoft/biogpt`).  
3. Open `notebooks/03_gradio_app.ipynb` → Run all → Get a shareable link for the QA bot.

## Dataset
- Hugging Face: `pubmed_qa` (config: `pqa_labeled`)
- Automatically downloaded via `datasets` library.

## Environment (pip)
See `requirements.txt` for exact versions.

## GitHub workflow (suggested)
```bash
# in a terminal (local) OR inside Colab's terminal
git init
git branch -M main
git add .
git commit -m "Initial commit: PubMedQA LLM starter"
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```
From Colab: **File → Save a copy in GitHub** to push notebooks.

## Project structure
```
pubmedqa_llm_starter/
├─ notebooks/
│  ├─ 01_load_data_and_baseline.ipynb
│  ├─ 02_biogpt_prompting.ipynb
│  └─ 03_gradio_app.ipynb
├─ src/
│  ├─ retrieval.py
│  └─ utils.py
├─ app/
│  └─ templates.txt
├─ data/               # (empty; datasets lib caches automatically elsewhere)
├─ requirements.txt
└─ README.md
```

---

## Evaluation ideas (for your report)
- Baseline accuracy (majority class vs TF–IDF LR) on validation set.
- LLM prompting accuracy on a small subset (e.g., 200 examples).
- Error analysis (where LLM says "Yes" but gold is "No", etc.).
- Ablations: with/without retrieval; different prompt templates.

