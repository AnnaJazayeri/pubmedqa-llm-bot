import os
import re
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import gdown

# Streamlit basic config
st.set_page_config(page_title="PubMedQA Biomedical QA", page_icon="🧬", layout="wide")
st.title("PubMedQA – Biomedical Question Answering")
st.write(
    "Ask a biomedical question. The app retrieves PubMed-style abstracts "
    "with a dual-encoder and then generates a simple answer using GPT-4o-mini, "
    "grounded in the retrieved evidence."
)

# Google Drive file IDs
FILE_ID_CONTEXT = "1e4BwDZaqNPe-8i3KHZskvIlVIUdRb6Fb"
FILE_ID_PARQUET = "1zx4RAR_csv4sutBDg2RuNcWH-UxR1WoB"
FILE_ID_MODEL = "1RNPiTK52eXe47WyKANLxMZ1OiAtS4eo5"

def download_from_gdrive(file_id: str, filename: str):
    if os.path.exists(filename):
        return
    with st.spinner(f"Downloading {filename} ..."):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        st.success(f"{filename} downloaded ({size_mb:.1f} MB).")

# Download artifacts if missing
download_from_gdrive(FILE_ID_CONTEXT, "context_embs_pubmedqa.npy")
download_from_gdrive(FILE_ID_PARQUET, "df_all_pubmedqa.parquet")
download_from_gdrive(FILE_ID_MODEL, "dual_encoder_pubmedqa.pt")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data and embeddings
@st.cache_resource(show_spinner=True)
def load_data():
    df_all_local = pd.read_parquet("df_all_pubmedqa.parquet", engine="pyarrow")
    context_embs_local = np.load("context_embs_pubmedqa.npy")
    return df_all_local, context_embs_local

df_all, context_embs = load_data()

# DualEncoder model
max_length = 128
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class DualEncoder(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state
        cls_emb = token_embs[:, 0, :]
        cls_emb = cls_emb / cls_emb.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return cls_emb

    def forward(self, q_input_ids, q_attention_mask, c_input_ids, c_attention_mask):
        q_emb = self.encode(q_input_ids, q_attention_mask)
        c_emb = self.encode(c_input_ids, c_attention_mask)
        return q_emb, c_emb

@st.cache_resource(show_spinner=True)
def load_dual_encoder():
    model = DualEncoder(model_name).to(device)
    state_dict = torch.load("dual_encoder_pubmedqa.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

dual_encoder = load_dual_encoder()

# OpenAI client using secrets / env / key.txt
def load_openai_client():
    api_key = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        elif "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            api_key = st.secrets["openai"]["api_key"]
    except Exception:
        api_key = None
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", None)
    if api_key is None and os.path.exists("key.txt"):
        with open("key.txt", "r") as f:
            api_key = f.read().strip()
    if api_key is None:
        st.error(
            "OpenAI API key not found. Set it in Streamlit secrets as "
            "`OPENAI_API_KEY`, or in the environment, or in a local key.txt file."
        )
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key)

client = load_openai_client()

# Retrieval function
def retrieve_topk_docs(question, top_k_docs=3):
    dual_encoder.eval()
    enc = tokenizer(
        question,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        q_emb = dual_encoder.encode(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device)
        ).cpu().numpy()[0]
    sims = context_embs @ q_emb
    ranked_indices = np.argsort(-sims)
    docs = []
    for rank_pos in range(top_k_docs):
        cid = ranked_indices[rank_pos]
        score = sims[cid]
        ctx = df_all.loc[df_all["id"] == cid, "context"].values[0]
        docs.append((score, cid, ctx))
    return docs

# Semi-strict GPT-4o-mini answer generator
def generate_plain_answer_gpt4o(question, docs, max_chars_per_doc=400):
    snippets = []
    for score, cid, ctx in docs:
        snippets.append(ctx[:max_chars_per_doc])
    evidence_text = "\n\n".join(snippets)
    prompt = f"""
You are a biomedical assistant.
You must answer using only the evidence below.

You are allowed to:
- infer likely meaning if multiple snippets point in the same direction,
- paraphrase or summarize what the evidence implies,
- generalize a little only if the evidence strongly suggests it.

You are not allowed to:
- use outside medical knowledge,
- add unsupported facts,
- contradict the evidence.

Question:
{question}

Evidence:
{evidence_text}

Output rules:
1. Start with exactly one of:
   - Short answer: Yes.
   - Short answer: No.
   - Short answer: It leans toward yes.
   - Short answer: It leans toward no.
   - Short answer: Unclear.

2. Then add 1–2 simple sentences summarizing what the evidence suggests.
3. If evidence is indirect, incomplete, or off-topic, choose "Short answer: Unclear."
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

# Streamlit UI
st.subheader("Ask a biomedical question")
question = st.text_input(
    "Type your question here:",
    placeholder="e.g., Does smoking increase the risk of heart attack?"
)
top_k = st.slider("Number of evidence documents to show:", 1, 5, 2)
run_button = st.button("Run QA")

if run_button and question.strip():
    with st.spinner("Retrieving relevant articles and generating answer..."):
        docs = retrieve_topk_docs(question, top_k_docs=top_k)
        answer = generate_plain_answer_gpt4o(question, docs)
    st.markdown("### Short Answer")
    st.write(answer)
    st.markdown("### Evidence Snippets from Retrieved Articles")
    for rank_pos, (score, cid, ctx) in enumerate(docs, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", ctx)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        snippet = " ".join(sentences[:2])[:400] + "..."
        st.markdown(f"**Document {rank_pos}** (similarity: `{score:.3f}`, id: `{cid}`)")
        st.write(snippet)
