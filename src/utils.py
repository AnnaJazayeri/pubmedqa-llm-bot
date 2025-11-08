LABELS = ["yes", "no", "maybe"]

def normalize_label(text: str) -> str:
    t = (text or "").strip().lower()
    for lab in LABELS:
        if t.startswith(lab):
            return lab
    # fallback: look for the first occurrence
    for lab in LABELS:
        if lab in t:
            return lab
    return "maybe"  # neutral fallback

PROMPT_TEMPLATE = """You are a biomedical Q&A assistant.
Answer strictly with one word: Yes, No, or Maybe.
Question: {question}
Context: {context}
Answer:"""
