from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleTfidfRetriever:
    def __init__(self, documents: List[str]):
        self.docs = documents
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        self.doc_mat = self.vectorizer.fit_transform(self.docs)

    def topk(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_mat)[0]
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    def fetch(self, idxs: List[int]) -> List[str]:
        return [self.docs[i] for i in idxs]
