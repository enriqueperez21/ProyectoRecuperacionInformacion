import math
from models.base import BaseModel
from index.inverted_index import InvertedIndex
from preprocessing import preprocess_query


class BM25Model(BaseModel):
    """
    Modelo BM25 (Best Match 25) — el estándar de la industria.

    Mejora TF-IDF de dos formas:
    1. Satura el TF: si una palabra aparece 100 veces no vale 100x más que si aparece 10
    2. Penaliza documentos largos: un doc largo que repite palabras no es más relevante

    Parámetros estándar:
        k1 = 1.5  → controla la saturación del TF
        b  = 0.75 → controla cuánto penalizar la longitud del doc
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

        # Longitud promedio de documento en el corpus
        if index.doc_lengths:
            self.avgdl = sum(index.doc_lengths.values()) / len(index.doc_lengths)
        else:
            self.avgdl = 1.0

    @property
    def name(self) -> str:
        return "BM25"

    def _idf(self, term: str) -> float:
        """IDF suavizado para BM25."""
        N  = self.index.num_docs
        df = len(self.index.get_docs(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_tokens = preprocess_query(query)

        if not query_tokens:
            return []

        candidates = self.index.get_candidates(query_tokens)

        scores = []
        for doc_id in candidates:
            doc_len = self.index.doc_lengths[doc_id]
            score = 0.0

            for term in set(query_tokens):
                tf = self.index.get_docs(term).get(doc_id, 0)

                if tf == 0:
                    continue

                idf = self._idf(term)

                # Fórmula BM25
                numerator   = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (numerator / denominator)

            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]