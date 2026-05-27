import math
from models.base import BaseModel
from index.inverted_index import InvertedIndex
from preprocessing import preprocess_query


class TFIDFCosineModel(BaseModel):
    """
    Modelo de recuperación usando TF-IDF con similitud coseno.

    Cada documento y cada query se representan como vectores de pesos TF-IDF.
    La similitud se mide con el coseno del ángulo entre los vectores.

    coseno = (query · doc) / (|query| × |doc|)
    """

    def __init__(self, index: InvertedIndex, tfidf: dict[str, dict[str, float]]):
        self.index = index
        self.tfidf = tfidf

    @property
    def name(self) -> str:
        return "TF-IDF Coseno"

    def _vector_norm(self, vector: dict[str, float]) -> float:
        """Calcula la magnitud (longitud) de un vector."""
        return math.sqrt(sum(v ** 2 for v in vector.values()))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_tokens = preprocess_query(query)

        if not query_tokens:
            return []

        # Calcular vector TF-IDF de la query
        N = self.index.num_docs
        query_vec: dict[str, float] = {}

        for token in set(query_tokens):
            tf = query_tokens.count(token) / len(query_tokens)
            df = len(self.tfidf.get(token, {}))
            idf = math.log((N + 1) / (df + 1))
            query_vec[token] = tf * idf

        query_norm = self._vector_norm(query_vec)
        if query_norm == 0:
            return []

        candidates = self.index.get_candidates(query_tokens)

        scores = []
        for doc_id in candidates:
            dot_product = 0.0
            doc_vec_sq_sum = 0.0

            for term, q_weight in query_vec.items():
                d_weight = self.tfidf.get(term, {}).get(doc_id, 0.0)
                dot_product += q_weight * d_weight

            # Norma del documento usando todos sus términos TF-IDF
            for term in self.index.doc_tokens[doc_id]:
                d_weight = self.tfidf.get(term, {}).get(doc_id, 0.0)
                doc_vec_sq_sum += d_weight ** 2

            doc_norm = math.sqrt(doc_vec_sq_sum)

            if doc_norm == 0:
                continue

            score = dot_product / (query_norm * doc_norm)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]