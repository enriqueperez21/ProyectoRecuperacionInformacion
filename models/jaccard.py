from models.base import BaseModel
from index.inverted_index import InvertedIndex
from preprocessing import preprocess_query


class JaccardModel(BaseModel):
    """
    Modelo de recuperación usando similitud de Jaccard.

    Jaccard = | intersección(query, doc) | / | unión(query, doc) |

    Trata cada documento como un CONJUNTO de palabras (sin importar frecuencia).
    Es el modelo más simple de todos.
    """

    def __init__(self, index: InvertedIndex):
        self.index = index

    @property
    def name(self) -> str:
        return "Jaccard"

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_tokens = preprocess_query(query)
        query_set = set(query_tokens)

        if not query_set:
            return []

        # Solo buscar entre documentos candidatos (contienen al menos un término)
        candidates = self.index.get_candidates(query_tokens)

        scores = []
        for doc_id in candidates:
            doc_set = set(self.index.doc_tokens[doc_id])

            intersection = query_set & doc_set   # palabras en común
            union = query_set | doc_set          # todas las palabras

            if not union:
                continue

            score = len(intersection) / len(union)
            scores.append((doc_id, score))

        # Ordenar de mayor a menor score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]