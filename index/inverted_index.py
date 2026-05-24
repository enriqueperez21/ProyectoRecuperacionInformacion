from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocess


def _identity(tokens):
    """Función identidad — le decimos a sklearn que no tokenice, ya lo hicimos."""
    return tokens


class InvertedIndex:
    """
    Índice invertido construido con scikit-learn CountVectorizer.

    CountVectorizer hace el conteo de frecuencias de forma eficiente
    sobre todo el corpus de una sola vez (mucho más rápido que un loop manual).

    Estructura interna:
        index["coffee"] = {"doc_001": 3, "doc_004": 7}
    """

    def __init__(self):
        self.index: dict[str, dict[str, int]] = defaultdict(dict)
        self.doc_lengths: dict[str, int] = {}
        self.doc_tokens: dict[str, list[str]] = {}
        self.num_docs: int = 0
        self._vectorizer = None

    def build(self, corpus: dict[str, dict]) -> None:
        """
        Construye el índice usando CountVectorizer de sklearn.

        CountVectorizer convierte el corpus en una matriz:
            filas = documentos
            columnas = términos únicos
            valores = frecuencia del término en ese documento
        """
        print("[index] Preprocesando documentos...")

        doc_ids = list(corpus.keys())
        # Preprocesar cada documento y guardar los tokens
        tokenized = []
        for doc_id in doc_ids:
            tokens = preprocess(corpus[doc_id]["text"])
            self.doc_tokens[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)
            tokenized.append(tokens)

        print("[index] Construyendo índice con CountVectorizer...")

        # CountVectorizer con tokenizador personalizado (ya tenemos los tokens)
        self._vectorizer = CountVectorizer(
            analyzer   = "word",
            tokenizer  = _identity,      # no tokenizar de nuevo
            preprocessor = _identity,    # no preprocesar de nuevo
            token_pattern = None,        # ignorar el patrón por defecto
        )

        # Matriz dispersa (sparse): shape = (num_docs, num_terms)
        doc_term_matrix = self._vectorizer.fit_transform(tokenized)

        # Vocabulario: término → índice de columna
        vocab = self._vectorizer.vocabulary_

        # Convertir la matriz a nuestro índice invertido
        # doc_term_matrix[i, j] = frecuencia del término j en el doc i
        cx = doc_term_matrix.tocoo()  # formato COO para iterar fácil
        term_list = self._vectorizer.get_feature_names_out()

        for i, j, freq in zip(cx.row, cx.col, cx.data):
            if freq > 0:
                doc_id = doc_ids[i]
                term   = term_list[j]
                self.index[term][doc_id] = int(freq)

        self.num_docs = len(doc_ids)
        print(f"[index] Índice listo: {len(self.index)} términos únicos, {self.num_docs} documentos")

    def get_docs(self, term: str) -> dict[str, int]:
        """Devuelve los documentos que contienen el término."""
        return self.index.get(term.lower(), {})

    def get_candidates(self, query_tokens: list[str]) -> set[str]:
        """
        Devuelve todos los doc_ids que contienen al menos un término de la query.
        """
        candidates = set()
        for token in query_tokens:
            candidates.update(self.get_docs(token).keys())
        return candidates