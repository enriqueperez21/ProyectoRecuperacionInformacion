from corpus.loader import load_corpus
from preprocessing import preprocess
from index.inverted_index import InvertedIndex
from index.tfidf import compute_tfidf
from models.jaccard import JaccardModel
from models.tfidf_cosine import TFIDFCosineModel
from models.bm25 import BM25Model
from models.semantic import SemanticModel


class SearchEngine:
    """
    Motor de búsqueda principal.

    Orquesta todo el pipeline:
        corpus → preprocessing → index → modelos → resultados

    Uso:
        engine = SearchEngine()
        engine.build()
        results = engine.search("coffee exports", model="bm25")
    """

    MODELS = ["jaccard", "tfidf", "bm25", "semantic"]

    def __init__(self, split: str = "train"):
        self.split   = split
        self.corpus  = {}       # { doc_id: {title, body, text, topics} }
        self.index   = None
        self.tfidf   = None

        # Los 4 modelos
        self.jaccard  = None
        self.tfidf_model = None
        self.bm25     = None
        self.semantic = None

        self._built = False

    def build(self, use_semantic: bool = True) -> None:
        """
        Construye todo el pipeline. Se llama una sola vez al iniciar la app.
        Puede tardar 1-3 minutos la primera vez (embeddings).
        """
        print("=" * 50)
        print("[engine] Iniciando pipeline de indexación...")
        print("=" * 50)

        # 1. Cargar corpus
        self.corpus = load_corpus(self.split)

        # 2. Construir índice invertido
        self.index = InvertedIndex()
        self.index.build(self.corpus)

        # 3. Calcular TF-IDF
        self.tfidf = compute_tfidf(self.index)

        # 4. Inicializar modelos clásicos
        self.jaccard     = JaccardModel(self.index)
        self.tfidf_model = TFIDFCosineModel(self.index, self.tfidf)
        self.bm25        = BM25Model(self.index)

        # 5. Modelo semántico (opcional, es el más lento)
        if use_semantic:
            self.semantic = SemanticModel()
            self.semantic.build(self.corpus)

        self._built = True
        print("=" * 50)
        print("[engine] Sistema listo para búsquedas")
        print("=" * 50)

    def search(
        self,
        query: str,
        model: str = "bm25",
        top_k: int = 10,
    ) -> list[dict]:
        """
        Ejecuta una búsqueda y devuelve resultados enriquecidos.

        model: "jaccard" | "tfidf" | "bm25" | "semantic"

        Retorna lista de dicts:
            [{ "rank": 1, "doc_id": "...", "score": 0.91,
               "title": "...", "body": "...", "topics": [...] }]
        """
        if not self._built:
            raise RuntimeError("Llama a build() primero")

        # Seleccionar modelo
        model_map = {
            "jaccard":  self.jaccard,
            "tfidf":    self.tfidf_model,
            "bm25":     self.bm25,
            "semantic": self.semantic,
        }

        if model not in model_map:
            raise ValueError(f"Modelo '{model}' no existe. Opciones: {self.MODELS}")

        selected = model_map[model]
        if selected is None:
            raise RuntimeError(f"Modelo '{model}' no fue inicializado")

        # Ejecutar búsqueda
        raw_results = selected.search(query, top_k=top_k)

        # Enriquecer con metadata del corpus
        results = []
        for rank, (doc_id, score) in enumerate(raw_results, start=1):
            doc = self.corpus.get(doc_id, {})
            results.append({
                "rank":   rank,
                "doc_id": doc_id,
                "score":  round(score, 4),
                "title":  doc.get("title", "Sin título"),
                "body":   doc.get("body", "")[:300] + "...",  # preview de 300 chars
                "topics": doc.get("topics", []),
            })

        return results

    def search_all_models(self, query: str, top_k: int = 10) -> dict[str, list[dict]]:
        """
        Corre la misma query en los 4 modelos y devuelve todos los resultados.
        Útil para la pantalla de comparación.
        """
        results = {}
        for model in self.MODELS:
            try:
                results[model] = self.search(query, model=model, top_k=top_k)
            except Exception as e:
                results[model] = []
                print(f"[engine] Error en modelo '{model}': {e}")
        return results