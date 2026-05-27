from models.base import BaseModel
from preprocessing import preprocess_query


class SemanticModel(BaseModel):
    """
    Modelo semántico usando embeddings de sentence-transformers + ChromaDB.

    A diferencia de los otros modelos:
    - NO usa el índice invertido
    - Entiende sinónimos y significado, no solo palabras exactas
    - Usa ChromaDB como base de datos vectorial para búsqueda rápida
    """

    COLLECTION_NAME = "reuters_docs"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None       # se carga lazy (solo cuando se necesita)
        self._collection = None  # colección de ChromaDB

    @property
    def name(self) -> str:
        return "Embeddings Semánticos"

    def _load_model(self):
        """Carga el modelo de embeddings (solo la primera vez)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[semantic] Cargando modelo '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            print("[semantic] Modelo listo")

    def build(self, corpus: dict[str, dict]) -> None:
        """
        Genera embeddings para todos los documentos y los guarda en ChromaDB.
        Solo se corre una vez — ChromaDB persiste en disco.
        """
        import chromadb
        self._load_model()

        client = chromadb.Client()

        # Si ya existe la colección, no reconstruir
        existing = [c.name for c in client.list_collections()]
        if self.COLLECTION_NAME in existing:
            self._collection = client.get_collection(self.COLLECTION_NAME)
            print(f"[semantic] Colección ya existe con {self._collection.count()} docs")
            return

        self._collection = client.create_collection(self.COLLECTION_NAME)

        print(f"[semantic] Generando embeddings para {len(corpus)} documentos...")

        # Procesar en lotes para no saturar memoria
        batch_size = 64
        ids    = list(corpus.keys())
        texts  = [corpus[doc_id]["text"][:512] for doc_id in ids]  # limitar longitud

        for i in range(0, len(ids), batch_size):
            batch_ids   = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            embeddings  = self._model.encode(batch_texts, show_progress_bar=False).tolist()

            self._collection.add(
                ids        = batch_ids,
                embeddings = embeddings,
                documents  = batch_texts,
            )

            if (i // batch_size) % 10 == 0:
                print(f"[semantic] Procesados {min(i + batch_size, len(ids))}/{len(ids)}")

        print("[semantic] Embeddings generados y guardados en ChromaDB")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._collection is None:
            raise RuntimeError("Debes llamar a build() antes de search()")

        self._load_model()

        # Generar embedding de la query
        query_embedding = self._model.encode([query]).tolist()

        # Buscar en ChromaDB
        results = self._collection.query(
            query_embeddings = query_embedding,
            n_results        = top_k,
        )

        # ChromaDB devuelve distancias, convertir a similitud (1 - distancia)
        doc_ids   = results["ids"][0]
        distances = results["distances"][0]

        scores = [(doc_id, 1 - dist) for doc_id, dist in zip(doc_ids, distances)]
        return scores