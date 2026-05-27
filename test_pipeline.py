"""
Script de prueba rápida del pipeline completo.
Corre esto desde la raíz del proyecto:

    python test_pipeline.py

Si todo está bien verás resultados de búsqueda de los 3 modelos clásicos.
El modelo semántico (embeddings) se omite aquí para que sea rápido.
"""
import sys
import os

# Asegurar que Python encuentre los módulos del proyecto
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("  TEST DEL PIPELINE - SRI Reuters")
print("=" * 55)

# ── 1. CORPUS ────────────────────────────────────────────────
print("\n[1/4] Cargando corpus...")
from corpus.loader import load_corpus

corpus = load_corpus("train")
assert len(corpus) > 0, "ERROR: corpus vacío"

# Mostrar un documento de ejemplo
sample_id = list(corpus.keys())[0]
sample    = corpus[sample_id]
print(f"      Ejemplo doc '{sample_id}':")
print(f"      Título:  {sample['title'][:60]}")
print(f"      Topics:  {sample['topics']}")
print(f"      Texto:   {sample['text'][:80]}...")
print(f"      ✓ {len(corpus)} documentos cargados")

# ── 2. PREPROCESSING ─────────────────────────────────────────
print("\n[2/4] Probando preprocesamiento...")
from preprocessing import preprocess, preprocess_query

test_text = "Brazil's COFFEE exports ROSE 12%, amid drought fears!"
tokens = preprocess(test_text)
print(f"      Entrada:  '{test_text}'")
print(f"      Salida:   {tokens}")
assert len(tokens) > 0, "ERROR: preprocessing devolvió lista vacía"
assert "coffee" in tokens or "COFFEE".lower() in tokens, "ERROR: 'coffee' no está en tokens"
print(f"      ✓ Preprocessing OK")

# ── 3. ÍNDICE INVERTIDO ──────────────────────────────────────
print("\n[3/4] Construyendo índice invertido...")
from index.inverted_index import InvertedIndex
from index.tfidf import compute_tfidf

index = InvertedIndex()
index.build(corpus)

# Verificar que el índice tiene contenido
assert index.num_docs > 0,    "ERROR: índice sin documentos"
assert len(index.index) > 0,  "ERROR: índice sin términos"

# Buscar un término común de Reuters
term = "cocoa"
docs_with_term = index.get_docs(term)
print(f"      Término '{term}' aparece en {len(docs_with_term)} documentos")
assert len(docs_with_term) > 0, f"ERROR: '{term}' no encontrado en el índice"

tfidf = compute_tfidf(index)
assert len(tfidf) > 0, "ERROR: TF-IDF vacío"
print(f"      ✓ Índice OK — {len(index.index)} términos únicos")

# ── 4. MODELOS ───────────────────────────────────────────────
print("\n[4/4] Probando los 3 modelos clásicos...")

from models.jaccard     import JaccardModel
from models.tfidf_cosine import TFIDFCosineModel
from models.bm25        import BM25Model

jaccard  = JaccardModel(index)
tfidf_m  = TFIDFCosineModel(index, tfidf)
bm25     = BM25Model(index)

QUERY = "coffee trade export"
print(f"\n      Query: '{QUERY}'")
print(f"      {'─'*50}")

for model in [jaccard, tfidf_m, bm25]:
    results = model.search(QUERY, top_k=3)
    assert len(results) > 0, f"ERROR: {model.name} no devolvió resultados"
    print(f"\n      [{model.name}] Top 3:")
    for doc_id, score in results:
        title = corpus[doc_id]["title"][:45]
        print(f"        #{doc_id:>6}  score={score:.4f}  '{title}'")

# ── RESUMEN ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✓ TODO OK — pipeline funcionando correctamente")
print("=" * 55)