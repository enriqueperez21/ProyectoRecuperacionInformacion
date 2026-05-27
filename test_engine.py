"""
Test del SearchEngine completo.
Corre desde la raíz del proyecto:

    python test_engine.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("  TEST DEL SEARCH ENGINE")
print("=" * 55)

from retrieval.engine import SearchEngine

# ── 1. BUILD ─────────────────────────────────────────────────
print("\n[1/3] Construyendo engine...")
engine = SearchEngine(split="train")
engine.build(use_semantic=False)
assert engine._built, "ERROR: engine no se construyó"
print("      ✓ Engine construido")

# ── 2. BÚSQUEDA SIMPLE ───────────────────────────────────────
print("\n[2/3] Probando búsqueda individual...")
QUERY = "oil prices barrel"

for model_name in ["jaccard", "tfidf", "bm25"]:
    results = engine.search(QUERY, model=model_name, top_k=3)
    assert len(results) > 0, f"ERROR: {model_name} no devolvió resultados"

    # Verificar que cada resultado tiene los campos correctos
    r = results[0]
    assert "rank"    in r, "ERROR: falta campo 'rank'"
    assert "doc_id"  in r, "ERROR: falta campo 'doc_id'"
    assert "score"   in r, "ERROR: falta campo 'score'"
    assert "title"   in r, "ERROR: falta campo 'title'"
    assert "preview" in r, "ERROR: falta campo 'preview'"
    assert "topics"  in r, "ERROR: falta campo 'topics'"
    assert r["rank"] == 1,  "ERROR: primer resultado debe tener rank=1"

    print(f"\n      [{model_name.upper()}] query='{QUERY}'")
    for res in results:
        print(f"        #{res['rank']}  score={res['score']:.4f}  '{res['title'][:50]}'")
        print(f"            topics={res['topics']}")

# ── 3. BÚSQUEDA TODOS LOS MODELOS ────────────────────────────
print("\n[3/3] Probando search_all_models...")
all_results = engine.search_all_models("interest rates federal reserve", top_k=2)

assert "jaccard" in all_results, "ERROR: falta jaccard en all_results"
assert "tfidf"   in all_results, "ERROR: falta tfidf en all_results"
assert "bm25"    in all_results, "ERROR: falta bm25 en all_results"

print("\n      Comparación del mismo query en 3 modelos:")
for model, results in all_results.items():
    if results:
        print(f"      [{model:8}] #{1} → '{results[0]['title'][:45]}'")

print("\n" + "=" * 55)
print("  ✓ ENGINE OK")
print("=" * 55)