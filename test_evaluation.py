"""
Test de evaluación: precision, recall y MAP.
Corre desde la raíz del proyecto:

    python test_evaluation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("  TEST DE EVALUACIÓN")
print("=" * 55)

# ── 1. QRELS ─────────────────────────────────────────────────
print("\n[1/3] Cargando qrels...")
from evaluation.qrels import load_qrels

qrels = load_qrels(split="train")
assert len(qrels) > 0, "ERROR: qrels vacío"

sample_topics = list(qrels.items())[:4]
print("      Muestra de qrels (topics limpios):")
for topic, doc_ids in sample_topics:
    print(f"        '{topic}' → {len(doc_ids)} docs relevantes")
print(f"      ✓ {len(qrels)} topics cargados")

# ── 2. MÉTRICAS UNITARIAS ────────────────────────────────────
print("\n[2/3] Probando métricas individuales...")
from evaluation.metrics import precision_at_k, recall_at_k, average_precision

retrieved = ["d1", "d2", "d3", "d4", "d5"]
relevant  = {"d1", "d3", "d5"}

p  = precision_at_k(retrieved, relevant, k=5)
r  = recall_at_k(retrieved, relevant, k=5)
ap = average_precision(retrieved, relevant)

assert abs(p  - 0.6)    < 0.001
assert abs(r  - 1.0)    < 0.001
assert abs(ap - 0.7556) < 0.001
print(f"      Precision@5={p:.4f}  Recall@5={r:.4f}  AP={ap:.4f}  ✓")

# ── 3. EVALUACIÓN COMPLETA ───────────────────────────────────
print("\n[3/3] Evaluando los 3 modelos...")
from retrieval.engine import SearchEngine
from evaluation.metrics import evaluate_model

engine = SearchEngine(split="train")
engine.build(use_semantic=False)

results_table = []
for model_name in ["jaccard", "tfidf", "bm25"]:
    print(f"\n      Evaluando {model_name}...")
    metrics = evaluate_model(engine, model_name, qrels, top_k=10, max_queries=30)
    results_table.append(metrics)
    print(f"        MAP={metrics['map']:.4f}  "
          f"P={metrics['avg_precision']:.4f}  "
          f"R={metrics['avg_recall']:.4f}")

print("\n" + "=" * 55)
print(f"  {'MODELO':<20} {'MAP':>8} {'PRECISION':>10} {'RECALL':>8}")
print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8}")
for m in results_table:
    print(f"  {m['model']:<20} {m['map']:>8.4f} {m['avg_precision']:>10.4f} {m['avg_recall']:>8.4f}")
print("=" * 55)
print("  ✓ EVALUACIÓN OK")
print("=" * 55)