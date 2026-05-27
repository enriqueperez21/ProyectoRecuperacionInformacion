def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@k: de los primeros k documentos devueltos,
    ¿qué fracción son relevantes?

    Ejemplo: retrieved=["d1","d2","d3"], relevant={"d1","d3"}, k=3
    → 2/3 = 0.667
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Recall@k: de todos los documentos relevantes que existen,
    ¿qué fracción encontró el sistema en los primeros k?

    Ejemplo: retrieved=["d1","d2","d3"], relevant={"d1","d3","d5"}, k=3
    → 2/3 = 0.667  (encontró d1 y d3, faltó d5)
    """
    if not relevant or k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """
    Average Precision (AP): promedio de precision en cada posición
    donde aparece un documento relevante.

    Premia los rankings donde los documentos relevantes aparecen arriba.
    """
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / i  # precision en esta posición

    return precision_sum / len(relevant)


def mean_average_precision(
    results_per_query: dict[str, list[str]],
    qrels: dict[str, set[str]],
) -> float:
    """
    MAP: promedio de AP sobre todas las queries.
    Es la métrica principal para comparar modelos.

    results_per_query: { "query": ["doc_id_1", "doc_id_2", ...] }
    qrels:             { "query": {"doc_id_relevante", ...} }
    """
    aps = []

    for query, retrieved in results_per_query.items():
        relevant = qrels.get(query, set())
        if not relevant:
            continue
        ap = average_precision(retrieved, relevant)
        aps.append(ap)

    if not aps:
        return 0.0

    return sum(aps) / len(aps)


def evaluate_model(
    engine,
    model_name: str,
    qrels: dict[str, set[str]],
    top_k: int = 10,
    max_queries: int = 50,
) -> dict:
    """
    Evalúa un modelo completo sobre las queries del qrels.

    Retorna un dict con:
        { "map": 0.71, "avg_precision": 0.65, "avg_recall": 0.58,
          "num_queries": 50, "model": "bm25" }
    """
    results_per_query: dict[str, list[str]] = {}
    precisions = []
    recalls = []

    queries = list(qrels.keys())[:max_queries]

    for query in queries:
        relevant = qrels[query]
        if not relevant:
            continue

        try:
            raw = engine.search(query, model=model_name, top_k=top_k)
            retrieved = [r["doc_id"] for r in raw]
        except Exception:
            retrieved = []

        results_per_query[query] = retrieved
        precisions.append(precision_at_k(retrieved, relevant, top_k))
        recalls.append(recall_at_k(retrieved, relevant, top_k))

    map_score = mean_average_precision(results_per_query, qrels)

    return {
        "model":         model_name,
        "map":           round(map_score, 4),
        "avg_precision": round(sum(precisions) / len(precisions), 4) if precisions else 0,
        "avg_recall":    round(sum(recalls)    / len(recalls),    4) if recalls    else 0,
        "num_queries":   len(queries),
    }