from pathlib import Path
import pandas as pd
import re

DATA_DIR = Path(__file__).parent.parent / "corpus" / "data"


def _parse_topics(raw: str) -> list[str]:
    """Mismo parser que corpus/loader.py — copia local para evitar import circular."""
    raw = raw.strip()
    if not raw or raw == "nan":
        return []
    if raw.startswith("[") and raw.endswith("]"):
        tokens = re.findall(r"'([^']+)'", raw)
        if tokens:
            return [t.strip() for t in tokens if t.strip()]
    if "," in raw:
        return [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]
    cleaned = raw.strip("[]'\"")
    return [cleaned] if cleaned else []


def load_qrels(split: str = "train") -> dict[str, set[str]]:
    """
    Construye el ground truth desde los topics del CSV.

    Usamos el mismo split que el engine (train por defecto) para que
    los doc_ids de los qrels coincidan con los docs indexados.

    Retorna:
        { "coffee": {"127", "431"}, "trade": {"12", "55"}, ... }
    """
    files = {
        "train": "ModApte_train.csv",
        "test":  "ModApte_test.csv",
        "all":   "ModApte_train.csv",
    }
    path = DATA_DIR / files.get(split, "ModApte_train.csv")
    df = pd.read_csv(path)

    qrels: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        doc_id = str(row["new_id"]).strip().strip('"').strip("'")
        topics = _parse_topics(str(row.get("topics", "")))

        for topic in topics:
            if not topic or topic == "nan":
                continue
            if topic not in qrels:
                qrels[topic] = set()
            qrels[topic].add(doc_id)

    # Filtrar topics con al menos 2 documentos relevantes (más útil para evaluar)
    qrels = {t: docs for t, docs in qrels.items() if len(docs) >= 2}

    print(f"[qrels] {len(qrels)} topics con ≥2 docs relevantes (split='{split}')")
    return qrels