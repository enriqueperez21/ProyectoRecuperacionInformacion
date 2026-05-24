import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def load_corpus(split: str = "train") -> dict[str, dict]:
    """
    Carga los documentos de Reuters-21578 desde los CSVs.

    split: "train" | "test" | "all"

    Retorna un diccionario:
        { "doc_id": { "title": "...", "body": "...", "topics": [...] } }
    """
    files = {
        "train": ["ModApte_train.csv"],
        "test":  ["ModApte_test.csv"],
        "all":   ["ModApte_train.csv", "ModApte_test.csv", "ModApte_unused.csv"],
    }

    if split not in files:
        raise ValueError(f"split debe ser 'train', 'test' o 'all'. Recibí: {split}")

    dfs = []
    for filename in files[split]:
        path = DATA_DIR / filename
        if path.exists():
            dfs.append(pd.read_csv(path))

    if not dfs:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {DATA_DIR}")

    df = pd.concat(dfs, ignore_index=True)

    # Normalizar nombres de columnas a minúsculas por si varían
    df.columns = [c.lower().strip() for c in df.columns]

    docs = {}
    for _, row in df.iterrows():
        # Intentar las columnas más comunes del dataset Reuters
        doc_id = str(row.get("newid", row.get("id", row.name)))
        title  = str(row.get("title", "")).strip()
        body   = str(row.get("body",  row.get("text", ""))).strip()
        topics = str(row.get("topics", "")).strip()

        # Ignorar documentos completamente vacíos
        if not title and not body:
            continue

        docs[doc_id] = {
            "title":  title,
            "body":   body,
            "text":   f"{title} {body}".strip(),  # texto completo para indexar
            "topics": [t.strip() for t in topics.split(",") if t.strip()],
        }

    print(f"[corpus] Cargados {len(docs)} documentos (split='{split}')")
    return docs