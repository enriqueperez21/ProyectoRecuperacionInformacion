import pandas as pd
import ast
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def _parse_topics(raw: str) -> list[str]:
    """
    Limpia el campo topics que puede venir en varios formatos:

        "['cocoa']"                    → ['cocoa']           lista Python 1 elemento
        "['cocoa', 'trade']"           → ['cocoa', 'trade']  lista Python con comas
        "['grain' 'wheat' 'corn']"     → ['grain', 'wheat', 'corn']  array NumPy (espacios)
        "cocoa,trade"                  → ['cocoa', 'trade']  separado por comas
    """
    raw = raw.strip()
    if not raw or raw == "nan":
        return []

    # Caso: viene entre corchetes — puede ser lista Python o array NumPy
    if raw.startswith("[") and raw.endswith("]"):
        # Quitar corchetes y extraer todo lo que esté entre comillas simples
        import re
        tokens = re.findall(r"'([^']+)'", raw)
        if tokens:
            return [t.strip() for t in tokens if t.strip()]

        # Si no hay comillas simples, intentar ast.literal_eval (lista Python normal)
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except Exception:
            pass

    # Caso: separado por comas sin corchetes — "cocoa,trade"
    if "," in raw:
        return [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]

    # Caso: un solo topic sin formato especial — "cocoa"
    cleaned = raw.strip("[]'\"")
    return [cleaned] if cleaned else []


def load_corpus(split: str = "train") -> dict[str, dict]:
    """
    Carga los documentos de Reuters-21578 desde los CSVs.

    split: "train" | "test" | "all"

    Retorna:
        { "doc_id": { "title": "...", "text": "...", "topics": [...] } }
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
        else:
            print(f"[corpus] Advertencia: no se encontró {filename}")

    if not dfs:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {DATA_DIR}")

    df = pd.concat(dfs, ignore_index=True)

    docs = {}
    for _, row in df.iterrows():
        # Limpiar comillas del id: '"1"' → '1'
        doc_id = str(row["new_id"]).strip().strip('"').strip("'")
        title  = str(row.get("title", "")).strip()
        text   = str(row.get("text",  "")).strip()
        topics = _parse_topics(str(row.get("topics", "")))

        if not text and not title:
            continue

        docs[doc_id] = {
            "title":  title,
            "text":   f"{title} {text}".strip(),
            "topics": topics,
        }

    print(f"[corpus] Cargados {len(docs)} documentos (split='{split}')")
    return docs