import pandas as pd
import ast
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def _parse_topics(raw: str) -> list[str]:
    """
    Limpia el campo topics que puede venir en varios formatos raros:
        "['cocoa']"     → ['cocoa']
        "cocoa,trade"   → ['cocoa', 'trade']
        "['cocoa', 'trade']" → ['cocoa', 'trade']
    """
    raw = raw.strip()
    if not raw or raw == "nan":
        return []

    # Intentar parsear como lista de Python
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            # Puede ser lista de strings o lista de listas
            result = []
            for item in parsed:
                if isinstance(item, list):
                    result.extend([str(i).strip() for i in item if str(i).strip()])
                else:
                    result.append(str(item).strip())
            return [t for t in result if t]
    except Exception:
        pass

    # Si no es lista Python, asumir separado por comas
    return [t.strip() for t in raw.split(",") if t.strip()]


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