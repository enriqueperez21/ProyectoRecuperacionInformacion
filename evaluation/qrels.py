from pathlib import Path
import pandas as pd
 
DATA_DIR = Path(__file__).parent.parent / "corpus" / "data"
 
 
def load_qrels() -> dict[str, set[str]]:
    """
    Construye el ground truth desde los topics del CSV de test.
 
    Para cada topic (ej: "coffee"), los documentos relevantes son
    todos los que tienen ese topic asignado.
 
    Retorna:
        { "coffee": {"127", "431", "892"}, "trade": {"12", "55"}, ... }
    """
    path = DATA_DIR / "ModApte_test.csv"
    df = pd.read_csv(path)
 
    qrels: dict[str, set[str]] = {}
 
    for _, row in df.iterrows():
        doc_id = str(row["new_id"])
        topics = str(row.get("topics", "")).strip()
 
        for topic in topics.split(","):
            topic = topic.strip()
            if not topic or topic == "nan":
                continue
            if topic not in qrels:
                qrels[topic] = set()
            qrels[topic].add(doc_id)
 
    print(f"[qrels] {len(qrels)} topics cargados como queries de evaluación")
    return qrels