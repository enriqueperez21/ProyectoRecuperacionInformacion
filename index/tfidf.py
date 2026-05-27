import math
from index.inverted_index import InvertedIndex


def compute_tfidf(index: InvertedIndex) -> dict[str, dict[str, float]]:
    """
    Calcula el peso TF-IDF para cada término en cada documento.

    TF  (Term Frequency)     = frecuencia del término en el doc / total tokens del doc
    IDF (Inverse Doc Freq)   = log( total_docs / docs_que_tienen_el_término )
    peso = TF × IDF

    Retorna:
        tfidf["coffee"]["doc_001"] = 0.516
    """
    tfidf: dict[str, dict[str, float]] = {}
    N = index.num_docs  # total de documentos

    for term, doc_freqs in index.index.items():
        df = len(doc_freqs)                    # en cuántos docs aparece el término
        idf = math.log((N + 1) / (df + 1))    # +1 para evitar división por cero

        tfidf[term] = {}
        for doc_id, freq in doc_freqs.items():
            doc_len = index.doc_lengths[doc_id]
            tf = freq / doc_len if doc_len > 0 else 0
            tfidf[term][doc_id] = tf * idf

    print(f"[tfidf] Calculados pesos TF-IDF para {len(tfidf)} términos")
    return tfidf