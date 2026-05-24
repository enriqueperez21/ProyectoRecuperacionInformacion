from .tokenizer import tokenize
from .normalizer import normalize


def preprocess(text: str) -> list[str]:
    """
    Pipeline completo: texto crudo → lista de tokens limpios.

    Ejemplo:
        "Brazil's COFFEE exports ROSE 12%, amid drought!"
        → ["brazil", "coffee", "exports", "rose", "12", "drought"]
    """
    tokens = tokenize(text)
    tokens = normalize(tokens)
    return tokens


def preprocess_query(query: str) -> list[str]:
    """
    Mismo pipeline para las queries del usuario.
    Es idéntico a preprocess() — está separado por claridad.
    """
    return preprocess(query)