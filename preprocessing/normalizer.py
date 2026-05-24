import nltk
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# Stopwords en inglés de NLTK (~179 palabras, más completa que la lista manual)
_STOPWORDS = set(stopwords.words("english"))


def normalize(tokens: list[str]) -> list[str]:
    """
    Normaliza una lista de tokens usando NLTK:
    1. Pasa todo a minúsculas
    2. Elimina stopwords de NLTK (más completa que la lista manual)

    Ejemplo:
        ["Brazil", "coffee", "exports", "amid", "the", "drought"]
        → ["brazil", "coffee", "exports", "drought"]
    """
    return [
        token.lower()
        for token in tokens
        if token.lower() not in _STOPWORDS
    ]