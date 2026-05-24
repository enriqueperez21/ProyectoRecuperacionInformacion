import nltk

# Descargar recursos necesarios de NLTK (solo la primera vez)
nltk.download("punkt",                      quiet=True)
nltk.download("punkt_tab",                  quiet=True)
nltk.download("wordnet",                    quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

_lemmatizer = WordNetLemmatizer()


def _get_wordnet_pos(nltk_tag: str) -> str:
    """
    Convierte la etiqueta gramatical de NLTK al formato que entiende WordNet.

    NLTK etiqueta cada palabra: NN=sustantivo, VB=verbo, JJ=adjetivo, RB=adverbio.
    WordNetLemmatizer necesita saber esto para lematizar bien.

    Sin esto:  "ran" → "ran"   (mal, no sabe que es verbo)
    Con esto:  "ran" → "run"   (bien)
               "better" → "good" (bien)
               "exports" → "export" (bien)
    """
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default


def tokenize(text: str) -> list[str]:
    """
    Tokeniza y lematiza el texto usando NLTK.

    Pasos:
    1. word_tokenize → divide en palabras
    2. pos_tag       → etiqueta gramatical de cada palabra (sustantivo, verbo, etc.)
    3. lemmatize     → reduce a forma base usando la etiqueta correcta

    Ejemplo:
        "Brazil's COFFEE exports ROSE 12%!"
        → tokens:    ["Brazil", "'s", "COFFEE", "exports", "ROSE", "12", "%"]
        → filtrados: ["Brazil", "COFFEE", "exports", "ROSE", "12"]
        → lematizados: ["Brazil", "COFFEE", "export", "rise", "12"]
    """
    # 1. Tokenizar
    raw_tokens = word_tokenize(text)

    # 2. Filtrar: solo alfanuméricos de longitud > 1
    filtered = [(t, ) for t in raw_tokens if t.isalnum() and len(t) > 1]
    words    = [t[0] for t in filtered]

    if not words:
        return []

    # 3. Etiquetar gramaticalmente con NLTK
    tagged = nltk.pos_tag(words)  # → [("exports", "NNS"), ("rose", "VBD"), ...]

    # 4. Lematizar usando la etiqueta correcta
    lemmatized = [
        _lemmatizer.lemmatize(word, _get_wordnet_pos(tag))
        for word, tag in tagged
    ]

    return lemmatized