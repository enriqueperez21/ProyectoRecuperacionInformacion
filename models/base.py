from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Clase base para todos los modelos de recuperación.
    Todos deben implementar el método search().
    """

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Busca documentos relevantes para la query.

        Retorna una lista de (doc_id, score) ordenada de mayor a menor score.
        Ejemplo: [("doc_004", 0.91), ("doc_001", 0.87), ...]
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del modelo para mostrar en la interfaz."""
        pass