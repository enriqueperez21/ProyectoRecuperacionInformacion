"""
Sistema de Recuperación de Información — Reuters-21578
Prof. Iván Carrera · 1er Bimestre 2026

Ejecutar desde la raíz del proyecto:
    python main.py
"""
import sys
import os

# Asegurar que Python encuentre los módulos del proyecto
sys.path.insert(0, os.path.dirname(__file__))

from retrieval.engine import SearchEngine
from gui.app import SRIApp


def main():
    engine = SearchEngine(split="train")
    app = SRIApp(engine)
    app.mainloop()


if __name__ == "__main__":
    main()