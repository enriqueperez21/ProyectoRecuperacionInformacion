import tkinter as tk
from tkinter import ttk
import threading

from gui.frames.search_frame  import SearchFrame
from gui.frames.metrics_frame import MetricsFrame
from gui.frames.results_frame import ResultsFrame


class SRIApp(tk.Tk):
    """
    Ventana principal de la aplicación.
    Muestra una pantalla de carga mientras el engine se inicializa,
    luego presenta los 3 tabs: Búsqueda, Evaluación y Comparación.
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

        self.title("Sistema de Recuperación de Información — Reuters-21578")
        self.geometry("900x650")
        self.minsize(750, 500)
        self.resizable(True, True)

        # Estilo
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook.Tab", padding=(14, 6), font=("Helvetica", 10))
        style.configure("TButton",       padding=(10, 5))
        style.configure("TLabel",        font=("Helvetica", 10))
        style.configure("TLabelframe.Label", font=("Helvetica", 10, "bold"))

        self._build_loading_screen()

    def _build_loading_screen(self):
        """Pantalla de carga mientras el engine hace build()."""
        self._loading_frame = ttk.Frame(self, padding=40)
        self._loading_frame.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(
            self._loading_frame,
            text="Sistema de Recuperación de Información",
            font=("Helvetica", 16, "bold"),
        ).pack(pady=(0, 6))

        ttk.Label(
            self._loading_frame,
            text="Reuters-21578 · Jaccard · TF-IDF · BM25 · Embeddings",
            font=("Helvetica", 10),
            foreground="gray",
        ).pack(pady=(0, 24))

        self._load_status = tk.StringVar(value="Iniciando...")
        ttk.Label(
            self._loading_frame,
            textvariable=self._load_status,
            foreground="gray",
        ).pack(pady=(0, 10))

        self._progress = ttk.Progressbar(
            self._loading_frame, mode="indeterminate", length=300
        )
        self._progress.pack()
        self._progress.start(12)

        # Lanzar build() en background
        thread = threading.Thread(target=self._build_engine, daemon=True)
        thread.start()

    def _build_engine(self):
        """Corre engine.build() en background sin congelar la ventana."""
        try:
            self._load_status.set("Cargando corpus Reuters-21578...")
            self.engine.build(use_semantic=False)
            self.after(0, self._on_engine_ready)
        except Exception as e:
            self.after(0, self._on_engine_error, str(e))

    def _on_engine_ready(self):
        """Cuando el engine terminó: quitar pantalla de carga y mostrar tabs."""
        self._progress.stop()
        self._loading_frame.destroy()
        self._build_main_ui()

    def _on_engine_error(self, msg):
        self._progress.stop()
        self._load_status.set(f"Error al inicializar: {msg}")

    def _build_main_ui(self):
        """Construye la interfaz principal con los 3 tabs."""
        # Header
        header = ttk.Frame(self, padding=(16, 10, 16, 0))
        header.pack(fill="x")

        ttk.Label(
            header,
            text="SRI · Reuters-21578",
            font=("Helvetica", 13, "bold"),
        ).pack(side="left")

        ttk.Label(
            header,
            text=f"{self.engine.index.num_docs:,} documentos indexados · "
                 f"{len(self.engine.index.index):,} términos únicos",
            font=("Helvetica", 9),
            foreground="gray",
        ).pack(side="right")

        ttk.Separator(self).pack(fill="x", padx=16, pady=6)

        # Notebook (tabs)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        search_tab  = SearchFrame(nb,  self.engine)
        metrics_tab = MetricsFrame(nb, self.engine)
        compare_tab = ResultsFrame(nb, self.engine)

        nb.add(search_tab,  text="  🔍  Búsqueda  ")
        nb.add(metrics_tab, text="  📊  Evaluación  ")
        nb.add(compare_tab, text="  ⚖️   Comparación  ")