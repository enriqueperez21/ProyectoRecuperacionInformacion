import tkinter as tk
from tkinter import ttk
import html


class ResultsFrame(ttk.Frame):
    """
    Tab 3 — Comparación.
    Corre la misma query en los 3 modelos clásicos y muestra
    los resultados en columnas paralelas para comparar.
    """

    MODELS = [
        ("Jaccard",  "jaccard"),
        ("TF-IDF",   "tfidf"),
        ("BM25",     "bm25"),
    ]

    def __init__(self, parent, engine, **kwargs):
        super().__init__(parent, **kwargs)
        self.engine = engine
        self._build_ui()

    def _build_ui(self):
        self.configure(padding=16)

        # ── Barra de búsqueda ────────────────────────────────
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(0, 12))

        ttk.Label(bar, text="Consulta:").pack(side="left", padx=(0, 8))
        self.query_var = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.query_var, width=50)
        entry.pack(side="left", padx=(0, 8))
        entry.bind("<Return>", lambda e: self._compare())

        ttk.Button(bar, text="Comparar modelos", command=self._compare).pack(side="left")

        self.status_var = tk.StringVar(value="Escribe una consulta para comparar los 3 modelos.")
        ttk.Label(self, textvariable=self.status_var,
                  foreground="gray").pack(anchor="w", pady=(0, 8))

        # ── 3 columnas ───────────────────────────────────────
        cols_frame = ttk.Frame(self)
        cols_frame.pack(fill="both", expand=True)
        cols_frame.columnconfigure(0, weight=1)
        cols_frame.columnconfigure(1, weight=1)
        cols_frame.columnconfigure(2, weight=1)

        self._trees = {}
        for col_idx, (label, key) in enumerate(self.MODELS):
            lf = ttk.LabelFrame(cols_frame, text=f"  {label}  ", padding=6)
            lf.grid(row=0, column=col_idx, sticky="nsew", padx=4)
            lf.rowconfigure(0, weight=1)
            lf.columnconfigure(0, weight=1)

            tree = ttk.Treeview(lf, columns=("Rank", "Título"),
                                show="headings", selectmode="none")
            tree.heading("Rank",   text="#",      anchor="center")
            tree.heading("Título", text="Título", anchor="w")
            tree.column("Rank",   width=30, stretch=False, anchor="center")
            tree.column("Título", width=200, stretch=True)

            sb = ttk.Scrollbar(lf, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=sb.set)

            tree.grid(row=0, column=0, sticky="nsew")
            sb.grid(row=0, column=1, sticky="ns")

            self._trees[key] = tree

    def _compare(self):
        query = self.query_var.get().strip()
        if not query:
            self.status_var.set("⚠ Escribe una consulta primero.")
            return

        self.status_var.set(f"Buscando '{query}' en los 3 modelos...")
        self.update_idletasks()

        try:
            all_results = self.engine.search_all_models(query, top_k=10)

            for key, tree in self._trees.items():
                tree.delete(*tree.get_children())
                for r in all_results.get(key, []):
                    title = html.unescape(r["title"])[:60]
                    tree.insert("", "end", values=(r["rank"], title))

            self.status_var.set(f"✓ Comparación lista para '{query}'")
        except Exception as e:
            self.status_var.set(f"✗ Error: {e}")