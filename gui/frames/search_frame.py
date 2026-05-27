import tkinter as tk
from tkinter import ttk
import html


class SearchFrame(ttk.Frame):
    """
    Tab 1 — Búsqueda.
    El usuario escribe una query, elige un modelo y ve el ranking de resultados.
    """

    MODELS = [
        ("BM25",              "bm25"),
        ("TF-IDF + Coseno",   "tfidf"),
        ("Jaccard",           "jaccard"),
        ("Embeddings",        "semantic"),
    ]

    def __init__(self, parent, engine, **kwargs):
        super().__init__(parent, **kwargs)
        self.engine = engine
        self._build_ui()

    def _build_ui(self):
        self.configure(padding=16)

        # ── Barra de búsqueda ────────────────────────────────
        search_bar = ttk.Frame(self)
        search_bar.pack(fill="x", pady=(0, 12))

        ttk.Label(search_bar, text="Consulta:").pack(side="left", padx=(0, 8))

        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(search_bar, textvariable=self.query_var, width=50)
        self.query_entry.pack(side="left", padx=(0, 8))
        self.query_entry.bind("<Return>", lambda e: self._search())

        ttk.Label(search_bar, text="Modelo:").pack(side="left", padx=(0, 6))

        self.model_var = tk.StringVar(value="bm25")
        model_cb = ttk.Combobox(
            search_bar,
            textvariable=self.model_var,
            values=[m[0] for m in self.MODELS],
            state="readonly",
            width=18,
        )
        model_cb.pack(side="left", padx=(0, 8))
        model_cb.bind("<<ComboboxSelected>>", self._on_model_change)
        self._model_cb = model_cb
        # Mostrar nombre legible al inicio
        model_cb.set("BM25")

        self.search_btn = ttk.Button(search_bar, text="Buscar", command=self._search)
        self.search_btn.pack(side="left")

        self.top_k_var = tk.IntVar(value=10)
        ttk.Label(search_bar, text="Top:").pack(side="left", padx=(12, 4))
        ttk.Spinbox(search_bar, from_=5, to=50, increment=5,
                    textvariable=self.top_k_var, width=5).pack(side="left")

        # ── Status ───────────────────────────────────────────
        self.status_var = tk.StringVar(value="Listo. Escribe una consulta y presiona Buscar.")
        ttk.Label(self, textvariable=self.status_var,
                  foreground="gray").pack(anchor="w", pady=(0, 8))

        # ── Resultados ───────────────────────────────────────
        results_frame = ttk.Frame(self)
        results_frame.pack(fill="both", expand=True)

        cols = ("#", "Score", "Título", "Topics")
        self.tree = ttk.Treeview(results_frame, columns=cols,
                                 show="headings", selectmode="browse")

        self.tree.heading("#",      text="#",      anchor="center")
        self.tree.heading("Score",  text="Score",  anchor="center")
        self.tree.heading("Título", text="Título", anchor="w")
        self.tree.heading("Topics", text="Topics", anchor="w")

        self.tree.column("#",      width=40,  stretch=False, anchor="center")
        self.tree.column("Score",  width=80,  stretch=False, anchor="center")
        self.tree.column("Título", width=420, stretch=True)
        self.tree.column("Topics", width=180, stretch=True)

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical",
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Click en fila → mostrar preview
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # ── Preview del documento ────────────────────────────
        preview_lf = ttk.LabelFrame(self, text="Preview del documento", padding=8)
        preview_lf.pack(fill="x", pady=(10, 0))

        self.preview_text = tk.Text(preview_lf, height=5, wrap="word",
                                    state="disabled", relief="flat",
                                    font=("Courier New", 9))
        self.preview_text.pack(fill="x")

        # Guardar resultados actuales para el preview
        self._current_results = []

    def _on_model_change(self, event=None):
        """Sincroniza el nombre visible con el valor interno."""
        label = self._model_cb.get()
        for name, key in self.MODELS:
            if name == label:
                self.model_var.set(key)
                break

    def _search(self):
        query = self.query_var.get().strip()
        if not query:
            self.status_var.set("⚠ Escribe una consulta primero.")
            return

        model_key = self.model_var.get()
        top_k     = self.top_k_var.get()

        # Verificar modelo semántico
        if model_key == "semantic" and self.engine.semantic is None:
            self.status_var.set("⚠ Modelo semántico no disponible. Instala sentence-transformers.")
            return

        self.status_var.set(f"Buscando '{query}' con {model_key}...")
        self.update_idletasks()

        try:
            results = self.engine.search(query, model=model_key, top_k=top_k)
            self._current_results = results
            self._populate_tree(results)
            self.status_var.set(
                f"✓ {len(results)} resultados para '{query}' — modelo: {model_key}"
            )
        except Exception as e:
            self.status_var.set(f"✗ Error: {e}")

    def _populate_tree(self, results):
        """Llena la tabla con los resultados."""
        self.tree.delete(*self.tree.get_children())
        for r in results:
            title  = html.unescape(r["title"])       # limpia &lt; etc.
            topics = ", ".join(r["topics"]) if r["topics"] else "—"
            score  = f"{r['score']:.4f}"
            self.tree.insert("", "end", iid=str(r["rank"]),
                             values=(r["rank"], score, title, topics))

    def _on_select(self, event=None):
        """Muestra el preview del documento seleccionado."""
        sel = self.tree.selection()
        if not sel:
            return
        rank = int(sel[0])
        doc  = next((r for r in self._current_results if r["rank"] == rank), None)
        if not doc:
            return

        preview = html.unescape(doc.get("preview", ""))
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("end", f"[Doc {doc['doc_id']}] {preview}")
        self.preview_text.configure(state="disabled")