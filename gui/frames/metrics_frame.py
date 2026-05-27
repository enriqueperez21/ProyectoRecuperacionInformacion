import tkinter as tk
from tkinter import ttk
import threading


class MetricsFrame(ttk.Frame):
    """
    Tab 2 — Evaluación.
    Corre las métricas (MAP, Precision, Recall) sobre los 3 modelos clásicos
    y muestra una tabla comparativa.
    """

    def __init__(self, parent, engine, **kwargs):
        super().__init__(parent, **kwargs)
        self.engine = engine
        self._results = []
        self._build_ui()

    def _build_ui(self):
        self.configure(padding=16)

        # ── Controles ────────────────────────────────────────
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=(0, 12))

        ttk.Label(ctrl, text="Queries a evaluar:").pack(side="left", padx=(0, 6))
        self.max_q_var = tk.IntVar(value=30)
        ttk.Spinbox(ctrl, from_=10, to=200, increment=10,
                    textvariable=self.max_q_var, width=6).pack(side="left", padx=(0, 12))

        ttk.Label(ctrl, text="Top-K:").pack(side="left", padx=(0, 6))
        self.topk_var = tk.IntVar(value=10)
        ttk.Spinbox(ctrl, from_=5, to=50, increment=5,
                    textvariable=self.topk_var, width=6).pack(side="left", padx=(0, 12))

        self.run_btn = ttk.Button(ctrl, text="▶  Evaluar modelos", command=self._run_eval)
        self.run_btn.pack(side="left")

        # ── Status ───────────────────────────────────────────
        self.status_var = tk.StringVar(value="Presiona 'Evaluar modelos' para comenzar.")
        ttk.Label(self, textvariable=self.status_var,
                  foreground="gray").pack(anchor="w", pady=(0, 8))

        # ── Barra de progreso ────────────────────────────────
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 12))

        # ── Tabla de resultados ──────────────────────────────
        cols = ("Modelo", "MAP", "Precision@K", "Recall@K", "Queries")
        self.tree = ttk.Treeview(self, columns=cols, show="headings",
                                 height=5, selectmode="none")

        for col in cols:
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, width=130, anchor="center")
        self.tree.column("Modelo", width=180, anchor="w")

        self.tree.pack(fill="x", pady=(0, 16))

        # ── Análisis textual ─────────────────────────────────
        analysis_lf = ttk.LabelFrame(self, text="Análisis de resultados", padding=8)
        analysis_lf.pack(fill="both", expand=True)

        self.analysis_text = tk.Text(analysis_lf, wrap="word", state="disabled",
                                     relief="flat", font=("Courier New", 9), height=10)
        sb = ttk.Scrollbar(analysis_lf, orient="vertical",
                           command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=sb.set)
        self.analysis_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _run_eval(self):
        """Lanza la evaluación en un thread separado para no congelar la GUI."""
        self.run_btn.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set("Evaluando... esto puede tardar 1-2 minutos.")
        self.tree.delete(*self.tree.get_children())

        thread = threading.Thread(target=self._eval_worker, daemon=True)
        thread.start()

    def _eval_worker(self):
        """Worker que corre en background."""
        try:
            from evaluation.qrels import load_qrels
            from evaluation.metrics import evaluate_model

            qrels   = load_qrels(split="train")
            max_q   = self.max_q_var.get()
            top_k   = self.topk_var.get()
            models  = ["jaccard", "tfidf", "bm25"]
            results = []

            for model in models:
                self.status_var.set(f"Evaluando {model}...")
                m = evaluate_model(self.engine, model, qrels,
                                   top_k=top_k, max_queries=max_q)
                results.append(m)

            self._results = results
            # Actualizar UI desde el thread principal
            self.after(0, self._show_results, results)

        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _show_results(self, results):
        """Muestra los resultados en la tabla."""
        self.progress.stop()
        self.run_btn.configure(state="normal")

        self.tree.delete(*self.tree.get_children())
        for r in results:
            self.tree.insert("", "end", values=(
                r["model"].upper(),
                f"{r['map']:.4f}",
                f"{r['avg_precision']:.4f}",
                f"{r['avg_recall']:.4f}",
                r["num_queries"],
            ))

        self._write_analysis(results)
        self.status_var.set(f"✓ Evaluación completa — {results[0]['num_queries']} queries.")

    def _write_analysis(self, results):
        """Genera el texto de análisis automático."""
        best = max(results, key=lambda r: r["map"])
        worst = min(results, key=lambda r: r["map"])

        lines = [
            "ANÁLISIS DE RESULTADOS",
            "=" * 50,
            "",
            f"Mejor modelo por MAP:  {best['model'].upper()} (MAP={best['map']:.4f})",
            f"Peor  modelo por MAP:  {worst['model'].upper()} (MAP={worst['map']:.4f})",
            "",
            "Ranking completo:",
        ]
        ranked = sorted(results, key=lambda r: r["map"], reverse=True)
        for i, r in enumerate(ranked, 1):
            lines.append(
                f"  {i}. {r['model'].upper():<12} "
                f"MAP={r['map']:.4f}  "
                f"P={r['avg_precision']:.4f}  "
                f"R={r['avg_recall']:.4f}"
            )

        lines += [
            "",
            "Interpretación:",
            f"  - BM25 suele ganar en corpus de noticias cortas como Reuters.",
            f"  - Jaccard tiene la precision más baja porque ignora la frecuencia",
            f"    de las palabras y no pondera términos raros.",
            f"  - TF-IDF mejora sobre Jaccard al ponderar por rareza del término.",
            f"  - El Recall bajo (~0.13) es normal: top-{results[0]['num_queries']}",
            f"    sobre un corpus grande solo cubre una fracción de los relevantes.",
        ]

        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("end", "\n".join(lines))
        self.analysis_text.configure(state="disabled")

    def _show_error(self, msg):
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.status_var.set(f"✗ Error: {msg}")