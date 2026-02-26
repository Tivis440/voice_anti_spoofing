"""GUI для проверки real/fake по признакам статьи (с эталонным real-файлом)."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

ROOT = Path(__file__).resolve().parent.parent


def run_gui() -> None:
    from .inference import predict_from_wav

    app = tk.Tk()
    app.title("Voice Anti-Spoofing (Article Features)")
    app.geometry("760x520")

    sample_var = tk.StringVar()
    ref_var = tk.StringVar()
    model_var = tk.StringVar(str((ROOT / "logs" / "best_model.pkl").resolve()))
    result_var = tk.StringVar("Ожидание")

    def pick_audio(var: tk.StringVar, title: str) -> None:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.opus *.m4a"), ("All", "*.*")],
            initialdir=str(ROOT),
        )
        if path:
            var.set(path)

    def pick_model() -> None:
        path = filedialog.askopenfilename(
            title="Выберите model .pkl",
            filetypes=[("Model", "*.pkl"), ("All", "*.*")],
            initialdir=str(ROOT / "logs"),
        )
        if path:
            model_var.set(path)

    def classify() -> None:
        sample = sample_var.get().strip()
        ref = ref_var.get().strip()
        model = model_var.get().strip()
        if not sample or not ref or not model:
            messagebox.showwarning("Проверка", "Заполните sample/ref/model")
            return

        result_var.set("Обработка...")

        def _run() -> None:
            try:
                label, p_real, p_fake, feats = predict_from_wav(sample, ref, model)
                text = (
                    f"Класс: {'FAKE' if label == 1 else 'REAL'}\n"
                    f"Real={p_real:.3f}, Fake={p_fake:.3f}\n"
                    f"mfcc_euclidean={feats['mfcc_euclidean']:.4f}\n"
                    f"mfcc_euclidean_norm={feats['mfcc_euclidean_norm']:.6f}\n"
                    f"mfcc_cosine={feats['mfcc_cosine']:.4f}\n"
                    f"centroid_diff={feats['spectral_centroid_diff']:.2f}\n"
                    f"energy_ratio={feats['energy_ratio_noise_to_signal']:.6f}"
                )
                app.after(0, lambda: result_var.set(text))
            except Exception as exc:  # noqa: BLE001
                app.after(0, lambda: result_var.set("Ошибка"))
                app.after(0, lambda: messagebox.showerror("Ошибка", str(exc)))

        threading.Thread(target=_run, daemon=True).start()

    row = 0
    for text, var, btn_title in [
        ("Sample audio:", sample_var, "Выбрать sample"),
        ("Reference real audio:", ref_var, "Выбрать reference"),
    ]:
        tk.Label(app, text=text, anchor="w").grid(row=row, column=0, sticky="w", padx=8, pady=8)
        tk.Entry(app, textvariable=var, width=78).grid(row=row, column=1, padx=8, pady=8)
        tk.Button(app, text=btn_title, command=lambda v=var, t=text: pick_audio(v, t)).grid(
            row=row,
            column=2,
            padx=8,
            pady=8,
        )
        row += 1

    tk.Label(app, text="Model (.pkl):", anchor="w").grid(row=row, column=0, sticky="w", padx=8, pady=8)
    tk.Entry(app, textvariable=model_var, width=78).grid(row=row, column=1, padx=8, pady=8)
    tk.Button(app, text="Выбрать model", command=pick_model).grid(row=row, column=2, padx=8, pady=8)
    row += 1

    tk.Button(app, text="Проверить", command=classify, width=20).grid(row=row, column=1, pady=14)
    row += 1

    tk.Label(app, textvariable=result_var, justify="left", anchor="nw", font=("Menlo", 12)).grid(
        row=row,
        column=0,
        columnspan=3,
        sticky="w",
        padx=8,
        pady=8,
    )

    app.mainloop()


if __name__ == "__main__":
    run_gui()
