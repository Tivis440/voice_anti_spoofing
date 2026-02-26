"""GUI для мультиклассового детекта: real / fake_engine."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

ROOT = Path(__file__).resolve().parent.parent


def run_gui() -> None:
    from .inference import predict_from_wav

    app = tk.Tk()
    app.title("Voice Anti-Spoofing")
    app.geometry("820x560")

    sample_var = tk.StringVar()
    model_var = tk.StringVar(str((ROOT / "logs" / "best_model.pt").resolve()))
    result_var = tk.StringVar("Ожидание")

    def pick_audio() -> None:
        path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.opus *.m4a"), ("All", "*.*")],
            initialdir=str(ROOT),
        )
        if path:
            sample_var.set(path)

    def pick_model() -> None:
        path = filedialog.askopenfilename(
            title="Выберите модель (.pt)",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
            initialdir=str(ROOT / "logs"),
        )
        if path:
            model_var.set(path)

    def classify() -> None:
        sample = sample_var.get().strip()
        model = model_var.get().strip()
        if not sample or not model:
            messagebox.showwarning("Проверка", "Выберите sample и model")
            return

        result_var.set("Обработка...")

        def _run() -> None:
            try:
                pred_class, is_fake, probs, feats = predict_from_wav(sample, model)
                top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                top_txt = "\n".join([f"{name}: {prob:.3f}" for name, prob in top])
                status = "FAKE" if is_fake else "REAL"
                text = (
                    f"Результат: {status}\n"
                    f"Класс: {pred_class}\n\n"
                    f"Top probabilities:\n{top_txt}\n\n"
                    f"MFCC0 mean/std: {feats.get('mfcc_0_mean', 0.0):.3f} / "
                    f"{feats.get('mfcc_0_std', 0.0):.3f}\n"
                    f"Centroid mean: {feats.get('centroid_mean', 0.0):.2f}\n"
                    f"RMS mean: {feats.get('rms_mean', 0.0):.5f}"
                )
                app.after(0, lambda: result_var.set(text))
            except Exception as exc:  # noqa: BLE001
                app.after(0, lambda: result_var.set("Ошибка"))
                app.after(0, lambda: messagebox.showerror("Ошибка", str(exc)))

        threading.Thread(target=_run, daemon=True).start()

    tk.Label(app, text="Sample audio:", anchor="w").grid(row=0, column=0, sticky="w", padx=8, pady=8)
    tk.Entry(app, textvariable=sample_var, width=84).grid(row=0, column=1, padx=8, pady=8)
    tk.Button(app, text="Выбрать sample", command=pick_audio).grid(row=0, column=2, padx=8, pady=8)

    tk.Label(app, text="Model (.pt):", anchor="w").grid(row=1, column=0, sticky="w", padx=8, pady=8)
    tk.Entry(app, textvariable=model_var, width=84).grid(row=1, column=1, padx=8, pady=8)
    tk.Button(app, text="Выбрать model", command=pick_model).grid(row=1, column=2, padx=8, pady=8)

    tk.Button(app, text="Проверить", command=classify, width=20).grid(row=2, column=1, pady=16)

    tk.Label(app, textvariable=result_var, justify="left", anchor="nw", font=("Menlo", 12)).grid(
        row=3,
        column=0,
        columnspan=3,
        sticky="w",
        padx=8,
        pady=8,
    )

    app.mainloop()


if __name__ == "__main__":
    run_gui()
