"""Адаптивный и user-friendly GUI для мультиклассового детекта."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent.parent


def run_gui() -> None:
    from .inference import predict_from_wav

    app = tk.Tk()
    app.title("Voice Anti-Spoofing")
    app.geometry("980x680")
    app.minsize(860, 580)

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root_frame = ttk.Frame(app, padding=16)
    root_frame.pack(fill=tk.BOTH, expand=True)
    root_frame.columnconfigure(1, weight=1)
    root_frame.rowconfigure(4, weight=1)

    sample_var = tk.StringVar(master=app, value="")
    model_var = tk.StringVar(master=app, value=str((ROOT / "logs" / "best_model.pt").resolve()))
    status_var = tk.StringVar(master=app, value="Готов к проверке")

    title = ttk.Label(
        root_frame,
        text="Проверка голоса: ",
        font=("Helvetica", 18, "bold"),
    )
    title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    subtitle = ttk.Label(
        root_frame,
        text="Загрузите аудио и модель, затем нажмите «Проверить».",
    )
    subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

    ttk.Label(root_frame, text="Аудиофайл:").grid(row=2, column=0, sticky="w", pady=6)
    sample_entry = ttk.Entry(root_frame, textvariable=sample_var)
    sample_entry.grid(row=2, column=1, sticky="ew", pady=6, padx=8)

    ttk.Label(root_frame, text="Модель (.pt):").grid(row=3, column=0, sticky="w", pady=6)
    model_entry = ttk.Entry(root_frame, textvariable=model_var)
    model_entry.grid(row=3, column=1, sticky="ew", pady=6, padx=8)

    buttons_row = ttk.Frame(root_frame)
    buttons_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 8))
    buttons_row.columnconfigure(0, weight=1)
    buttons_row.columnconfigure(1, weight=1)
    buttons_row.columnconfigure(2, weight=1)
    buttons_row.columnconfigure(3, weight=1)

    result_box = ScrolledText(
        root_frame,
        wrap=tk.WORD,
        font=("Menlo", 12),
        height=20,
        padx=10,
        pady=10,
    )
    result_box.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
    result_box.insert(
        tk.END,
        "Результат появится здесь.\n\n"
        "для лучшего качества используйте WAV 16 kHz mono.\n",
    )
    result_box.configure(state=tk.DISABLED)

    progress = ttk.Progressbar(root_frame, mode="indeterminate")
    progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    progress.grid_remove()

    status_label = ttk.Label(root_frame, textvariable=status_var, foreground="#2f6f3e")
    status_label.grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def set_result(text: str) -> None:
        result_box.configure(state=tk.NORMAL)
        result_box.delete("1.0", tk.END)
        result_box.insert(tk.END, text)
        result_box.configure(state=tk.DISABLED)

    def pick_audio() -> None:
        path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.opus *.m4a"), ("All", "*.*")],
            initialdir=str(ROOT),
        )
        if path:
            sample_var.set(path)
            status_var.set("Выбран аудиофайл")

    def pick_model() -> None:
        path = filedialog.askopenfilename(
            title="Выберите модель (.pt)",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
            initialdir=str(ROOT / "logs"),
        )
        if path:
            model_var.set(path)
            status_var.set("Выбрана модель")

    def set_controls_state(enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        sample_entry.configure(state=state)
        model_entry.configure(state=state)
        pick_audio_btn.configure(state=state)
        pick_model_btn.configure(state=state)
        run_btn.configure(state=state)

    def classify() -> None:
        sample = sample_var.get().strip()
        model = model_var.get().strip()
        if not sample or not model:
            messagebox.showwarning("Проверка", "Выберите аудиофайл и модель")
            return

        set_controls_state(False)
        status_var.set("Обработка...")
        progress.grid()
        progress.start(10)

        def _run() -> None:
            try:
                pred_class, is_fake, probs, feats = predict_from_wav(sample, model)
                probs_sorted = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                probs_txt = "\n".join([f"  {name:<18} {prob:.3f}" for name, prob in probs_sorted])
                status = "FAKE" if is_fake else "REAL"
                text = (
                    f"Результат: {status}\n"
                    f"Класс: {pred_class}\n\n"
                    f"Вероятности:\n{probs_txt}\n"
                )
                app.after(0, lambda: set_result(text))
                app.after(0, lambda: status_var.set("Готово"))
            except Exception as exc:  # noqa: BLE001
                app.after(0, lambda: set_result("Ошибка при проверке.\n\n" + str(exc)))
                app.after(0, lambda: status_var.set("Ошибка"))
                app.after(0, lambda: messagebox.showerror("Ошибка", str(exc)))
            finally:
                app.after(0, progress.stop)
                app.after(0, progress.grid_remove)
                app.after(0, lambda: set_controls_state(True))

        threading.Thread(target=_run, daemon=True).start()

    def clear_result() -> None:
        set_result(
            "Результат очищен.\n\n"
            "Загрузите аудио и модель, затем нажмите «Проверить»."
        )
        status_var.set("Готов к проверке")

    pick_audio_btn = ttk.Button(buttons_row, text="Выбрать аудио", command=pick_audio)
    pick_audio_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

    pick_model_btn = ttk.Button(buttons_row, text="Выбрать модель", command=pick_model)
    pick_model_btn.grid(row=0, column=1, sticky="ew", padx=6)

    run_btn = ttk.Button(buttons_row, text="Проверить", command=classify)
    run_btn.grid(row=0, column=2, sticky="ew", padx=6)

    clear_btn = ttk.Button(buttons_row, text="Очистить", command=clear_result)
    clear_btn.grid(row=0, column=3, sticky="ew", padx=(6, 0))

    app.bind("<Return>", lambda _e: classify())

    app.mainloop()


if __name__ == "__main__":
    run_gui()
