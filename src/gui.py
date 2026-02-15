"""
Простой графический интерфейс для Voice Anti-Spoofing.
Выбор wav-файла → загрузка модели → предсказание Real/Fake.
"""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Корень проекта для дефолтных путей
ROOT = Path(__file__).resolve().parent.parent


def run_gui():
    from .inference import predict_from_wav

    def select_file():
        path = filedialog.askopenfilename(
            title="Выберите WAV-файл",
            filetypes=[("WAV", "*.wav"), ("Все файлы", "*.*")],
            initialdir=ROOT,
        )
        if path:
            wav_var.set(path)

    def select_model():
        path = filedialog.askopenfilename(
            title="Выберите чекпоинт модели",
            filetypes=[("PyTorch", "*.pt"), ("Все файлы", "*.*")],
            initialdir=ROOT / "logs",
        )
        if path:
            model_var.set(path)

    def classify():
        wav_path = wav_var.get().strip()
        model_path = model_var.get().strip()
        if not wav_path:
            messagebox.showwarning("Внимание", "Выберите WAV-файл.")
            return
        if not model_path:
            messagebox.showwarning("Внимание", "Выберите файл модели (.pt).")
            return
        if not Path(wav_path).exists():
            messagebox.showerror("Ошибка", f"Файл не найден:\n{wav_path}")
            return
        if not Path(model_path).exists():
            messagebox.showerror("Ошибка", f"Модель не найдена:\n{model_path}")
            return

        result_var.set("Обработка...")
        progress.start(10)

        def run():
            try:
                label, prob_real, prob_fake = predict_from_wav(wav_path, model_path)
                name = "Реальный голос" if label == 0 else "Синтетический голос"
                text = f"{name}\nReal: {prob_real:.1%}  |  Fake: {prob_fake:.1%}"
                app.after(0, lambda: result_var.set(text))
            except Exception as e:
                app.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                app.after(0, lambda: result_var.set("—"))
            finally:
                app.after(0, progress.stop)

        threading.Thread(target=run, daemon=True).start()

    app = tk.Tk()
    app.title("Voice Anti-Spoofing")
    app.geometry("480x280")
    app.resizable(True, True)

    # Стиль
    font_title = ("", 12, "bold")
    font_result = ("", 14, "bold")
    padding = {"padx": 12, "pady": 8}

    main = ttk.Frame(app, padding=16)
    main.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main, text="Voice Anti-Spoofing", font=font_title).pack(anchor=tk.W, **padding)

    # WAV
    f_wav = ttk.Frame(main)
    f_wav.pack(fill=tk.X, **padding)
    wav_var = tk.StringVar(value="")
    ttk.Label(f_wav, text="WAV-файл:").pack(side=tk.LEFT, padx=(0, 8))
    ttk.Entry(f_wav, textvariable=wav_var, width=42).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
    ttk.Button(f_wav, text="Обзор…", command=select_file).pack(side=tk.LEFT)

    # Модель
    f_model = ttk.Frame(main)
    f_model.pack(fill=tk.X, **padding)
    default_model = ROOT / "logs" / "best_model.pt"
    model_var = tk.StringVar(value=str(default_model) if default_model.exists() else "")
    ttk.Label(f_model, text="Модель (.pt):").pack(side=tk.LEFT, padx=(0, 8))
    ttk.Entry(f_model, textvariable=model_var, width=42).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
    ttk.Button(f_model, text="Обзор…", command=select_model).pack(side=tk.LEFT)

    # Кнопка проверки
    ttk.Button(main, text="Проверить", command=classify).pack(**padding)

    # Прогресс
    progress = ttk.Progressbar(main, mode="indeterminate", length=200)
    progress.pack(pady=4)

    # Результат
    result_var = tk.StringVar(value="—")
    ttk.Label(main, text="Результат:", font=("", 10)).pack(anchor=tk.W, **padding)
    result_label = ttk.Label(main, textvariable=result_var, font=font_result)
    result_label.pack(anchor=tk.W, **padding)

    app.mainloop()


if __name__ == "__main__":
    run_gui()
