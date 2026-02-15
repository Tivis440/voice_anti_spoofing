"""
Графический интерфейс для Voice Anti-Spoofing.
Выбор файлов через диалоги → проверка Real/Fake.
"""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent

# Контрастные цвета: чёрный текст на белом
BG = "#e8e8e8"
CARD_BG = "#ffffff"
BUTTON_BG = "#1a73e8"
GO_BG = "#1e7e34"
TEXT = "#000000"
TEXT_SECONDARY = "#333333"
# Системный шрифт — лучше читается на любой ОС
FONT = ("", 14)
FONT_TITLE = ("", 20, "bold")
FONT_RESULT = ("", 18, "bold")


def run_gui():
    from .inference import predict_from_wav

    def select_wav():
        path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("WAV", "*.wav"), ("Все файлы", "*.*")],
            initialdir=ROOT,
        )
        if path:
            wav_var.set(path)
            name = Path(path).name
            wav_label.config(text=name if len(name) <= 45 else name[:42] + "...")
            wav_label.config(fg=TEXT)

    def select_model():
        path = filedialog.askopenfilename(
            title="Выберите модель",
            filetypes=[("PyTorch", "*.pt"), ("Все файлы", "*.*")],
            initialdir=ROOT / "logs",
        )
        if path:
            model_var.set(path)
            name = Path(path).name
            model_label.config(text=name if len(name) <= 45 else name[:42] + "...")
            model_label.config(fg=TEXT)

    def classify():
        wav_path = wav_var.get().strip()
        model_path = model_var.get().strip()
        if not wav_path:
            messagebox.showwarning("Внимание", "Сначала выберите аудиофайл.")
            return
        if not model_path:
            messagebox.showwarning("Внимание", "Сначала выберите файл модели (.pt).")
            return
        if not Path(wav_path).exists():
            messagebox.showerror("Ошибка", f"Файл не найден:\n{wav_path}")
            return
        if not Path(model_path).exists():
            messagebox.showerror("Ошибка", f"Модель не найдена:\n{model_path}")
            return

        result_text.set("Обработка…")
        result_label.config(fg=TEXT)
        progress.start(8)

        def run():
            try:
                label, prob_real, prob_fake = predict_from_wav(wav_path, model_path)
                if label == 0:
                    name = "Реальный голос"
                    color = "#0d6b0d"
                else:
                    name = "Синтетический голос"
                    color = "#b91c1c"
                text = f"{name}\n\nReal: {prob_real:.0%}   Fake: {prob_fake:.0%}"
                app.after(0, lambda: result_text.set(text))
                app.after(0, lambda: result_label.config(fg=color))
            except Exception as e:
                app.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                app.after(0, lambda: result_text.set("—"))
                app.after(0, lambda: result_label.config(fg=TEXT))
            finally:
                app.after(0, progress.stop)

        threading.Thread(target=run, daemon=True).start()

    app = tk.Tk()
    app.title("Voice Anti-Spoofing")
    app.geometry("500x520")
    app.resizable(True, True)
    app.configure(bg=BG)

    pad = 24
    main = tk.Frame(app, bg=BG, padx=pad, pady=pad)
    main.pack(fill=tk.BOTH, expand=True)

    # Заголовок
    title = tk.Label(
        main,
        text="Voice Anti-Spoofing",
        font=FONT_TITLE,
        fg=TEXT,
        bg=BG,
    )
    title.pack(anchor=tk.W, pady=(0, 4))
    sub = tk.Label(
        main,
        text="Проверка: реальный или синтетический голос",
        font=FONT,
        fg=TEXT_SECONDARY,
        bg=BG,
    )
    sub.pack(anchor=tk.W, pady=(0, 24))

    # Блок: аудиофайл
    card_wav = tk.Frame(main, bg=CARD_BG, padx=20, pady=16)
    card_wav.pack(fill=tk.X, pady=(0, 12))
    tk.Label(
        card_wav,
        text="Аудиофайл",
        font=("", 13, "bold"),
        fg=TEXT,
        bg=CARD_BG,
    ).pack(anchor=tk.W)
    btn_wav = tk.Button(
        card_wav,
        text="  Выбрать WAV-файл  ",
        font=FONT,
        fg="white",
        bg=BUTTON_BG,
        activeforeground="white",
        activebackground="#1557b0",
        relief=tk.RAISED,
        padx=20,
        pady=10,
        cursor="hand2",
        command=select_wav,
    )
    btn_wav.pack(anchor=tk.W, pady=(10, 6))
    wav_var = tk.StringVar(value="")
    wav_label = tk.Label(
        card_wav,
        text="Файл не выбран",
        font=FONT,
        fg=TEXT_SECONDARY,
        bg=CARD_BG,
    )
    wav_label.pack(anchor=tk.W)

    # Блок: модель
    card_model = tk.Frame(main, bg=CARD_BG, padx=20, pady=16)
    card_model.pack(fill=tk.X, pady=(0, 12))
    tk.Label(
        card_model,
        text="Модель",
        font=("", 13, "bold"),
        fg=TEXT,
        bg=CARD_BG,
    ).pack(anchor=tk.W)
    tk.Label(
        card_model,
        text="Файл с обученной нейросетью (появляется после обучения). По умолчанию — лучшая сохранённая.",
        font=("", 11),
        fg=TEXT_SECONDARY,
        bg=CARD_BG,
    ).pack(anchor=tk.W)
    default_model = ROOT / "logs" / "best_model.pt"
    model_var = tk.StringVar(value=str(default_model) if default_model.exists() else "")
    if model_var.get():
        default_name = Path(model_var.get()).name
        model_placeholder = default_name if len(default_name) <= 45 else default_name[:42] + "..."
    else:
        model_placeholder = "Файл не выбран"
    btn_model = tk.Button(
        card_model,
        text="  Выбрать модель (.pt)  ",
        font=FONT,
        fg="white",
        bg=BUTTON_BG,
        activeforeground="white",
        activebackground="#1557b0",
        relief=tk.RAISED,
        padx=20,
        pady=10,
        cursor="hand2",
        command=select_model,
    )
    btn_model.pack(anchor=tk.W, pady=(10, 6))
    model_label = tk.Label(
        card_model,
        text=model_placeholder,
        font=FONT,
        fg=TEXT if model_var.get() else TEXT_SECONDARY,
        bg=CARD_BG,
    )
    model_label.pack(anchor=tk.W)

    # Кнопка «Проверить»
    btn_go = tk.Button(
        main,
        text="  П Р О В Е Р И Т Ь  ",
        font=("", 16, "bold"),
        fg="white",
        bg=GO_BG,
        activeforeground="white",
        activebackground="#176329",
        relief=tk.RAISED,
        padx=28,
        pady=14,
        cursor="hand2",
        command=classify,
    )
    btn_go.pack(pady=24)

    # Прогресс
    progress_bar = ttk.Progressbar(main, mode="indeterminate", length=260)

    def progress_start(_=None):
        progress_bar.pack(pady=8)
        progress_bar.start(8)

    def progress_stop(_=None):
        progress_bar.stop()
        progress_bar.pack_forget()

    class Progress:
        start = progress_start
        stop = progress_stop

    progress = Progress()

    # Результат
    result_frame = tk.Frame(main, bg=CARD_BG, padx=20, pady=20)
    result_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
    tk.Label(
        result_frame,
        text="Результат",
        font=("", 13, "bold"),
        fg=TEXT,
        bg=CARD_BG,
    ).pack(anchor=tk.W)
    result_text = tk.StringVar(value="—")
    result_label = tk.Label(
        result_frame,
        textvariable=result_text,
        font=FONT_RESULT,
        fg=TEXT,
        bg=CARD_BG,
        justify=tk.LEFT,
    )
    result_label.pack(anchor=tk.W, pady=(12, 0))

    app.mainloop()


if __name__ == "__main__":
    run_gui()
