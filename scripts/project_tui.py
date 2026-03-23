"""Русский TUI-центр управления Voice Anti-Spoofing (cyber-style)."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import yaml

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
except Exception as exc:  # noqa: BLE001
    print("Для TUI нужен пакет rich. Установите: pip install rich")
    raise SystemExit(1) from exc

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"

console = Console()

PRESETS = {
    "1": {
        "title": "Быстрый",
        "desc": "Быстрые итерации и проверка гипотез",
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "hidden_dim": 96,
            "dropout": 0.25,
            "epochs": 8,
            "early_stopping_patience": 3,
        },
    },
    "2": {
        "title": "Сбалансированный",
        "desc": "Основной режим обучения",
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "hidden_dim": 128,
            "dropout": 0.30,
            "epochs": 20,
            "early_stopping_patience": 5,
        },
    },
    "3": {
        "title": "Финальный",
        "desc": "Максимум качества, дольше по времени",
        "training": {
            "batch_size": 128,
            "learning_rate": 0.0007,
            "weight_decay": 0.0001,
            "hidden_dim": 192,
            "dropout": 0.35,
            "epochs": 35,
            "early_stopping_patience": 7,
        },
    },
}


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def read_metrics() -> dict | None:
    p = ROOT / "logs" / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def count_classes(csv_path: Path) -> dict[str, int]:
    if not csv_path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "class_name" not in df.columns:
            return {}
        return dict(df["class_name"].value_counts().sort_index())
    except Exception:  # noqa: BLE001
        return {}


def ok_mark(v: bool) -> str:
    return "[bold #27d17f]OK[/]" if v else "[bold #ff6b6b]нет[/]"


def run_cmd(cmd: list[str], cwd: Path = ROOT) -> int:
    console.print(f"\n[bold #64d3ff]$ {' '.join(shlex.quote(c) for c in cmd)}[/]\n")
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def sparkline(values: list[float]) -> str:
    if not values:
        return "-"
    bars = "▁▂▃▄▅▆▇█"
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-9:
        return bars[0] * len(values)
    out = []
    for v in values:
        idx = int((v - vmin) / (vmax - vmin) * (len(bars) - 1))
        out.append(bars[idx])
    return "".join(out)


def render_dashboard(cfg: dict) -> None:
    console.clear()

    now = datetime.now().strftime("%H:%M:%S")
    header = Text(" VOICE ANTI-SPOOFING TUI ", style="bold #64d3ff on #0b1a22")
    header.append(f"   {now}", style="bold #c9f9ff on #0b1a22")
    console.print(Panel(header, border_style="#1f8ea8", box=box.SQUARE, padding=(0, 1)))

    data = cfg.get("data", {})
    train_csv = ROOT / data.get("train_csv", "data/splits/train.csv")
    val_csv = ROOT / data.get("val_csv", "data/splits/val.csv")
    test_csv = ROOT / data.get("test_csv", "data/splits/test.csv")
    model_pt = ROOT / "logs" / "best_model.pt"

    status = Table.grid(expand=True)
    status.add_column(ratio=1)
    status.add_column(ratio=1)

    left = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    left.add_column(style="#a6f6ff")
    left.add_column(style="#e6faff")
    left.add_row("Корень", str(ROOT))
    left.add_row("Конфиг", str(CONFIG_PATH))
    left.add_row("Train", f"{ok_mark(train_csv.exists())}  {train_csv}")
    left.add_row("Val", f"{ok_mark(val_csv.exists())}  {val_csv}")
    left.add_row("Test", f"{ok_mark(test_csv.exists())}  {test_csv}")
    left.add_row("Модель", f"{ok_mark(model_pt.exists())}  {model_pt}")

    tr = cfg.get("training", {})
    right = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    right.add_column(style="#a6f6ff")
    right.add_column(style="#e6faff")
    right.add_row("batch_size", str(tr.get("batch_size")))
    right.add_row("learning_rate", str(tr.get("learning_rate")))
    right.add_row("weight_decay", str(tr.get("weight_decay")))
    right.add_row("hidden_dim", str(tr.get("hidden_dim")))
    right.add_row("dropout", str(tr.get("dropout")))
    right.add_row("epochs", str(tr.get("epochs")))
    right.add_row("patience", str(tr.get("early_stopping_patience")))

    status.add_row(
        Panel(left, title="[bold #ff5fa2]Состояние[/]", border_style="#2bb3d6", box=box.SQUARE),
        Panel(right, title="[bold #ff5fa2]Параметры[/]", border_style="#2bb3d6", box=box.SQUARE),
    )
    console.print(status)

    middle = Table.grid(expand=True)
    middle.add_column(ratio=1)
    middle.add_column(ratio=1)

    actions = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold #64d3ff")
    actions.add_column("Клавиша", style="#ff8fb8", width=10)
    actions.add_column("Действие", style="#e6faff")
    actions.add_row("1", "Статус проекта и пути")
    actions.add_row("2", "Применить пресет (quick / balanced / best)")
    actions.add_row("3", "Ручная правка параметров обучения")
    actions.add_row("4", "Собрать train/val/test сплиты")
    actions.add_row("5", "Проверить сплиты (валидация)")
    actions.add_row("6", "Запустить обучение (live лог)")
    actions.add_row("7", "Показать метрики")
    actions.add_row("8", "Проверить один файл")
    actions.add_row("9", "Открыть GUI")
    actions.add_row("0", "Выход")

    cls_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold #64d3ff")
    cls_table.add_column("Split", style="#ff8fb8")
    cls_table.add_column("Классы", style="#e6faff")

    for name, p in (("train", train_csv), ("val", val_csv), ("test", test_csv)):
        counts = count_classes(p)
        txt = ", ".join([f"{k}:{v}" for k, v in counts.items()]) if counts else "нет данных"
        cls_table.add_row(name, txt)

    middle.add_row(
        Panel(actions, title="[bold #ff5fa2]Меню[/]", border_style="#2bb3d6", box=box.SQUARE),
        Panel(cls_table, title="[bold #ff5fa2]Сводка сплитов[/]", border_style="#2bb3d6", box=box.SQUARE),
    )
    console.print(middle)

    metrics = read_metrics()
    if metrics:
        mtable = Table(box=box.SIMPLE_HEAVY, show_header=False)
        mtable.add_column(style="#a6f6ff")
        mtable.add_column(style="#e6faff")
        for key in ("val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"):
            if key in metrics:
                mtable.add_row(key, f"{metrics[key]:.4f}")
        console.print(Panel(mtable, title="[bold #ff5fa2]Последние метрики[/]", border_style="#2bb3d6", box=box.SQUARE))

    console.print("[bold #64d3ff]Горячие клавиши: нажмите 0-9 (без Enter)")


def read_key() -> str:
    if not sys.stdin.isatty():
        return Prompt.ask("Команда", choices=[str(i) for i in range(10)], default="1")

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in [str(i) for i in range(10)]:
                    return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return Prompt.ask("Команда", choices=[str(i) for i in range(10)], default="1")


def wait_enter() -> None:
    Prompt.ask("\n[bold #64d3ff]Нажмите Enter для возврата в меню[/]", default="")


def apply_preset(cfg: dict) -> None:
    console.print("\n[bold #64d3ff]Выбор пресета:[/]")
    for idx, p in PRESETS.items():
        console.print(f"  [#ff8fb8]{idx}[/]. [bold]{p['title']}[/] — {p['desc']}")
    raw = Prompt.ask("Номер пресета", choices=list(PRESETS.keys()), default="2")
    picked = PRESETS[raw]
    cfg.setdefault("training", {}).update(picked["training"])
    write_yaml(CONFIG_PATH, cfg)
    console.print(f"[bold #27d17f]Применён пресет:[/] {picked['title']}")


def edit_training(cfg: dict) -> None:
    tr = cfg.setdefault("training", {})
    fields = [
        ("batch_size", int),
        ("learning_rate", float),
        ("weight_decay", float),
        ("hidden_dim", int),
        ("dropout", float),
        ("epochs", int),
        ("early_stopping_patience", int),
    ]
    console.print("\n[bold #64d3ff]Редактирование параметров[/] (Enter = оставить текущее)")
    for name, typ in fields:
        old = tr.get(name)
        raw = Prompt.ask(f"{name} [{old}]", default="")
        if raw == "":
            continue
        try:
            tr[name] = typ(raw)
        except ValueError:
            console.print(f"[bold #ff6b6b]Неверный формат:[/] {name}")
    write_yaml(CONFIG_PATH, cfg)
    console.print("[bold #27d17f]Параметры сохранены[/]")


def make_splits() -> None:
    max_per_class = Prompt.ask("max_per_class (0 = без лимита)", default="2000")
    rc = run_cmd([
        sys.executable,
        "scripts/make_splits.py",
        "--root",
        ".",
        "--max-per-class",
        max_per_class,
    ])
    console.print("[bold #27d17f]Готово[/]" if rc == 0 else "[bold #ff6b6b]Ошибка[/]")


def validate_splits() -> None:
    allow = Prompt.ask("Разрешить пересечение speaker_id? (y/n)", choices=["y", "n"], default="n")
    cmd = [
        sys.executable,
        "scripts/validate_dataset.py",
        "--root",
        ".",
        "--train",
        "data/splits/train.csv",
        "--val",
        "data/splits/val.csv",
        "--test",
        "data/splits/test.csv",
    ]
    if allow == "y":
        cmd.append("--allow-speaker-overlap")
    rc = run_cmd(cmd)
    console.print("[bold #27d17f]OK[/]" if rc == 0 else "[bold #ff6b6b]Есть ошибки[/]")


def train_model_live() -> None:
    cmd = [sys.executable, "-m", "src.train", "--config", "configs/config.yaml", "--root", "."]
    console.print(f"\n[bold #64d3ff]$ {' '.join(shlex.quote(c) for c in cmd)}[/]\n")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = deque(maxlen=24)
    f1_vals: list[float] = []

    with Live(console=console, refresh_per_second=8) as live:
        for raw in proc.stdout or []:
            line = raw.rstrip("\n")
            lines.append(line)
            m = re.search(r"val_f1=([0-9.]+)", line)
            if m:
                try:
                    f1_vals.append(float(m.group(1)))
                    if len(f1_vals) > 40:
                        f1_vals = f1_vals[-40:]
                except ValueError:
                    pass

            log_text = Text("\n".join(lines), style="#d7f8ff")
            spark = sparkline(f1_vals)
            stat = Table(box=box.SIMPLE_HEAVY, show_header=False)
            stat.add_column(style="#a6f6ff")
            stat.add_column(style="#e6faff")
            stat.add_row("val_f1 trend", spark)
            if f1_vals:
                stat.add_row("last val_f1", f"{f1_vals[-1]:.4f}")
                stat.add_row("best val_f1", f"{max(f1_vals):.4f}")

            group = Group(
                Panel(log_text, title="[bold #ff5fa2]Live лог обучения[/]", border_style="#2bb3d6", box=box.SQUARE),
                Panel(stat, title="[bold #ff5fa2]Мини-график[/]", border_style="#2bb3d6", box=box.SQUARE),
            )
            live.update(group)

    rc = proc.wait()
    console.print("\n[bold #27d17f]Обучение завершено[/]" if rc == 0 else "\n[bold #ff6b6b]Ошибка обучения[/]")


def show_metrics() -> None:
    metrics = read_metrics()
    if not metrics:
        console.print("[bold #ff6b6b]Файл logs/metrics.json не найден[/]")
        return
    console.print_json(json.dumps(metrics, ensure_ascii=False))


def infer_one_file() -> None:
    audio = Prompt.ask("Путь к аудиофайлу")
    if not audio.strip():
        return
    cmd = [
        sys.executable,
        "-c",
        (
            "from src.inference import predict_from_wav;"
            f"pred,is_fake,probs,_=predict_from_wav({audio!r},'logs/best_model.pt');"
            "print('class=',pred);print('is_fake=',is_fake);"
            "print('probs=',sorted(probs.items(), key=lambda x: x[1], reverse=True))"
        ),
    ]
    rc = run_cmd(cmd)
    console.print("[bold #27d17f]Готово[/]" if rc == 0 else "[bold #ff6b6b]Ошибка инференса[/]")


def run_gui() -> None:
    rc = run_cmd([sys.executable, "-m", "src.gui"])
    console.print("[bold #27d17f]GUI закрыт[/]" if rc == 0 else "[bold #ff6b6b]GUI завершился с ошибкой[/]")


def main() -> int:
    while True:
        cfg = read_yaml(CONFIG_PATH)
        render_dashboard(cfg)
        choice = read_key()

        console.clear()
        if choice == "0":
            console.print("[bold #64d3ff]Выход[/]")
            return 0
        if choice == "1":
            console.print("[bold #27d17f]Статус обновлён[/]")
        elif choice == "2":
            apply_preset(cfg)
        elif choice == "3":
            edit_training(cfg)
        elif choice == "4":
            make_splits()
        elif choice == "5":
            validate_splits()
        elif choice == "6":
            train_model_live()
        elif choice == "7":
            show_metrics()
        elif choice == "8":
            infer_one_file()
        elif choice == "9":
            run_gui()

        wait_enter()


if __name__ == "__main__":
    raise SystemExit(main())
