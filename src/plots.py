"""Funkcje do analizy wyników treningu modeli YOLO """

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

C_TRAIN, C_VAL = "#2a78d6", "#008300"  # kolory: niebieski = train, zielony = val
METRIC_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]

METRICS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


def load_results_csv(results_csv: str | Path) -> pd.DataFrame:
    """Wczytuje results.csv wygenerowany przez Ultralytics podczas treningu."""
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    return df


def plot_learning_curves(results_csv: str | Path, title: str = "Krzywe uczenia") -> plt.Figure:
    """Rysuje krzywe uczenia: trzy panele strat (train vs val) i panel metryk walidacyjnych.

    Parameters
    ----------
    results_csv : ścieżka do runs/detect/<run>/results.csv
    title : tytuł całej figury (np. nazwa modelu i liczba epok)
    """
    df = load_results_csv(results_csv)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # --- trzy panele strat: train vs val ---
    for ax, loss in zip(axes.flat, ["box_loss", "cls_loss", "dfl_loss"]):
        ax.plot(df["epoch"], df[f"train/{loss}"], color=C_TRAIN, lw=1.8, label="train")
        ax.plot(df["epoch"], df[f"val/{loss}"], color=C_VAL, lw=1.8, ls="--", label="val")
        ax.set_title(loss)
        ax.set_xlabel("epoka")
        ax.legend()

    # --- panel metryk ---
    ax = axes[1, 1]
    for (name, col), color in zip(METRICS.items(), METRIC_COLORS):
        ax.plot(df["epoch"], df[col], color=color, lw=1.8, label=name)

    # zaznaczenie najlepszej epokę wg mAP50-95 (z niej pochodzi best.pt)
    best = df["metrics/mAP50-95(B)"].idxmax()
    ax.axvline(df.loc[best, "epoch"], color="#898781", ls=":", lw=1)
    ax.annotate(
        f"best: epoka {int(df.loc[best, 'epoch'])}",
        xy=(df.loc[best, "epoch"], df.loc[best, "metrics/mAP50-95(B)"]),
        xytext=(5, -12),
        textcoords="offset points",
        fontsize=9,
        color="#52514e",
    )
    ax.set_title("Metryki walidacyjne")
    ax.set_xlabel("epoka")
    ax.set_ylim(0, 1.05)
    ax.legend()

    for ax in axes.flat:
        ax.grid(color="#e1e0d9", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_predictions(
    model,
    image_paths,
    title: str = "Przykładowe wykrycia",
    cols: int = 3,
    conf: float = 0.25,
) -> plt.Figure:
    """Odpala inferencję modelu na podanych obrazkach i rysuje wykryte obiekty.

    Parameters
    ----------
    model : wytrenowany model ultralytics.YOLO
    image_paths : lista ścieżek do obrazków
    title : tytuł całej figury (np. nazwa modelu)
    cols : liczba kolumn w gridzie
    conf : próg confidence dla predykcji
    """
    image_paths = list(image_paths)
    rows = math.ceil(len(image_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flat

    results = model.predict(image_paths, conf=conf, verbose=False)
    for ax, result in zip(axes, results):
        annotated = result.plot()[..., ::-1]  # BGR -> RGB
        ax.imshow(annotated)
        ax.set_title(Path(result.path).name, fontsize=9)
        ax.axis("off")

    for ax in list(axes)[len(image_paths):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
