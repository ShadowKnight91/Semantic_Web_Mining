"""Shared chart style: one validated categorical order, recessive axes, a legend for every multi-series chart."""
from __future__ import annotations

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
# fixed categorical order: a model keeps its colour in every chart
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_LABELS = {"E1__lr": "Bag of words (E1 LR)", "E2__lr": "Words + KG (E2 LR)", "E2G__lr": "KG only (E2G LR)",
                "E3__distilbert": "DistilBERT", "E3__sbert_mlp": "Sentence-BERT MLP",
                "E4__graphsage": "GraphSAGE", "E4__graphsage_text": "GraphSAGE, text features only",
                "E4__graphsage_topo": "GraphSAGE, structure only"}
MODEL_COLORS = {"E1__lr": SERIES[0], "E2__lr": SERIES[1], "E2G__lr": SERIES[2], "E3__distilbert": SERIES[3],
                "E3__sbert_mlp": SERIES[4], "E4__graphsage": SERIES[5], "E4__graphsage_text": SERIES[6],
                "E4__graphsage_topo": SERIES[7]}


def style_axes(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
