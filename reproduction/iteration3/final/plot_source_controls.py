from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path(__file__).parent
rows = json.loads((root / "source_control_results.json").read_text())["rows"]
models = [
    "Qwen/Qwen3.5-4B",
    "anthropic/claude-haiku-5.5",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.8-27b",
    "anthropic/claude-opus-5.5",
]
labels = ["Qwen3.5 4B", "Haiku5.5", "Qwen3.5 27B", "Qwen3.8 27B", "Opus5.5 (low)"]
fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
for ax, subgroup, title in zip(
    axes,
    ["all", "short", "descriptive"],
    ["All 24 families", "Short options · 13 families", "Descriptive options · 11 families"],
):
    for offset, kind, color, label in [
        (-0.13, "no_story", "#c8641d", "No story"),
        (0.13, "full_story", "#136f9c", "Full story"),
    ]:
        selected = [
            next(r for r in rows if r["model"] == m and r["kind"] == kind and r["subgroup"] == subgroup)
            for m in models
        ]
        errors = [
            [r["accuracy"] - r["parent_bootstrap_95_ci"][0] for r in selected],
            [r["parent_bootstrap_95_ci"][1] - r["accuracy"] for r in selected],
        ]
        ax.errorbar([r["accuracy"] for r in selected], [i + offset for i in range(5)], xerr=errors, fmt="o", color=color, capsize=3, label=label)
    ax.set_title(title)
    ax.set_xlim(0, 1.04)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.axvline(0.5, color="#999999", ls=":", lw=1)
    ax.grid(axis="x", alpha=0.15)
    ax.set_xlabel("Answer accuracy")
axes[0].set_yticks(range(5), labels)
axes[0].invert_yaxis()
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.55, 0.91), ncol=2, frameon=False)
fig.suptitle("Fresh source controls: reading the story versus inferring from the options", fontsize=13)
fig.text(0.5, 0.015, "Both answer orders; 95% family-bootstrap intervals. Perfect-sample intervals do not imply population certainty.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.055, 1, 0.86))
fig.savefig(root / "source_controls.png", dpi=180)
fig.savefig(root / "source_controls.svg")
plt.close(fig)
