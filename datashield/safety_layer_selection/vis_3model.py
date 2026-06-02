import os
import json
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

configs = [
    {
        "folder_path": os.path.join(BASE_DIR, "compliance-aware-score", "safety", "llama3-8B-Ins"),
        "num_layers": 32,
        "title": "LLama-3-8B-Instruct",
    },
    {
        "folder_path": os.path.join(BASE_DIR, "compliance-aware-score", "safety", "llama3.1-8B-Ins"),
        "num_layers": 32,
        "title": "LLama-3.1-8B-Instruct",
    },
    {
        "folder_path": os.path.join(BASE_DIR, "compliance-aware-score", "safety", "Qwen2.5-7B"),
        "num_layers": 28,
        "title": "Qwen2.5-7B-Instruct",
    },
]

save_dir = os.path.join(BASE_DIR, "pics")
os.makedirs(save_dir, exist_ok=True)

color_score = "#b299c8"

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=False, sharex=False)

plt.subplots_adjust(
    left=0.5,
    wspace=0.2,
)

for idx, (ax, cfg) in enumerate(zip(axs, configs)):
    folder = cfg["folder_path"]
    n_layers = cfg["num_layers"]
    sub_title = cfg["title"]

    compliance_aware_scores = []

    for i in range(n_layers):
        file_path = os.path.join(folder, f"L{i}.json")

        with open(file_path, "r") as f:
            data = json.load(f)
            compliance_aware_scores.append(data["compliance_aware_score"])

    layers = np.arange(1, n_layers + 1)

    max_score = max(compliance_aware_scores)
    max_score_pos = compliance_aware_scores.index(max_score) + 1

    ax.bar(layers, compliance_aware_scores, width=0.6, color=color_score)

    ax.annotate(
        f"Layer {max_score_pos}",
        xy=(max_score_pos, max_score),
        xytext=(max_score_pos, max_score * 1.1),
        ha="center",
        fontsize=12,
        color="crimson",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1),
    )

    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(sub_title, fontsize=12)
    ax.grid(True, linewidth=0.5, linestyle="--")
    ax.set_ylim(0, max_score * 1.25)
    ax.set_xlabel("Layer", fontsize=12, labelpad=2)

fig.text(
    0.0,
    0.5,
    "Compliance-Aware Score",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=12,
)

fig.tight_layout()

save_path_png = os.path.join(save_dir, "layer_compliance_aware_scores_three.png")
save_path_pdf = os.path.join(save_dir, "layer_compliance_aware_scores_three.pdf")

plt.savefig(save_path_png, dpi=500, bbox_inches="tight")
plt.savefig(save_path_pdf, bbox_inches="tight")
plt.close()

print(f"✅ PNG 已保存：{save_path_png}")
print(f"✅ PDF 已保存：{save_path_pdf}")