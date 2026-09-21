"""Figures for REPORT.md: python make_figures.py  (needs matplotlib)"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333"))

# 1. pipeline
fig, ax = plt.subplots(figsize=(13, 4.4)); ax.set_xlim(0, 13.4); ax.set_ylim(0, 4.4); ax.axis("off")
def b(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=c, ec="#333", lw=1.2)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.3)
b(0.1, 2.9, 2.6, 1.1, "1. COLLECT\nPubMed Central, MedlinePlus,\nGhana guidelines, Q&A sets", "#dbe9f6")
b(3.2, 2.9, 2.6, 1.1, "2. PREPARE (on laptop)\nclean, split, and hold out\nthe test text FIRST", "#dbe9f6")
b(6.3, 2.9, 2.6, 1.1, "3. STAGE 1: read widely\ncontinued pre-training\n(next-word prediction)", "#fde9c9")
b(9.4, 2.9, 2.9, 1.1, "4. STAGE 2: learn to answer\ninstruction tuning\n(question to GP-style answer)", "#fde9c9")
for x1, x2 in ((2.7, 3.2), (5.8, 6.3), (8.9, 9.4)): arrow(ax, x1, 3.45, x2, 3.45)
b(3.2, 0.5, 2.6, 1.0, "Model A: BASE\n(untouched Qwen3.5-4B-Base)", "#eeeeee")
b(6.3, 0.5, 2.6, 1.0, "Model B:\nbase + stage 1", "#eeeeee")
b(9.4, 0.5, 2.9, 1.0, "Model C:\nbase + stage 1 + stage 2", "#eeeeee")
arrow(ax, 7.6, 2.9, 7.6, 1.5); arrow(ax, 10.85, 2.9, 10.85, 1.5); arrow(ax, 5.8, 1.0, 6.3, 1.0); arrow(ax, 8.9, 1.0, 9.4, 1.0)
ax.add_patch(FancyBboxPatch((0.1, 0.3), 2.6, 1.4, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#d8f0d8", ec="#333", lw=1.2))
ax.text(1.4, 1.0, "5. EVALUATE\nthe SAME held-out tests\nfor models A, B and C", ha="center", va="center", fontsize=8.3)
ax.plot([4.5, 4.5, 10.85, 10.85], [0.5, 0.36, 0.36, 0.5], color="#333", lw=1.4); ax.plot([7.6, 7.6], [0.5, 0.36], color="#333", lw=1.4)
arrow(ax, 4.5, 0.36, 2.7, 0.36)
ax.text(0.1, 2.35, "Each stage saves a small add-on (a LoRA adapter);\nthe base model itself is never changed.", fontsize=8.3, style="italic")
plt.savefig("report_assets/pipeline.png", dpi=170, bbox_inches="tight"); plt.close()

# 2. LoRA
fig, ax = plt.subplots(figsize=(9.5, 3.6)); ax.set_xlim(0, 9.5); ax.set_ylim(0, 3.6); ax.axis("off")
def b2(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=c, ec="#333", lw=1.2)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
b2(0.2, 1.3, 2.6, 1.6, "Pre-trained weights W\n(billions of numbers)\nFROZEN: never changed", "#dddddd")
b2(3.5, 0.25, 1.35, 0.9, "A  (r x d)\nsmall", "#fde9c9"); b2(5.05, 0.25, 1.35, 0.9, "B  (d x r)\nsmall", "#fde9c9")
ax.text(4.75, 1.3, "only A and B are trained", ha="center", fontsize=8.5, style="italic")
b2(6.9, 1.3, 2.4, 1.6, "Weights actually used\nW' = W + (alpha / r) x B x A", "#d8f0d8")
arrow(ax, 2.8, 2.1, 6.9, 2.1); arrow(ax, 6.4, 0.75, 7.6, 1.3)
ax.text(4.75, 3.3, "LoRA: teach the model by training a tiny add-on, not by rewriting all of it", ha="center", fontsize=10, weight="bold")
plt.savefig("report_assets/lora.png", dpi=170, bbox_inches="tight"); plt.close()
print("figures written")
