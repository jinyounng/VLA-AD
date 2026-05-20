#!/usr/bin/env python3
"""
Visualize cross-view feature similarity results.

Reads the JSON outputs from cross_view_feature_similarity.py and generates:
  1. Histogram of same-object vs different-object cosine similarity
  2. Turn vs non-turn comparison
  3. Per-category breakdown
  4. View-pair heatmap
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(output_dir: str):
    with open(os.path.join(output_dir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(output_dir, "same_object_pairs.json")) as f:
        same = json.load(f)
    diff_path = os.path.join(output_dir, "diff_object_pairs.json")
    diff = []
    if os.path.exists(diff_path):
        with open(diff_path) as f:
            diff = json.load(f)
    return summary, same, diff


def plot_histogram(same_sims, diff_sims, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-0.2, 1.0, 60)

    if same_sims:
        ax.hist(same_sims, bins=bins, alpha=0.6, label=f"Same object (n={len(same_sims)})",
                color="#2196F3", density=True, edgecolor="white", linewidth=0.5)
    if diff_sims:
        ax.hist(diff_sims, bins=bins, alpha=0.6, label=f"Diff object (n={len(diff_sims)})",
                color="#FF5722", density=True, edgecolor="white", linewidth=0.5)

    if same_sims:
        ax.axvline(np.mean(same_sims), color="#1565C0", linestyle="--", linewidth=2,
                   label=f"Same mean={np.mean(same_sims):.3f}")
    if diff_sims:
        ax.axvline(np.mean(diff_sims), color="#BF360C", linestyle="--", linewidth=2,
                   label=f"Diff mean={np.mean(diff_sims):.3f}")

    ax.set_xlabel("Cosine Similarity", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_turn_comparison(same, diff, save_path):
    turn_same = [r["cosine_similarity"] for r in same if r.get("is_turn")]
    turn_diff = [r["cosine_similarity"] for r in diff if r.get("is_turn")]
    nonturn_same = [r["cosine_similarity"] for r in same if not r.get("is_turn")]
    nonturn_diff = [r["cosine_similarity"] for r in diff if not r.get("is_turn")]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    bins = np.linspace(-0.2, 1.0, 50)

    for ax, label, s_sims, d_sims in [
        (axes[0], "Turn Samples", turn_same, turn_diff),
        (axes[1], "Non-Turn Samples", nonturn_same, nonturn_diff),
    ]:
        if s_sims:
            ax.hist(s_sims, bins=bins, alpha=0.6, label=f"Same obj (n={len(s_sims)})",
                    color="#2196F3", density=True, edgecolor="white", linewidth=0.5)
        if d_sims:
            ax.hist(d_sims, bins=bins, alpha=0.6, label=f"Diff obj (n={len(d_sims)})",
                    color="#FF5722", density=True, edgecolor="white", linewidth=0.5)
        if s_sims:
            ax.axvline(np.mean(s_sims), color="#1565C0", linestyle="--", linewidth=2)
        if d_sims:
            ax.axvline(np.mean(d_sims), color="#BF360C", linestyle="--", linewidth=2)
        ax.set_xlabel("Cosine Similarity", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Density", fontsize=12)
    fig.suptitle("Turn vs Non-Turn: Cross-View Feature Similarity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_bar_summary(summary, save_path):
    categories = ["Overall", "Turn", "Non-Turn"]
    same_means = []
    diff_means = []

    for key in ["overall", "turn", "non_turn"]:
        s = summary[key]["same_object"]
        d = summary[key]["diff_object"]
        same_means.append(s["mean"] if s["n"] > 0 else 0)
        diff_means.append(d["mean"] if d["n"] > 0 else 0)

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, same_means, width, label="Same Object", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, diff_means, width, label="Diff Object", color="#FF5722", alpha=0.8)

    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title("Cross-View Feature Similarity Summary", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_view_pair_heatmap(same, save_path):
    CAMERAS = [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]
    n = len(CAMERAS)
    sim_matrix = np.full((n, n), np.nan)
    count_matrix = np.zeros((n, n), dtype=int)

    cam_idx = {c: i for i, c in enumerate(CAMERAS)}
    pair_sims = defaultdict(list)
    for r in same:
        a, b = r["view_a"], r["view_b"]
        pair_sims[(a, b)].append(r["cosine_similarity"])
        pair_sims[(b, a)].append(r["cosine_similarity"])

    for (a, b), sims in pair_sims.items():
        i, j = cam_idx.get(a), cam_idx.get(b)
        if i is not None and j is not None:
            sim_matrix[i, j] = np.mean(sims)
            count_matrix[i, j] = len(sims)

    fig, ax = plt.subplots(figsize=(9, 8))
    short_names = [c.replace("CAM_", "") for c in CAMERAS]

    masked = np.ma.masked_where(np.isnan(sim_matrix), sim_matrix)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(short_names, fontsize=11)

    for i in range(n):
        for j in range(n):
            if not np.isnan(sim_matrix[i, j]):
                ax.text(j, i, f"{sim_matrix[i,j]:.3f}\n(n={count_matrix[i,j]})",
                       ha="center", va="center", fontsize=9,
                       color="black" if sim_matrix[i, j] > 0.5 else "white")

    ax.set_title("Mean Cosine Similarity by Camera Pair", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_category(summary, save_path):
    per_cat = summary.get("per_category", {})
    cats = []
    same_vals = []
    diff_vals = []
    for cat, data in sorted(per_cat.items()):
        s_n = data["same"]["n"]
        if s_n >= 5:
            cats.append(cat.split(".")[-1] if "." in cat else cat)
            same_vals.append(data["same"]["mean"])
            diff_vals.append(data["diff"]["mean"] if data["diff"]["n"] > 0 else 0)

    if not cats:
        return

    y = np.arange(len(cats))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, len(cats) * 0.5)))
    ax.barh(y - height / 2, same_vals, height, label="Same Object", color="#2196F3", alpha=0.8)
    ax.barh(y + height / 2, diff_vals, height, label="Diff Object", color="#FF5722", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=10)
    ax.set_xlabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title("Per-Category Cross-View Similarity", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="workspace/cross_view_similarity/")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output dir for plots. Defaults to input_dir/plots/")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.input_dir} ...")
    summary, same, diff = load_results(args.input_dir)

    same_sims = [r["cosine_similarity"] for r in same]
    diff_sims = [r["cosine_similarity"] for r in diff]

    print("Generating plots ...")
    plot_histogram(same_sims, diff_sims, "Cross-View Feature Similarity: Same vs Different Object",
                   os.path.join(args.output_dir, "histogram_overall.png"))
    plot_turn_comparison(same, diff, os.path.join(args.output_dir, "histogram_turn_vs_nonturn.png"))
    plot_bar_summary(summary, os.path.join(args.output_dir, "bar_summary.png"))
    plot_view_pair_heatmap(same, os.path.join(args.output_dir, "view_pair_heatmap.png"))
    plot_per_category(summary, os.path.join(args.output_dir, "per_category.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
