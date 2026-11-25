"""README figures for genai-diffusion-lab, drawn from results.json (numbers copied from the executed notebooks)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).parent
G = json.loads((HERE / "results.json").read_text())
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
S1, S2 = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated pair)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def dreambooth():
    rows = G["db"]
    base = next(r for r in rows if r["rank"] == 0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for ax, key, title, ylab in [(axes[0], "dino", "Subject fidelity rises with training…", "DINO similarity to the real photos"),
                                 (axes[1], "clip_t", "…while prompt fidelity falls", "CLIP-T (image–prompt similarity)")]:
        style(ax)
        ax.axhline(base[key], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.annotate("base SD 1.5", (100 if key == "dino" else 300, base[key]), xytext=(2, 4), textcoords="offset points", color=INK2, fontsize=8.5)
        for rank, color in [(4, S1), (32, S2)]:
            pts = sorted((r for r in rows if r["rank"] == rank), key=lambda r: r["step"])
            xs, ys = [p["step"] for p in pts], [p[key] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"LoRA r = {rank} ({pts[0]['adapter_mb']:.1f} MB)")
            ax.annotate(f"r = {rank}", (xs[-1], ys[-1]), xytext=(7, -3), textcoords="offset points", color=INK, fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks([100, 250, 500, 1000]); ax.set_xticklabels(["100", "250", "500", "1000"])
        ax.set_xlim(80, 1500)
        ax.set_xlabel("training steps (log scale)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.savefig(FIG / "dreambooth_tradeoff.png", dpi=170)
    plt.close(fig)


def inference():
    rows = sorted(G["inf"], key=lambda r: r["s_per_img"], reverse=True)
    ref = next(r for r in G["inf"] if r["name"].startswith("fp32"))
    short = lambda n: n.replace("fp16 · ", "").replace("DPM-Solver++", "DPM++").replace(" + VAE & attention slicing", " + slicing")
    names = [short(r["name"]) for r in rows]
    y = range(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 0.42 * len(rows) + 1.3), sharey=True, constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (a1, a2):
        ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    a1.barh(list(y), [r["s_per_img"] for r in rows], height=0.62, color=S1)
    for i, r in enumerate(rows):
        a1.annotate(f"{r['s_per_img']:.2f} s  ({r['speedup']:.1f}×)", (r["s_per_img"], i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    a1.set_xlim(0, max(r["s_per_img"] for r in rows) * 1.35)
    a1.set_yticks(list(y)); a1.set_yticklabels(names, fontsize=9)
    a1.set_xlabel("seconds per 512×512 image on a T4")
    a1.set_title("Latency (fp16 unless marked fp32)")
    a2.axvline(ref["clip_t"], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    a2.annotate("fp32 reference", (ref["clip_t"], len(rows) - 0.45), xytext=(3, 0), textcoords="offset points", fontsize=8, color=INK2)
    a2.scatter([r["clip_t"] for r in rows], list(y), s=46, color=S2, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    lo = min(r["clip_t"] for r in rows); hi = max(r["clip_t"] for r in rows)
    a2.set_xlim(lo - 0.006, hi + 0.006)
    a2.set_xlabel("CLIP-T (prompt alignment)")
    a2.set_title("Quality proxy")
    fig.savefig(FIG / "inference_benchmark.png", dpi=170)
    plt.close(fig)


def nb1():
    if "nb1" not in G:
        return
    E = G["nb1"]["exp1"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.6, 3.8), constrained_layout=True)
    series = [("ddpm", S1, "DDPM + DDIM"), ("fm", S2, "flow matching + Euler")]
    for ax in (a1, a2):
        style(ax)
    for kind, color, label in series:
        pts = sorted((r for r in E if r["kind"] == kind and abs(r["w"] - 2.0) < 1e-9), key=lambda r: r["steps"])
        xs, ys, err = [r["steps"] for r in pts], [r["fid"] for r in pts], [r["fid_sd"] for r in pts]
        a1.errorbar(xs, ys, yerr=err, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, capsize=3, label=label)
        a1.annotate("DDPM" if kind == "ddpm" else "flow matching", (xs[-1], ys[-1]), xytext=(7, -3), textcoords="offset points", fontsize=9)
        g = sorted((r for r in E if r["kind"] == kind and r["steps"] == 32), key=lambda r: r["w"])
        a2.errorbar([r["w"] for r in g], [r["fid"] for r in g], yerr=[r["fid_sd"] for r in g], color=color, linewidth=2, marker="o",
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=3, label=label)
    steps = sorted({r["steps"] for r in E if abs(r["w"] - 2.0) < 1e-9})
    a1.set_xscale("log", base=2); a1.set_xticks(steps); a1.set_xticklabels([str(x) for x in steps])
    a1.set_xlim(steps[0] * 0.8, steps[-1] * 2.6)
    a1.set_xlabel("sampling steps (log scale)"); a1.set_ylabel("FID (lower is better)")
    a1.set_title("Quality vs steps at guidance w = 2")
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")
    a2.set_xlabel("guidance scale w (32 steps)"); a2.set_ylabel("FID (lower is better)")
    a2.set_title("The ranking flips with guidance")
    best = min((r for r in E if r["steps"] == 32), key=lambda r: r["fid"])
    a2.annotate(f"best FID {best['fid']:.1f}\nflow matching, w = {best['w']:g}", (best["w"], best["fid"]), xytext=(1.25, 150),
                textcoords="data", fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for kind, label in (("ddpm", "DDPM"), ("fm", "flow matching")):
        last = max((r for r in E if r["kind"] == kind and r["steps"] == 32), key=lambda r: r["w"])
        a2.annotate(label, (last["w"], last["fid"]), xytext=(7, -3), textcoords="offset points", fontsize=9)
    a2.set_xlim(-0.3, 6.6)
    style(a3); a3.grid(axis="x", color=GRID, linewidth=0.8)
    names = {"emb": "embedding only", "lora4": "LoRA r = 4", "lora16": "LoRA r = 16", "full": "full fine-tune"}
    for r in G["nb1"]["exp2"]:
        x, y = r["forgetting (ΔFID old)"], r["new FID"]
        a3.errorbar([x], [y], xerr=[r.get("± old FID") or 0], yerr=[r.get("± new FID") or 0], color=S1, marker="o",
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=3, linewidth=1.2)
        off = {"lora4": (-6, 12), "lora16": (6, 12)}.get(r["method"], (8, 4))
        a3.annotate(names.get(r["method"], r["method"]), (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=INK,
                    ha="right" if r["method"] == "lora4" else "left")
    a3.set_xlabel("forgetting: rise in old-class FID (lower is better)")
    a3.set_ylabel("new-class FID (lower is better)")
    a3.set_title("Learning 2 new classes vs forgetting 8")
    a3.set_ylim(bottom=0)
    xs = [r["forgetting (ΔFID old)"] for r in G["nb1"]["exp2"]]
    a3.set_xlim(min(xs) - 120, max(xs) + 420)
    fig.savefig(FIG / "flow_matching_vs_ddpm_and_lora.png", dpi=170)
    plt.close(fig)

if __name__ == "__main__":
    dreambooth()
    inference()
    nb1()
    print("figures written to", FIG)
