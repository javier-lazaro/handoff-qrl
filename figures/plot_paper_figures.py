#!/usr/bin/env python3
"""Paper figures for paper (public paper evidence surface).

Reads only results/{lunarlander,minatar}/data and analysis/tables.
No statistics are computed here: every interval is read from a frozen table.
Per-seed differences are plotted only as raw paired points and are checked
against the tabulated paired means before plotting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MM = 1 / 25.4
W1, W2 = 84 * MM, 174 * MM  # Springer Nature single / double column

C_H = "#D55E00"      # hybrid VQC head
C_C = "#0072B2"      # compact classical head
C_FC = "#4D4D4D"     # full-capacity classical
C_KD = "#CC79A7"     # KD bridge
C_TEACH = "#009E73"  # teacher
GRAD_ALPHA, NOGRAD_ALPHA = 1.0, 0.45

ENVS = ("lunarlander", "minatar")
ENV_TITLE = {
    "lunarlander": "LunarLander",
    "minatar": "MinAtar Breakout",
}
INTERIOR = (0.25, 0.5, 0.75)
ATTR_T = (0.0, 0.25, 0.5, 0.75)
TOL = 1e-9


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.0, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def lab(t: float, head: str = "H") -> str:
    return f"{head}({t:g})"


class Data:
    def __init__(self, root: Path):
        self.root = root
        r = lambda e, f: pd.read_csv(root / "results" / e / "data" / f)
        ae = lambda e, f: pd.read_csv(root / "analysis/tables" / e / f)
        a = lambda f: pd.read_csv(root / "analysis/tables" / f)
        self.rq1 = {e: ae(e, "rq1_allocation.csv") for e in ENVS}
        self.rq2 = {e: ae(e, "rq2_attribution.csv") for e in ENVS}
        self.rq3 = {e: ae(e, "rq3_decomposition.csv") for e in ENVS}
        self.p3 = {e: ae(e, "p3_classical_continuation.csv") for e in ENVS}
        self.hm = {e: r(e, "handoff_matrix.csv") for e in ENVS}
        self.fc = {
            e: r(e, "full_classical_matrix.csv").set_index("seed")["final_return_mean"]
            for e in ENVS
        }
        self.cs = a("condition_summary.csv")
        self.fcr = a("fc_reference_contrasts.csv")
        self.res = a("resource_breakdown.csv")
        self.crc = a("continuation_reference_contrasts.csv")
        self.qfa = {e: r(e, "quantum_forward_accounting.csv") for e in ENVS}

    def seeds(self, env: str, cond: str) -> pd.Series:
        h = self.hm[env]
        return h[h["condition"] == cond].set_index("seed")["final_return_mean"].sort_index()

    def mean(self, env: str, cond: str) -> float:
        row = self.cs[(self.cs.environment == env) & (self.cs.condition == cond)]
        assert len(row) == 1, (env, cond)
        return float(row.mean_final_return.iloc[0])

    @staticmethod
    def contrast(table: pd.DataFrame, name: str) -> pd.Series:
        row = table[table.contrast == name]
        assert len(row) == 1, name
        return row.iloc[0]


FIG_NAMES = {  # order follows first citation in the manuscript (narrative v3)
    "fig1_design": "Fig1_draft", "fig2_allocation_frontier": "Fig2", "fig6_resources": "Fig3",
    "figA2_h1_retention": "Fig4", "fig3_attribution": "Fig5", "fig5_classical_continuation": "Fig6",
    "fig4_decomposition": "Fig7", "figA1_training_curves": "Fig8", "figA3_kd_diagnostics": "Fig9",
    "figA4_minatar_timing": "Fig10", "figA5_iqm_poi": "Fig11",
}


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    if len(panels) > 1:
        for i, ax in enumerate(panels):
            ax.annotate(f"({'abcdefgh'[i]})", (0, 1), xycoords="axes fraction", xytext=(-6, 9),
                        textcoords="offset points", ha="right", va="bottom", fontsize=8, fontweight="bold")
    name = FIG_NAMES.get(name, name)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------- F1 design
def _box(ax, x, y, w, h, text, fc="white", ec="0.3", tc="black", fs=7.2, bold=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.7, transform=ax.transAxes, clip_on=False))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, transform=ax.transAxes,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, colour="0.35"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=colour, shrinkA=0, shrinkB=0, mutation_scale=7))


def _draw_pipeline(ax) -> None:
    ax.set_axis_off()
    light = lambda c, a=0.16: mpl.colors.to_rgba(c, a)
    xc, w = 0.5, 0.94
    _box(ax, xc, 0.935, w, 0.085, "1  Classical PPO (full-capacity actor/critic)\nfor the first fraction $t$ of the budget",
         fc=light(C_FC, 0.12), ec=C_FC)
    _arrow(ax, xc, 0.892, xc, 0.858)
    _box(ax, xc, 0.815, w, 0.085, "2  KD dataset collected from the teacher\n65,536 train + 8,192 validation states",
         fc=light(C_KD), ec=C_KD)
    _arrow(ax, xc, 0.772, xc, 0.738)
    _box(ax, xc, 0.695, w, 0.085, "3  Distil policy and value into a compact head\nidentical data, recipe and budget for H and C",
         fc=light(C_KD), ec=C_KD)
    _arrow(ax, 0.36, 0.652, 0.25, 0.606)
    _arrow(ax, 0.64, 0.652, 0.75, 0.606)
    for x, col, title, body in (
        (0.25, C_H, "H($t$): VQC head", "8-qubit VQC\nangle encoding\nPauli-Z readout"),
        (0.75, C_C, "C($t$): classical head", "8-unit linear\nlayer + tanh\n(matched width)"),
    ):
        _box(ax, x, 0.58, 0.46, 0.05, title, fc=col, ec=col, tc="white", bold=True, fs=7.2)
        _box(ax, x, 0.475, 0.46, 0.13, body + "\n→ MLP actor / critic", fc=light(col, 0.10), ec=col, fs=7.2)
    _arrow(ax, 0.25, 0.408, 0.40, 0.345)
    _arrow(ax, 0.75, 0.408, 0.60, 0.345)
    _box(ax, xc, 0.30, w, 0.085, "4  Online PPO with the installed head\nfor the remaining fraction $1-t$", fc="white")
    _arrow(ax, xc, 0.257, xc, 0.223)
    _box(ax, xc, 0.18, w, 0.085, "5  Common evaluation: 100 stochastic episodes,\nfixed seeds; 20 training seeds per condition",
         fc="white")
    ax.text(xc, 0.09, "$t$=0: steps 1–3 skipped (head trained from scratch)\n$t$=1: step 4 skipped (KD only)",
            ha="center", va="center", fontsize=5.8, color="0.3", transform=ax.transAxes, linespacing=1.3)
    ax.text(xc, 0.025, "Quantum cost: sample-level forward evaluations (grad / no-grad)",
            ha="center", va="center", fontsize=5.8, style="italic", color="0.3", transform=ax.transAxes)


def _draw_schedule(ax) -> None:
    rows = [("Full classical", None, "fc"), (lab(1.0), 1.0, "h1")]
    for t in (0.75, 0.5, 0.25):
        rows += [(lab(t), t, "H"), (lab(t, "C"), t, "C")]
    rows += [(lab(0.0), 0.0, "H"), (lab(0.0, "C"), 0.0, "C")]
    h = 0.62
    ys, y = [], 0.0
    for i, (_, t, kind) in enumerate(rows):
        if i > 0 and kind in ("H", "h1"):
            y -= 0.45
        ys.append(y)
        y -= 1.0
    ys = [v - min(ys) for v in ys]
    for (name, t, kind), y in zip(rows, ys):
        if kind == "fc":
            ax.barh(y, 1.0, left=0, height=h, color=C_FC)
        elif kind == "h1":
            ax.barh(y, 1.0, left=0, height=h, color=C_FC)
            ax.plot(1.0, y, marker="D", color=C_KD, ms=5, mec="white", mew=0.5, zorder=4, clip_on=False)
        else:
            col = C_H if kind == "H" else C_C
            if t > 0:
                ax.barh(y, t, left=0, height=h, color=C_FC)
                ax.plot(t, y, marker="D", color=C_KD, ms=5, mec="white", mew=0.5, zorder=4)
            ax.barh(y, 1 - t, left=t, height=h, color=col)
        ax.text(-0.02, y, name, ha="right", va="center")
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, max(ys) + 0.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Fraction of the online PPO update budget")
    ax.spines["left"].set_visible(False)
    handles = [
        Patch(color=C_FC, label="Full-capacity classical PPO"),
        Patch(color=C_H, label="PPO, hybrid VQC head"),
        Patch(color=C_C, label="PPO, compact classical head"),
        Line2D([], [], marker="D", ls="", color=C_KD, mec="white", label="KD bridge"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2,
              frameon=False, handlelength=1.3, columnspacing=1.2)


def fig_design(out: Path) -> None:
    fig = plt.figure(figsize=(W2, 112 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.34)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _draw_pipeline(a)
    _draw_schedule(b)
    a.set_title("Staged training protocol")
    b.set_title("Handoff schedules and matched twins")
    save(fig, out, "fig1_design")


# -------------------------------------------------- F2 allocation frontier
def fig_allocation(d: Data, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 72 * MM))
    for ax, env in zip(axes, ENVS):
        h0 = d.mean(env, "H(0)")
        conds = [0.0, 0.25, 0.5, 0.75, 1.0]
        for t in conds:
            name = lab(t)
            x = d.cs.query("environment == @env and condition == @name").mean_qforward_total.iloc[0] / 1e6
            if t == 0.0:
                y, lo, hi = 0.0, 0.0, 0.0
            else:
                c = d.contrast(d.rq1[env], f"{name} - H(0)")
                y, lo, hi = c.paired_mean_difference, c.ci95_low, c.ci95_high
                assert abs(y - (d.mean(env, name) - h0)) < 1e-6
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=C_H, ecolor=C_H,
                        elinewidth=0.8, capsize=2, ms=4, zorder=3, mfc="white" if t == 0 else C_H)
            right = t == 1.0
            ax.annotate(name, (x, y), xytext=(4 if right else -4, 0), textcoords="offset points",
                        fontsize=7.5, ha="left" if right else "right", va="center")
        fc_delta = -d.contrast(d.fcr[d.fcr.environment == env], "H(0) - full_classical").paired_mean_difference
        c0_delta = d.mean(env, "C(0)") - h0
        ax.axhline(0, color="0.75", lw=0.6, zorder=1)
        ax.axhline(fc_delta, color=C_FC, ls="--", lw=0.8)
        ax.axhline(c0_delta, color=C_C, ls=":", lw=1.0)
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("Quantum forward evaluations (millions)")
        ax.set_xlim(0, ax.get_xlim()[1] * 1.04)
    axes[0].set_ylabel("Final return, paired $\\Delta$ vs H(0)")
    handles = [
        Line2D([], [], marker="o", color=C_H, ls="", label="H($t$) − H(0), mean and 95% CI"),
        Line2D([], [], color=C_FC, ls="--", label="Full classical − H(0)"),
        Line2D([], [], color=C_C, ls=":", label="C(0) − H(0)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.25, bottom=0.22)
    save(fig, out, "fig2_allocation_frontier")


# --------------------------------------------------------- F3 attribution
def fig_attribution(d: Data, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 72 * MM))
    rng = np.random.default_rng(0)  # jitter only
    for ax, env in zip(axes, ENVS):
        for i, t in enumerate(ATTR_T):
            hs, cs = d.seeds(env, lab(t)), d.seeds(env, lab(t, "C"))
            diff = (hs - cs).dropna()
            c = d.contrast(d.rq2[env], f"{lab(t)} - {lab(t, 'C')}")
            assert len(diff) == 20 and abs(diff.mean() - c.paired_mean_difference) < 1e-6
            jit = rng.uniform(-0.16, 0.16, len(diff))
            ax.scatter(i + jit, diff.values, s=7, color=C_H, alpha=0.35, lw=0, zorder=2)
            ax.errorbar(i, c.paired_mean_difference,
                        yerr=[[c.paired_mean_difference - c.ci95_low], [c.ci95_high - c.paired_mean_difference]],
                        fmt="o", color="black", capsize=2.5, ms=4, elinewidth=0.9, zorder=3)
            ax.annotate(f"$p_\\mathrm{{Holm}}$={c.holm_p:.2f}", (i, 1.01), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7.5)
        ax.axhline(0, color="0.5", lw=0.6, ls="--", zorder=1)
        ax.set_xticks(range(len(ATTR_T)))
        ax.set_xticklabels([f"$t$={t:g}" for t in ATTR_T])
        ax.set_xlim(-0.5, len(ATTR_T) - 0.5)
        ax.set_title(ENV_TITLE[env], pad=13)
    axes[0].set_ylabel("Final return, H($t$) − C($t$) (same seed)")
    handles = [
        Line2D([], [], marker="o", color=C_H, alpha=0.5, ls="", ms=3, label="Training seed"),
        Line2D([], [], marker="o", color="black", ls="-", label="Paired mean and 95% CI"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig.subplots_adjust(wspace=0.25, bottom=0.16)
    save(fig, out, "fig3_attribution")


# ------------------------------------------------------- F4 decomposition
def fig_decomposition(d: Data, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(W2, 110 * MM), sharex=True)
    cols = [
        ("Installation: post-KD H − C", [("post_kd_gap_h_minus_c", "black", "o", 0.0)]),
        ("Online gain after KD", [("hybrid_online_gain", C_H, "o", -0.09),
                                  ("classical_online_gain", C_C, "s", 0.09)]),
        ("Final H − C", [("final_gap_h_minus_c", "black", "o", 0.0)]),
    ]
    for r, env in enumerate(ENVS):
        t3 = d.rq3[env].set_index("handoff_t")
        for k, (title, series) in enumerate(cols):
            ax = axes[r, k]
            for col, colour, marker, off in series:
                for i, t in enumerate(INTERIOR):
                    m, lo, hi = (t3.loc[t, f"{col}_mean"], t3.loc[t, f"{col}_ci95_low"], t3.loc[t, f"{col}_ci95_high"])
                    ax.errorbar(i + off, m, yerr=[[m - lo], [hi - m]], fmt=marker, color=colour,
                                capsize=2, ms=3.5, elinewidth=0.8, mfc=colour if marker == "o" else "white")
            if k != 1:
                ax.axhline(0, color="0.5", lw=0.6, ls="--", zorder=1)
            ax.set_xticks(range(3))
            ax.set_xticklabels([f"$t$={t:g}" for t in INTERIOR])
            ax.set_xlim(-0.5, 2.5)
            if r == 0:
                ax.set_title(title)
            if k == 0:
                ax.set_ylabel(f"{ENV_TITLE[env]}\nreturn")
    handles = [
        Line2D([], [], marker="o", color=C_H, ls="", label="Hybrid H($t$)"),
        Line2D([], [], marker="s", color=C_C, mfc="white", ls="", label="Compact classical C($t$)"),
        Line2D([], [], marker="o", color="black", ls="", label="Paired H − C"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.32, hspace=0.25, bottom=0.12)
    save(fig, out, "fig4_decomposition")


# ---------------------------------------------- F5 classical continuation
def fig_continuation(d: Data, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 80 * MM))
    rng = np.random.default_rng(1)
    for ax, env in zip(axes, ENVS):
        tab = d.crc[d.crc.environment == env].set_index(["handoff_t", "right_head"])
        for i, t in enumerate(INTERIOR):
            for head, col, off, mk in (("H", C_H, -0.17, "o"), ("C", C_C, 0.17, "s")):
                diff = (d.fc[env] - d.seeds(env, lab(t, head))).dropna()
                m, lo, hi = tab.loc[(t, head), ["paired_mean_difference", "ci95_low", "ci95_high"]]
                assert len(diff) == 20 and abs(diff.mean() - m) < 1e-6
                ax.scatter(i + off + rng.uniform(-0.08, 0.08, 20), diff.values, s=6, color=col, alpha=0.3, lw=0)
                ax.errorbar(i + off, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, capsize=2.5, ms=4,
                            elinewidth=0.9, mfc=col if head == "H" else "white", mec=col, zorder=3)
        ax.axhline(0, color="0.5", lw=0.6, ls="--")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"$t$={t:g}" for t in INTERIOR])
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(ENV_TITLE[env])
    axes[0].set_ylabel("Full classical − switched head")
    handles = [
        Line2D([], [], marker="o", color=C_H, ls="", label="Full classical − H($t$), mean and 95% CI"),
        Line2D([], [], marker="s", color=C_C, mfc="white", ls="", label="Full classical − C($t$), mean and 95% CI"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    fig.text(0.5, -0.11, "Positive: continuing the full-capacity classical trajectory beats switching to the compact head",
             ha="center", fontsize=7.5, color="0.3")
    fig.subplots_adjust(wspace=0.25, bottom=0.2)
    save(fig, out, "fig5_classical_continuation")


# ---------------------------------------------------------- F6 resources
def fig_resources(d: Data, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * MM))
    order = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ax, env in zip(axes, ENVS):
        q = d.qfa[env].copy().rename(columns={"condition": "cond"})
        q = q[q["cond"].isin([lab(t) for t in order])]
        g = q.groupby("cond")[["online_grad", "online_nograd", "kd_grad", "kd_nograd", "combined_total"]].mean()
        h0_total = None
        for i, t in enumerate(order):
            name = lab(t)
            row = g.loc[name]
            total = row.iloc[-1]
            tab = d.res.query("environment == @env and condition == @name").mean_qforward_total.iloc[0]
            assert abs(total - tab) < 1e-6 and abs(row[:4].sum() - tab) < 1e-6
            h0_total = h0_total or total
            left = 0.0
            for col, colour, alpha in (("online_grad", C_H, GRAD_ALPHA), ("online_nograd", C_H, NOGRAD_ALPHA),
                                       ("kd_grad", C_KD, GRAD_ALPHA), ("kd_nograd", C_KD, NOGRAD_ALPHA)):
                v = row[col] / 1e6
                ax.barh(i, v, left=left, color=colour, alpha=alpha, height=0.62, lw=0)
                left += v
            change = 100 * (total / h0_total - 1)
            suffix = "" if t == 0 else f" ({change:+.0f}% vs H(0))".replace("-", "\u2212")
            ax.text(left, i, f"  {total / 1e6:.1f}M{suffix}", va="center", fontsize=7.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([lab(t) for t in order])
        ax.invert_yaxis()
        ax.set_xlabel("Quantum forward evaluations (millions)")
        ax.set_title(ENV_TITLE[env])
        ax.set_xlim(0, ax.get_xlim()[1] * 1.35)
    handles = [
        Patch(color=C_H, alpha=GRAD_ALPHA, label="Online PPO, with gradient"),
        Patch(color=C_H, alpha=NOGRAD_ALPHA, label="Online PPO, no gradient"),
        Patch(color=C_KD, alpha=GRAD_ALPHA, label="KD bridge, with gradient"),
        Patch(color=C_KD, alpha=NOGRAD_ALPHA, label="KD bridge, no gradient"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False)
    fig.subplots_adjust(wspace=0.45, bottom=0.25)
    save(fig, out, "fig6_resources")


# ================================================================ appendix
def _curves(d: Data, env: str) -> pd.DataFrame:
    base = d.root / "results" / env / "data"
    cols = ["condition", "seed", "phase", "global_step", "episode_return"]
    return pd.read_csv(base / "training_episode_returns.csv.gz", usecols=cols)


def _binned(t: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    t = t.assign(b=np.clip(np.digitize(t.global_step, edges) - 1, 0, len(edges) - 2))
    per_seed = t.groupby(["b", "seed"]).episode_return.mean().reset_index()
    g = per_seed.groupby("b").episode_return
    out = pd.DataFrame({"mean": g.mean(), "q25": g.quantile(0.25), "q75": g.quantile(0.75), "n": g.size()})
    out["x"] = 0.5 * (edges[out.index] + edges[out.index + 1])
    return out[out.n >= 10]


def fig_training_curves(d: Data, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(W2, 100 * MM), sharex="row", sharey="row")
    for r, env in enumerate(ENVS):
        t = _curves(d, env)
        horizon = t.global_step.max()
        edges = np.linspace(0, horizon, 41)
        fc = _binned(t[t.condition == "full_classical"], edges)
        for k, tt in enumerate(ATTR_T):
            ax = axes[r, k]
            ax.plot(fc.x / 1e6, fc["mean"], color=C_FC, lw=0.9)
            for head, col in (("H", C_H), ("C", C_C)):
                b = _binned(t[t.condition == lab(tt, head)], edges)
                ax.fill_between(b.x / 1e6, b.q25, b.q75, color=col, alpha=0.18, lw=0)
                ax.plot(b.x / 1e6, b["mean"], color=col, lw=0.9, ls="-" if head == "H" else "--")
            if tt > 0:
                ax.axvline(tt * horizon / 1e6, color=C_KD, lw=0.8, ls=":")
            if r == 0:
                ax.set_title(f"$t$={tt:g}")
            if k == 0:
                ax.set_ylabel(f"{ENV_TITLE[env]}\ntraining return")
            ax.set_xlabel("Env. steps (millions)" if r == 1 else "")
    handles = [
        Line2D([], [], color=C_FC, label="Full classical (teacher trajectory)"),
        Line2D([], [], color=C_H, label="H($t$), mean and IQR over seeds"),
        Line2D([], [], color=C_C, ls="--", label="C($t$), mean and IQR over seeds"),
        Line2D([], [], color=C_KD, ls=":", label="Handoff"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.12, bottom=0.14)
    save(fig, out, "figA1_training_curves")


def fig_h1_retention(d: Data, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 72 * MM))
    for ax, env in zip(axes, ENVS):
        h = d.hm[env]
        h1 = h[h.condition == "H(1)"].set_index("seed")
        x, y = h1.teacher_return_mean, h1.final_return_mean
        c = d.contrast(d.fcr[d.fcr.environment == env], "H(1) - teacher")
        assert abs((y - x).mean() - c.paired_mean_difference) < 1e-6
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=0.7, ls="--")
        ax.scatter(x, y, s=12, color=C_H, lw=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        note = (f"retention of teacher mean: {c.retention_pct_of_teacher_mean:.1f}%\n"
                f"same-seed Pearson $r$ = {c.pearson_r_same_seed:.3f}\n"
                f"paired $\\Delta$ = {c.paired_mean_difference:.2f} [{c.ci95_low:.2f}, {c.ci95_high:.2f}]")
        ax.text(0.04, 0.96, note.replace("-", "\u2212"),
                transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("Full-classical teacher, final return")
    axes[0].set_ylabel("H(1) after KD, final return")
    fig.subplots_adjust(wspace=0.3)
    save(fig, out, "figA2_h1_retention")


def fig_kd_diagnostics(d: Data, out: Path) -> None:
    k = pd.read_csv(d.root / "analysis/tables/kd_hc_diagnostics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 64 * MM))
    metrics = [("action_agreement_mean", "Validation action agreement"),
               ("policy_kl_mean", "Validation policy KL (teacher ‖ student)")]
    for ax, (m, title) in zip(axes, metrics):
        for j, env in enumerate(ENVS):
            s = k[k.environment == env].sort_values("handoff_t")
            xs = np.arange(3) + (j - 0.5) * 0.36
            ax.bar(xs - 0.08, s[f"hybrid_{m}"], width=0.16, color=C_H, alpha=1.0 if j == 0 else 0.55)
            ax.bar(xs + 0.08, s[f"classical_{m}"], width=0.16, color=C_C, alpha=1.0 if j == 0 else 0.55)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"$t$={t:g}" for t in INTERIOR])
        ax.set_title(title)
        if m.startswith("action"):
            ax.set_ylim(0.9, 1.0)
    handles = [Patch(color=C_H, label="H, LunarLander"), Patch(color=C_H, alpha=0.55, label="H, MinAtar"),
               Patch(color=C_C, label="C, LunarLander"), Patch(color=C_C, alpha=0.55, label="C, MinAtar")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    save(fig, out, "figA3_kd_diagnostics")


def fig_minatar_timing(d: Data, out: Path) -> None:
    tm = pd.read_csv(d.root / "analysis/tables/minatar_serial_timing.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 64 * MM))
    order = [lab(t, hd) for t in ATTR_T for hd in ("H", "C")] + ["H(1)"]
    order = [o for o in order if o in set(tm.condition)]
    tm = tm.set_index("condition").loc[order]
    cols = [C_H if o.startswith("H") else C_C for o in order]
    wall = (tm.kd_wall_clock_mean_seconds + tm.online_wall_clock_mean_seconds) / 3600
    axes[0].bar(range(len(order)), wall, color=cols)
    axes[0].set_ylabel("KD + online wall-clock (h)")
    axes[1].bar(range(len(order)), tm.combined_energy_mean_kwh, color=cols)
    axes[1].set_ylabel("KD + online energy (kWh)")
    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right")
    fig.suptitle("MinAtar Breakout, serial cohort (seeds 0–4), condition-stage measurements", fontsize=7, y=1.0)
    fig.subplots_adjust(wspace=0.3)
    save(fig, out, "figA4_minatar_timing")


def fig_iqm_forest(d: Data, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 80 * MM))
    for ax, env in zip(axes, ENVS):
        rows = pd.concat([d.rq1[env], d.rq2[env]])
        ys = np.arange(len(rows))[::-1]
        for y, (_, r) in zip(ys, rows.iterrows()):
            col = C_H if r.family == "allocation" else "black"
            ax.errorbar(r.iqm_difference, y, xerr=[[r.iqm_difference - r.iqm_ci95_low], [r.iqm_ci95_high - r.iqm_difference]],
                        fmt="o", color=col, capsize=2, ms=3.5, elinewidth=0.8)
            ax.text(1.02, y, f"PoI {r.probability_of_improvement:.2f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5)
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([c.replace(" - ", " − ") for c in rows.contrast])
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("IQM difference in final return (95% CI)")
    handles = [Line2D([], [], marker="o", color=C_H, ls="", label="RQ1 allocation"),
               Line2D([], [], marker="o", color="black", ls="", label="RQ2 attribution")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    fig.subplots_adjust(wspace=0.75, bottom=0.18)
    save(fig, out, "figA5_iqm_poi")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("figures/paper"))
    args = ap.parse_args()
    style()
    d = Data(args.root)
    out = args.root / args.out if not args.out.is_absolute() else args.out
    # Fig. 1 is produced by make_fig1.py (fig_design below is the superseded draft)
    fig_allocation(d, out)
    fig_attribution(d, out)
    fig_decomposition(d, out)
    fig_continuation(d, out)
    fig_resources(d, out)
    fig_training_curves(d, out)
    fig_h1_retention(d, out)
    fig_kd_diagnostics(d, out)
    fig_minatar_timing(d, out)
    fig_iqm_forest(d, out)


if __name__ == "__main__":
    main()
