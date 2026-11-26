"""Fig. 1 for paper (schematic; uses no data): (a) handoff-training protocol, (b) schedules with matched classical twins.
Outputs Fig1a, Fig1b and Fig1 (combined) as PDF + PNG. Drawn in millimetre coordinates."""
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Ellipse, Polygon, FancyArrowPatch, Arc

FC, H, C, KD, INK, MUTED = "#4D4D4D", "#D55E00", "#0072B2", "#CC79A7", "#1A1A1A", "#666666"
LIGHT = {H: "#FBEEE6", C: "#E6F0F8", KD: "#F7E6F0", FC: "#EFEFEF"}
MM = 1 / 25.4
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "pdf.fonttype": 42})
W = 174.0


# ------------------------------------------------------------------ primitives
def rbox(ax, x, y, w, h, ec, fc="white", lw=0.9, ls="-", r=1.6, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", ec=ec, fc=fc, lw=lw, ls=ls, zorder=z))


def arrow(ax, p, q, colour="#555555", lw=0.9, rad=0.0, ms=6, z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=colour,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def t(ax, x, y, s, size=6.5, colour=INK, weight="normal", ha="center", va="center", style="normal", z=5):
    ax.text(x, y, s, fontsize=size, color=colour, fontweight=weight, ha=ha, va=va, fontstyle=style, zorder=z,
            linespacing=1.25)


def nn(ax, cx, cy, layers, colour, dx, dy, r, lw=0.35, z=3, hollow=False):
    xs = [cx + (i - (len(layers) - 1) / 2) * dx for i in range(len(layers))]
    pts = [[(x, cy + (j - (n - 1) / 2) * dy) for j in range(n)] for x, n in zip(xs, layers)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#A8A8A8", lw=lw, zorder=z - 1)
    for col in pts:
        for p in col:
            ax.add_patch(Circle(p, r, fc="white" if hollow else colour, ec=colour if hollow else "none",
                                lw=0.7, zorder=z))
    return xs[0] - r, xs[-1] + r


def database(ax, cx, cy, w, h, colour=KD):
    e = h * 0.16
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + e / 2), w, h - e, fc=LIGHT[colour], ec="none", zorder=2))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2 + e / 2, cy + h / 2 - e / 2], color=colour, lw=0.9, zorder=3)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2 + e / 2, cy + h / 2 - e / 2], color=colour, lw=0.9, zorder=3)
    ax.add_patch(Arc((cx, cy - h / 2 + e / 2), w, e, theta1=180, theta2=360, color=colour, lw=0.9, zorder=3))
    for k in (0.33, 0.66):
        ax.add_patch(Arc((cx, cy - h / 2 + e / 2 + k * (h - e)), w, e, theta1=180, theta2=360, color=colour, lw=0.5, zorder=3))
    ax.add_patch(Ellipse((cx, cy + h / 2 - e / 2), w, e, fc="white", ec=colour, lw=0.9, zorder=4))


def meter(ax, x, y, s, colour):
    ax.add_patch(Rectangle((x, y), s, s * 0.8, fc="white", ec=colour, lw=0.6, zorder=4))
    ax.add_patch(Arc((x + s / 2, y + s * 0.18), s * 0.7, s * 0.7, theta1=15, theta2=165, color=colour, lw=0.6, zorder=5))
    ax.plot([x + s / 2, x + s * 0.78], [y + s * 0.18, y + s * 0.62], color=colour, lw=0.6, zorder=5)


def vqc(ax, x, y, w, h, colour=H):
    n = 3
    ys = [y + h * (k + 1) / (n + 1) for k in range(n)]          # ys[0] bottom wire, ys[2] top wire
    for yy in ys:
        ax.plot([x, x + w], [yy, yy], color="#555555", lw=0.6, zorder=3)
    gw = w * 0.15
    for gx, lab in ((x + w * 0.03, r"$R$"), (x + w * 0.55, r"$U_\theta$")):
        ax.add_patch(FancyBboxPatch((gx, ys[0] - h * 0.11), gw, ys[-1] - ys[0] + h * 0.22,
                                    boxstyle="round,pad=0,rounding_size=0.5", fc=colour, ec="none", zorder=4))
        t(ax, gx + gw / 2, (ys[0] + ys[-1]) / 2, lab, 6.0, "white")
    # CNOT ladder: control on the upper wire, target on the wire below
    for ctrl, targ, xx in ((2, 1, x + w * 0.29), (1, 0, x + w * 0.43)):
        ax.plot([xx, xx], [ys[targ] - 0.9, ys[ctrl]], color="#333333", lw=0.6, zorder=4)
        ax.add_patch(Circle((xx, ys[ctrl]), 0.5, fc="#333333", ec="none", zorder=5))
        ax.add_patch(Circle((xx, ys[targ]), 0.9, fc="white", ec="#333333", lw=0.6, zorder=5))
        ax.plot([xx - 0.9, xx + 0.9], [ys[targ], ys[targ]], color="#333333", lw=0.5, zorder=6)
        ax.plot([xx, xx], [ys[targ] - 0.9, ys[targ] + 0.9], color="#333333", lw=0.5, zorder=6)
    for yy in ys:
        meter(ax, x + w * 0.82, yy - 1.2, 2.9, colour)


def schedule_bar(ax, x, y, w, h, frac):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="#888888", lw=0.5, zorder=3))
    ax.add_patch(Rectangle((x, y), w * frac, h, fc=FC, ec="none", zorder=4))
    ax.add_patch(Polygon([(x + w * frac, y + h + 0.3), (x + w * frac - 1.0, y + h + 1.6), (x + w * frac + 1.0, y + h + 1.6)],
                         fc=INK, ec="none", zorder=5))


# ------------------------------------------------------------------ panel (a)
def panel_a(ax, y0, label=True):
    bw, gap, bh = 25.6, (W - 6 * 25.6) / 5, 38.0
    xs = [i * (bw + gap) for i in range(6)]
    top = y0 + bh
    if label:
        t(ax, 0, top + 4.0, "(a)  Handoff training", 8, weight="bold", ha="left")
    styles = [("#555555", "-"), (KD, "-"), (KD, "-"), ("#9A9A9A", "--"), ("#555555", "-"), ("#555555", "-")]
    for x, (ec, ls) in zip(xs, styles):
        rbox(ax, x, y0, bw, bh, ec, "white", ls=ls)
    titles = ["Classical training", "Teacher data", "Distillation", "", "Online PPO", "Evaluation"]
    subs = ["PPO up to $t$", "65,536 + 8,192 states", "policy + value", "", "remaining $1-t$", "100 episodes"]
    for x, a, b in zip(xs, titles, subs):
        if a:
            t(ax, x + bw / 2, y0 + 10.0, a, 7.0, weight="bold")
            t(ax, x + bw / 2, y0 + 5.0, b, 6.2, MUTED)
    icon_y = y0 + 25.0
    # 1 classical training up to t -> teacher
    x = xs[0]
    nn(ax, x + bw / 2, icon_y + 2.8, (3, 4, 4, 2), FC, 4.4, 3.4, 0.8)
    schedule_bar(ax, x + 3.0, y0 + 15.6, bw - 6.0, 1.6, 0.5)
    t(ax, x + 3.0 + (bw - 6.0) * 0.5, y0 + 14.4, "$t$", 5.8, INK, va="top")
    # 2 teacher -> dataset
    x = xs[1]
    ny = icon_y + 1.6
    l, r = nn(ax, x + 7.0, ny, (3, 4, 2), FC, 3.2, 3.3, 0.7)
    arrow(ax, (r + 0.8, ny), (x + 14.8, ny), KD, 0.8, ms=5)
    database(ax, x + 19.8, ny, 7.6, 10.5)
    t(ax, x + 7.0, y0 + 16.4, "teacher", 5.6, MUTED)
    # 3 distillation: teacher (filled) -> student (smaller, hollow)
    x = xs[2]
    ny = icon_y + 1.6
    l, r = nn(ax, x + 6.5, ny, (3, 4, 2), FC, 3.2, 3.3, 0.7)
    arrow(ax, (r + 0.8, ny), (x + 14.6, ny), KD, 0.8, ms=5)
    nn(ax, x + 19.6, ny, (3, 3, 2), FC, 2.8, 3.3, 0.7, hollow=True)
    t(ax, x + 6.5, y0 + 16.4, "teacher", 5.6, MUTED)
    t(ax, x + 19.6, y0 + 16.4, "student", 5.6, MUTED)
    # 4 compact heads: VQC (top) vs small NN (bottom)
    x = xs[3]
    half = (bh - 3.6) / 2
    rbox(ax, x + 1.2, y0 + 2.4 + half, bw - 2.4, half, H, LIGHT[H], lw=0.8, r=1.2, z=2)
    vqc(ax, x + 2.4, y0 + 2.4 + half + 3.6, bw - 4.8, half - 3.8)
    t(ax, x + bw / 2, y0 + 2.4 + half + 2.0, "$H(t)$: VQC head", 6.2, H, weight="bold")
    rbox(ax, x + 1.2, y0 + 1.2, bw - 2.4, half, C, LIGHT[C], lw=0.8, r=1.2, z=2)
    nn(ax, x + bw / 2, y0 + 1.2 + half / 2 + 1.8, (3, 4, 4, 3), C, 4.4, 2.6, 0.75)
    t(ax, x + bw / 2, y0 + 1.2 + 2.0, "$C(t)$: classical twin", 6.2, C, weight="bold")
    # 5 online PPO loop
    x = xs[4]
    ax_c, ay = x + bw / 2, icon_y
    rbox(ax, ax_c - 10.5, ay - 3.4, 8.4, 6.8, "#555555", "white", lw=0.7, r=1.0, z=3)
    t(ax, ax_c - 6.3, ay, "agent", 5.8)
    rbox(ax, ax_c + 2.1, ay - 3.4, 8.4, 6.8, "#555555", LIGHT[FC], lw=0.7, r=1.0, z=3)
    t(ax, ax_c + 6.3, ay, "env", 5.8)
    arrow(ax, (ax_c - 6.3, ay + 3.6), (ax_c + 6.3, ay + 3.6), "#555555", 0.7, rad=-0.6, ms=5)
    arrow(ax, (ax_c + 6.3, ay - 3.6), (ax_c - 6.3, ay - 3.6), "#555555", 0.7, rad=-0.6, ms=5)
    t(ax, ax_c, ay + 8.4, "action", 5.6, MUTED)
    t(ax, ax_c, ay - 8.4, "reward", 5.6, MUTED)
    # 6 evaluation bars
    x = xs[5]
    base = icon_y - 6.5
    for k, (hh, col) in enumerate(zip((6.5, 11.5, 9.0, 12.5), (FC, H, C, H))):
        ax.add_patch(Rectangle((x + 5.3 + k * 4.2, base), 3.2, hh, fc=col, ec="none", zorder=3))
    ax.plot([x + 4.2, x + bw - 4.2], [base, base], color="#555555", lw=0.6, zorder=4)
    # arrows between stages
    mid = y0 + bh / 2
    hy, cy = y0 + 2.4 + half + half / 2, y0 + 1.2 + half / 2
    for i in (0, 1):
        arrow(ax, (xs[i] + bw + 0.4, mid), (xs[i + 1] - 0.4, mid))
    arrow(ax, (xs[2] + bw, mid + 3), (xs[3] + 1.0, hy), KD)
    arrow(ax, (xs[2] + bw, mid - 3), (xs[3] + 1.0, cy), KD)
    arrow(ax, (xs[3] + bw - 1.0, hy), (xs[4] - 0.4, mid + 3))
    arrow(ax, (xs[3] + bw - 1.0, cy), (xs[4] - 0.4, mid - 3))
    arrow(ax, (xs[4] + bw + 0.4, mid), (xs[5] - 0.4, mid))
    # quantum-accounting ribbon and endpoint note
    rx0, rx1 = xs[2], xs[4] + bw
    rbox(ax, rx0, y0 - 5.6, rx1 - rx0, 4.0, "none", LIGHT[H], r=1.8)
    t(ax, (rx0 + rx1) / 2, y0 - 3.6, "quantum evaluations counted for $H(t)$: distillation + online", 6.2, H, style="italic")
    t(ax, 0.4, y0 - 3.6, "$t=0$: no teacher;  $t=1$: no online phase", 6.0, MUTED, ha="left")
    return top + (6.5 if label else 1.5), y0 - 7.0


# ------------------------------------------------------------------ panel (b)
def panel_b(ax, ytop, label=True):
    if label:
        t(ax, 0, ytop - 2.5, "(b)  Handoff schedules and matched classical twins", 8, weight="bold", ha="left")
        ytop -= 6.0
    lab_w, col_w, gap = 16.0, 66.0, 10.0
    cols = [(0.0, "Hybrid schedules", H, "H"), (lab_w + col_w + gap, "Matched classical twins", C, "C")]
    rows = [("Full classical", None), ("1", 1.0), ("0.75", 0.75), ("0.5", 0.5), ("0.25", 0.25), ("0", 0.0)]
    rh, bh = 5.6, 3.6
    for cx, title, colour, head in cols:
        bx = cx + lab_w
        t(ax, bx + col_w / 2, ytop - 1.5, title, 7.2, colour, weight="bold")
        for k, (name, tt) in enumerate(rows):
            y = ytop - 6.5 - k * rh
            t(ax, bx - 1.5, y + bh / 2, "Full classical" if tt is None else f"${head}({name})$", 6.6, ha="right")
            if tt is None:
                ax.add_patch(Rectangle((bx, y), col_w, bh, fc=FC, ec="none"))
            elif head == "C" and tt == 1.0:
                ax.add_patch(Rectangle((bx, y), col_w, bh, fc="white", ec="#BBBBBB", lw=0.5, ls="--"))
                t(ax, bx + col_w / 2, y + bh / 2, "no twin needed (a smaller copy of the teacher)", 5.6, "#999999", style="italic")
            else:
                if tt > 0:
                    ax.add_patch(Rectangle((bx, y), col_w * tt, bh, fc=FC, ec="none"))
                if tt < 1:
                    ax.add_patch(Rectangle((bx + col_w * tt, y), col_w * (1 - tt), bh, fc=colour, ec="none"))
                if tt > 0:
                    d = 1.5
                    ax.add_patch(Polygon([(bx + col_w * tt, y + bh / 2 + d), (bx + col_w * tt + d, y + bh / 2),
                                          (bx + col_w * tt, y + bh / 2 - d), (bx + col_w * tt - d, y + bh / 2)],
                                         fc=KD, ec="white", lw=0.6, zorder=5, clip_on=False))
        ay = ytop - 6.5 - (len(rows) - 1) * rh - 1.6
        ax.plot([bx, bx + col_w], [ay, ay], color="#333333", lw=0.6)
        for f in (0, 0.25, 0.5, 0.75, 1):
            ax.plot([bx + col_w * f] * 2, [ay, ay - 0.9], color="#333333", lw=0.6)
            t(ax, bx + col_w * f, ay - 2.4, f"{f:g}", 6.0)
        t(ax, bx + col_w / 2, ay - 5.4, "handoff point $t$ (fraction of the PPO update budget)", 6.0, MUTED)
    ly = ay - 10.5
    items = [(FC, "Full-capacity classical PPO (teacher trajectory)", "s"), (H, "Online PPO, hybrid VQC head", "s"),
             (C, "Online PPO, matched classical head", "s"), (KD, "Distillation", "d")]
    for (colour, lab, shape), lx in zip(items, (2.0, 64.0, 104.0, 152.0)):
        if shape == "d":
            ax.add_patch(Polygon([(lx + 1.2, ly + 1.4), (lx + 2.4, ly), (lx + 1.2, ly - 1.4), (lx, ly)], fc=colour, ec="none"))
        else:
            ax.add_patch(Rectangle((lx, ly - 1.0), 2.6, 2.0, fc=colour, ec="none"))
        t(ax, lx + 3.6, ly, lab, 6.0, ha="left")
    return ly - 3.0


def render(name, parts):
    heights = {"a": 56.0, "b": 58.0}
    total = sum(heights[p] for p in parts) + (2.0 if len(parts) > 1 else 0.0)
    fig = plt.figure(figsize=(W * MM, total * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.5, W + 0.5)
    ax.set_ylim(0, total)
    ax.set_aspect("equal")
    ax.axis("off")
    y = total
    for p in parts:
        lab = len(parts) > 1
        if p == "a":
            panel_a(ax, y - heights["a"] + 8.0, label=lab)
        else:
            panel_b(ax, y - (1.0 if lab else 0.0), label=lab)
        y -= heights[p] + (2.0 if lab else 0.0)
    fig.canvas.draw()
    from matplotlib.transforms import Bbox
    rend = fig.canvas.get_renderer()
    arts = [a for a in ax.patches + ax.texts + ax.lines if a.get_visible()]
    bb = Bbox.union([a.get_window_extent(rend) for a in arts]).transformed(fig.dpi_scale_trans.inverted()).padded(0.03)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches=bb)
    plt.close(fig)


ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("figures"))
OUT = ap.parse_args().out
OUT.mkdir(parents=True, exist_ok=True)
render("Fig1a_protocol", ["a"])
render("Fig1b_schedules", ["b"])
render("Fig1", ["a", "b"])
print("ok")
