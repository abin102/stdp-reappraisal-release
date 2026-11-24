"""Figure 1: the evaluation protocol (Section 2.1). Purely schematic (no data).
Left: the conventional evaluation, one accuracy under neuron labelling.
Right: the protocol: the encoded input, the untrained network (plasticity off) and the trained network,
read by a linear probe (solid arrows) and, when neuron labelling is reported, both networks also by
neuron labelling (dashed grey arrows, dashed boxes);
the three differences of Section 2.1.2. Drawn at the printed width in the shared style of paper_style.py.
Writes $FIG_OUT_DIR/fig8_protocol.pdf and .png (default figures/).
Run: python scripts/figures/make_protocol_figure.py
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
ps.apply()

OUT = os.environ.get("FIG_OUT_DIR", "figures"); os.makedirs(OUT, exist_ok=True)
INK = ps.INK
FILL = "#EEF2F5"; RAND_FILL = "#E2E2E2"; OUT_FILL = "#D4D4D4"; LAB_FILL = "#F7F7F7"
FS = ps.FS_LABEL
LW, HEAD = 0.8, 7

W, H = ps.FULL_W, 3.45
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
rend = fig.canvas.get_renderer()


def tsize(txt, fs=FS):
    t = ax.text(0, 0, txt, fontsize=fs, ha="center", va="center")
    bb = t.get_window_extent(renderer=rend); t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def box(cx, cy, text, fc=FILL, ec=INK, w=None, h=None, ls="-"):
    tw, th = tsize(text) if text else (0.0, 0.0)
    w = w or tw + 0.18; h = h or th + 0.14
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=LW, ls=ls, zorder=2))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color=INK, zorder=3)
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def arrow(x0, y0, x1, y1, style="-|>", ls="-", color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=HEAD,
                                 lw=LW, color=color, zorder=4, shrinkA=0, shrinkB=0, linestyle=ls))


def mid(b): return (b[0] + b[1]) / 2


Y_IMG, Y_ENC, Y_NET, Y_ACC, Y_NOTE = 3.17, 2.70, 2.05, 1.30, 0.42

# ---------------- left: the conventional evaluation ----------------
xl = 0.95
L = [box(xl, Y_IMG, "Image"), box(xl, Y_ENC, "STDP network"), box(xl, Y_NET, "Neuron labelling"),
     box(xl, Y_ACC, "One accuracy", fc=OUT_FILL)]
for a, b in zip(L, L[1:]):
    arrow(mid(a), a[2], mid(b), b[3])
box(xl, Y_NOTE, "mixes the input,\nthe architecture,\nthe scaffold, the readout\nand the rule",
    fc="white", ls="--", w=1.70, h=0.70)
ax.plot([2.02, 2.02], [0.08, H - 0.08], color=ps.GREY, lw=0.6, ls=(0, (1, 2)))

# ---------------- right: the protocol ----------------
xi, xu, xt = 2.55, 3.62, 4.75
img = box(xu, Y_IMG, "Image")
enc = box(xu, Y_ENC, "Encoder")
arrow(mid(img), img[2], mid(enc), enc[3])
ni = box(xi, Y_NET, "Input\nspikes")
nu = box(xu, Y_NET, "Untrained network\n(plasticity off)", fc=RAND_FILL, ec=ps.GREY)
nt = box(xt, Y_NET, "Trained\nnetwork")
for n in (ni, nu, nt):
    arrow(mid(enc) + (mid(n) - mid(enc)) * 0.12, enc[2], mid(n), n[3])

dx = 0.30
a_in = box(xi, Y_ACC, r"$A_{\mathrm{input}}$")
a_rl = box(xu - dx, Y_ACC, r"$A^{\mathrm{lab}}_{\mathrm{random}}$", fc=LAB_FILL, ec=ps.GREY, ls="--")
a_r = box(xu + dx, Y_ACC, r"$A_{\mathrm{random}}$", fc=RAND_FILL, ec=ps.GREY)
a_s = box(xt - dx, Y_ACC, r"$A_{\mathrm{STDP}}$")
a_sl = box(xt + dx, Y_ACC, r"$A^{\mathrm{lab}}_{\mathrm{STDP}}$", fc=LAB_FILL, ec=ps.GREY, ls="--")
arrow(mid(ni), ni[2], mid(a_in), a_in[3])
arrow(mid(nu) - 0.12, nu[2], mid(a_rl), a_rl[3], ls=(0, (3, 2)), color=ps.GREY)
arrow(mid(nu) + 0.12, nu[2], mid(a_r), a_r[3])
arrow(mid(nt) - 0.12, nt[2], mid(a_s), a_s[3])
arrow(mid(nt) + 0.12, nt[2], mid(a_sl), a_sl[3], ls=(0, (3, 2)), color=ps.GREY)

# readout legend
yl = 0.95
ax.add_line(Line2D([2.44, 2.72], [yl, yl], color=INK, lw=LW))
ax.text(2.77, yl, "linear probe", ha="left", va="center", fontsize=FS, color=INK)
ax.add_line(Line2D([3.55, 3.83], [yl, yl], color=ps.GREY, lw=LW, ls=(0, (3, 2))))
ax.text(3.88, yl, "neuron labelling (if reported)", ha="left", va="center", fontsize=FS, color=INK)

# the three differences
box(3.78, Y_NOTE, "", fc="white", w=2.95, h=0.70)
x0 = 2.44
for k, (f, what) in enumerate([(r"$\Delta = A_{\mathrm{STDP}} - A_{\mathrm{random}}$", "the rule"),
                               (r"$A_{\mathrm{STDP}} - A^{\mathrm{lab}}_{\mathrm{STDP}}$", "the readout"),
                               (r"$A_{\mathrm{STDP}} - A_{\mathrm{input}}$", "the input")]):
    y = Y_NOTE + 0.22 - k * 0.22
    ax.text(x0, y, f, ha="left", va="center", fontsize=FS, color=INK, zorder=3)
    ax.text(x0 + 1.62, y, what, ha="left", va="center", fontsize=FS, color=INK, zorder=3)

ps.save(fig, os.path.join(OUT, "fig8_protocol"))
plt.close(fig)
print("wrote", os.path.join(OUT, "fig8_protocol.pdf"))
