"""Figure 1: the evaluation protocol (Section 2.1). Purely schematic (no data).
Left: the conventional evaluation, one accuracy under neuron labelling.
Right: the protocol: the encoded input, the trained network (middle) and the untrained network
(plasticity off), read by a linear probe (solid arrows) and, when neuron labelling (NL) is reported,
both networks also by neuron labelling (dashed grey arrows, dashed boxes);
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


# Rows, with equal gaps between boxes (top margin 0.08 in).
Y_IMG, Y_ENC, Y_NET, Y_ACC, Y_LEG, Y_NOTE = 3.24, 2.74, 2.05, 1.39, 1.05, 0.48
NL_LS = (0, (3, 2))
NOTE_H = 0.80

# ---------------- left: the conventional evaluation ----------------
xl = 0.66
L = [box(xl, Y_IMG, "Image"), box(xl, Y_ENC, "STDP network"), box(xl, Y_NET, "Neuron labelling"),
     box(xl, Y_ACC, "One accuracy", fc=OUT_FILL)]
for a, b in zip(L, L[1:]):
    arrow(mid(a), a[2], mid(b), b[3])
box(xl, Y_NOTE, "mixes the input,\nthe architecture,\nthe scaffold,\nthe readout\nand the rule",
    fc="white", ls="--", w=1.14, h=NOTE_H)
XDIV = 1.34
ax.plot([XDIV, XDIV], [0.08, H - 0.08], color=ps.GREY, lw=0.6, ls=(0, (1, 2)))

# ---------------- right: the protocol ----------------
# Three equally spaced columns with the trained network in the middle, under the encoder (every
# difference involves it). Each network's accuracies sit straight below it: linear probe on the
# left (solid arrow), neuron labelling (NL) on the right (dashed, grey). All accuracy boxes share
# one size.
NW, AW, AH, CG = 1.04, 0.58, 0.32, 0.06         # network width; accuracy box width, height; gap
s = (AW + CG) / 2                               # offset of each accuracy box from its network
d = 2 * AW + CG + 0.16                          # column pitch: one box pair plus 0.16 in
xc = (XDIV + 0.06 + W - 0.04 + (d + NW / 2) - (d + s + AW / 2)) / 2   # equal outer margins
xi, xt, xu = xc - d, xc, xc + d                 # input spikes, trained, untrained
img = box(xc, Y_IMG, "Image")
enc = box(xc, Y_ENC, "Encoder")
arrow(xc, img[2], xc, enc[3])
ni = box(xi, Y_NET, "Input\nspikes", w=NW)
nt = box(xt, Y_NET, "Trained\nnetwork", w=NW)
nu = box(xu, Y_NET, "Untrained\nnetwork", w=NW, fc=RAND_FILL, ec=ps.GREY)
for n in (ni, nt, nu):
    arrow(xc + (mid(n) - xc) * 0.15, enc[2], mid(n), n[3])

NLKW = dict(w=AW, h=AH, fc=LAB_FILL, ec=ps.GREY, ls="--")
a_in = box(xi, Y_ACC, r"$A_{\mathrm{input}}$", w=AW, h=AH)
a_t = box(xt - s, Y_ACC, r"$A_{\mathrm{trained}}$", w=AW, h=AH)
a_tl = box(xt + s, Y_ACC, r"$A^{\mathrm{NL}}_{\mathrm{trained}}$", **NLKW)
a_u = box(xu - s, Y_ACC, r"$A_{\mathrm{untrained}}$", w=AW, h=AH, fc=RAND_FILL, ec=ps.GREY)
a_ul = box(xu + s, Y_ACC, r"$A^{\mathrm{NL}}_{\mathrm{untrained}}$", **NLKW)
arrow(xi, ni[2], xi, a_in[3])
for n, p, q in ((nt, a_t, a_tl), (nu, a_u, a_ul)):
    arrow(mid(p), n[2], mid(p), p[3])
    arrow(mid(q), n[2], mid(q), q[3], ls=NL_LS, color=ps.GREY)

# readout legend, centred on the trained network
lw1, lw2 = tsize("linear probe")[0], tsize("neuron labelling (NL)")[0]
seg, pad, sep = 0.28, 0.05, 0.40
x = xc - (2 * (seg + pad) + lw1 + lw2 + sep) / 2
ax.add_line(Line2D([x, x + seg], [Y_LEG, Y_LEG], color=INK, lw=LW))
ax.text(x + seg + pad, Y_LEG, "linear probe", ha="left", va="center", fontsize=FS, color=INK)
x += seg + pad + lw1 + sep
ax.add_line(Line2D([x, x + seg], [Y_LEG, Y_LEG], color=ps.GREY, lw=LW, ls=NL_LS))
ax.text(x + seg + pad, Y_LEG, "neuron labelling (NL)", ha="left", va="center", fontsize=FS, color=INK)

# the three differences, centred on the trained network
rows = [(r"$\Delta = A_{\mathrm{trained}} - A_{\mathrm{untrained}}$", "the rule"),
        (r"$A_{\mathrm{trained}} - A^{\mathrm{NL}}_{\mathrm{trained}}$", "the readout"),
        (r"$A_{\mathrm{trained}} - A_{\mathrm{input}}$", "the input")]
c1 = max(tsize(f)[0] for f, _ in rows); c2 = max(tsize(w_)[0] for _, w_ in rows); gap = 0.45
box(xc, Y_NOTE, "", fc="white", w=c1 + gap + c2 + 0.50, h=NOTE_H)
x0 = xc - (c1 + gap + c2) / 2
for k, (f, what) in enumerate(rows):
    y = Y_NOTE + 0.22 - k * 0.22
    ax.text(x0, y, f, ha="left", va="center", fontsize=FS, color=INK, zorder=3)
    ax.text(x0 + c1 + gap, y, what, ha="left", va="center", fontsize=FS, color=INK, zorder=3)

ps.save(fig, os.path.join(OUT, "fig8_protocol"))
plt.close(fig)
print("wrote", os.path.join(OUT, "fig8_protocol.pdf"))
