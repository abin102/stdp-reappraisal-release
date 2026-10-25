"""Figure 8 — conceptual summary of the evaluation protocol (Discussion §5.3).
Left: the conventional single-number pipeline that conflates four factors.
Right: this paper's linear-readout + Δ-over-random protocol that isolates the rule.
Purely schematic (no data). Drawn at its printed width (the full text width) in the
shared journal style of scripts/figures/paper_style.py, so the font sizes are the printed sizes;
box widths follow their text. Box, line and arrow styling is shared with Figure 1, and
the colours are neutral because blue, vermillion and green mean the three rules in the
data figures. The caption, not the artwork, carries the cross-reference (Remark 2).
Written as a vector PDF (included by the manuscript) and a 600 dpi PNG.
Run: python scripts/figures/make_protocol_figure.py
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scripts.figures import paper_style as ps
ps.apply()

OUT = os.environ.get("FIG_OUT_DIR", "figures"); os.makedirs(OUT, exist_ok=True)
INK = ps.INK
FILL = "#EEF2F5"; RAND_FILL = "#E2E2E2"; OUT_FILL = "#D4D4D4"
FS = ps.FS_LABEL
LW, HEAD = 0.8, 7                             # the same line width and arrowhead as Figure 1

W, H = ps.FULL_W, 3.10                        # inches; axes coordinates are inches too
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
rend = fig.canvas.get_renderer()


def tsize(txt, fs=FS):
    t = ax.text(0, 0, txt, fontsize=fs, ha="center", va="center")
    bb = t.get_window_extent(renderer=rend); t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def box(cx, cy, text, fc=FILL, ec=INK, w=None, h=None, ls="-"):
    """Rounded box centred on (cx, cy), sized to its text; returns (x0, x1, y0, y1)."""
    tw, th = tsize(text) if text else (0.0, 0.0)
    w = w or tw + 0.20; h = h or th + 0.15
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.05",
                 fc=fc, ec=ec, lw=LW, ls=ls, zorder=2))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color=INK, zorder=3)
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def arrow(x0, y0, x1, y1, style="-|>"):
    # above the boxes, so the heads are not hidden under the box they point at
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=HEAD,
                 lw=LW, color=INK, zorder=4, shrinkA=0, shrinkB=0))


def down(b_from, b_to):
    x = (b_from[0] + b_from[1]) / 2
    arrow(x, b_from[2], x, b_to[3])


# ---------------- left: the conventional evaluation (confounded) ----------------
xl = W * 0.25
L = [box(xl, 2.92, "Image"), box(xl, 2.34, "STDP layer"), box(xl, 1.76, "Post-hoc readout"),
     box(xl, 1.18, "Single accuracy number", fc=OUT_FILL)]
for a, b in zip(L, L[1:]):
    down(a, b)
box(xl, 0.35, "One number conflates\narchitecture · scaffold · readout · plasticity",
    fc="white", ls="--", w=W / 2 - 0.30, h=0.66)

# ---------------- right: this paper (representation quality) ----------------
xc = W * 0.75; xa, xb = xc - 0.68, xc + 0.68
img = box(xc, 2.92, "Image")
enc = box(xc, 2.46, "Encoder")
down(img, enc)
fr = box(xa, 1.90, "Random-init\nfeatures", fc=RAND_FILL, ec=ps.GREY)
fs_ = box(xb, 1.90, "STDP\nfeatures")
arrow(xc - 0.12, enc[2], xa + 0.12, fr[3]); arrow(xc + 0.12, enc[2], xb - 0.12, fs_[3])
pa = box(xa, 1.37, "Linear readout"); pb = box(xb, 1.37, "Linear readout")
down(fr, pa); down(fs_, pb)
aa = box(xa, 0.93, r"$A_{\mathrm{random}}$", fc=RAND_FILL, ec=ps.GREY)
ab = box(xb, 0.93, r"$A_{\mathrm{STDP}}$")
down(pa, aa); down(pb, ab)
arrow(aa[1] + 0.03, 0.93, ab[0] - 0.03, 0.93, style="<|-|>")
ax.text(xc, 0.96, r"$\Delta$", ha="center", va="bottom", fontsize=FS, color=INK)
# the note: the formula on its own line, the wording of the caption below it
box(xc, 0.35, "", fc="white", w=W / 2 - 0.30, h=0.66)
ax.text(xc, 0.535, r"$\Delta = A_{\mathrm{STDP}} - A_{\mathrm{random}}$", ha="center",
        va="center", fontsize=FS, color=INK, zorder=3)
ax.text(xc, 0.255, "isolates the rule’s marginal contribution\nto representation quality",
        ha="center", va="center", fontsize=FS, color=INK, zorder=3, linespacing=1.2)

# the manuscript folder gets a copy only where it exists (not in the release copies)
for stem in (f"{OUT}/fig8_protocol",) + (("paper_nn/figures/fig8_protocol",) if os.path.isdir("paper_nn") else ()):
    os.makedirs(os.path.dirname(stem), exist_ok=True); ps.save(fig, stem)
plt.close(fig)
print(f"wrote {OUT}/fig8_protocol.pdf/.png"
      + (" and paper_nn/figures/fig8_protocol.pdf/.png" if os.path.isdir("paper_nn") else ""))
