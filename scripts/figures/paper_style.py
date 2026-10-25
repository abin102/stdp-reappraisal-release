"""Shared figure style for the STDP paper, following Elsevier's artwork instructions.

Every figure is drawn at the size it is printed, so a font size here is the printed size.
Elsevier asks for lettering of about 7 pt (never below 6 pt for sub/superscripts), a
sans-serif such as Arial or Helvetica, and widths of 90 mm (single column), 140 mm
(1.5 column) or 190 mm (double column):
https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
The manuscript's text width is 390 pt = 137 mm, close to the 1.5-column size, and
paper_nn/main.tex includes figures at \\linewidth, so FULL_W figures print at 1:1.
SINGLE_W figures carry {width=90mm} in their markdown image link.

Rule colours are the Okabe-Ito palette, which stays distinguishable under the common
forms of colour blindness (the earlier red/green pair did not).
"""
import matplotlib as mpl

MM = 1 / 25.4
FULL_W = 390 / 72.27          # in, = \linewidth of paper_nn/main.tex (137 mm)
SINGLE_W = 90 * MM            # in, Elsevier single column

# matplotlib sets sub/superscripts at 0.7 of the font size, so labels are set at 6/0.7 pt:
# a subscript in any label then prints at Elsevier's 6 pt minimum, and labels with and
# without subscripts are the same size.
FS = 7              # tick labels, legends, annotations
FS_LABEL = 6 / 0.7  # axis labels, panel titles, schematic text (8.6 pt)
FS_SUB = FS_LABEL
FS_PANEL = 9        # panel letters

RULE_C = {"pair_exp": "#0072B2", "triplet": "#D55E00", "voltage": "#009E73"}
RULE_L = {"pair_exp": "pair-STDP", "triplet": "triplet-STDP", "voltage": "voltage-STDP"}
RULE_M = {"pair_exp": "o", "triplet": "s", "voltage": "^"}   # line plots: shape as a second cue
GREY = "#9A9A9A"      # post-hoc readout, random initialisation
INK = "#222222"
# schematic accents that are not rule colours: Okabe-Ito reddish purple and orange,
# darkened to at least 4.5:1 contrast on white so text in them stays legible
ACC1, ACC2 = "#A8508A", "#A86400"
ERR = dict(elinewidth=0.6, capthick=0.6, capsize=1.8, ecolor=INK)


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.cal": "Liberation Sans:italic",
        "font.size": FS,
        "axes.labelsize": FS_LABEL, "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.labelpad": 2.5, "axes.titlepad": 3,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.labelcolor": INK, "legend.handlelength": 1.4, "legend.handleheight": 0.7,
        "legend.handletextpad": 0.5, "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.3, "legend.columnspacing": 1.2, "legend.borderpad": 0.2,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.02, "figure.constrained_layout.wspace": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType fonts, not Type 3
        "savefig.dpi": 600, "savefig.facecolor": "white",
    })


def panel_label(ax, s):
    """Bold panel letter above the top-left corner of the axes (a left-aligned title, so
    constrained layout reserves room for it and it never collides with a centred title)."""
    ax.set_title(s, loc="left", fontsize=FS_PANEL, fontweight="bold", color=INK)


def guide_line(ax, x, axis="x"):
    """The one style for a non-data guide (a threshold, a group separator)."""
    (ax.axvline if axis == "x" else ax.axhline)(x, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1.5)


def zero_line(ax, axis="y"):
    """The one style for a zero reference line."""
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK, lw=0.6, zorder=1.5)


def save(fig, stem):
    """Vector PDF for the manuscript, 600 dpi PNG for slides and previews.
    No creation date in the PDF, so re-running a script reproduces the file byte for byte."""
    fig.savefig(f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(f"{stem}.png", dpi=600)
