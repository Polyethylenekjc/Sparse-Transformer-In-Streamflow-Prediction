"""Fonts are specified at final printed size, never scaled from a poster canvas."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from pathlib import Path
import os

BLUE, ORANGE, GREEN, GRAY = "#245B78", "#C47A31", "#3B7D68", "#65717B"
MM = 1/25.4


def style():
    # WSL can use the existing Windows fonts without downloading or bundling them.
    font_dirs=[Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts",Path("/mnt/c/Windows/Fonts")]
    if os.environ.get("PAPER_FONT_PATH"):
        font_manager.fontManager.addfont(os.environ["PAPER_FONT_PATH"])
    for directory in font_dirs:
        for filename in ("times.ttf","timesbd.ttf","timesi.ttf","timesbi.ttf"):
            path=directory/filename
            if path.exists():
                font_manager.fontManager.addfont(str(path))
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in installed else "DejaVu Serif"
    plt.rcParams.update({"font.family": family, "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": .65, "lines.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.dpi": 600, "axes.spines.top": False,
        "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=[BLUE,ORANGE,GREEN,GRAY])})
    return family


def canvas(height_mm, rows=1, cols=1):
    return plt.subplots(rows,cols,figsize=(170*MM,height_mm*MM),squeeze=False,layout="constrained")


def save(fig, out, name, registry):
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside_ticks=set()
    for ax in fig.axes:
        for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            lo,hi=sorted(limits)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    outside_ticks.update((tick.label1,tick.label2))
    for text in fig.findobj(Text):
        if text in outside_ticks or not text.get_visible() or not text.get_text().strip():
            continue
        bounds=text.get_window_extent(renderer)
        page=fig.bbox
        # Ignore tick labels outside axis limits (not painted by the axes).
        if text.axes is not None and text.get_clip_on():
            continue
        if bounds.x0 < page.x0-1 or bounds.y0 < page.y0-1 or bounds.x1 > page.x1+1 or bounds.y1 > page.y1+1:
            raise ValueError(f"Text leaves figure canvas in {name}: {text.get_text()!r}")
    # No bbox='tight': it changes physical width and the final font scale.
    for extension in ("pdf","svg","png"):
        fig.savefig(out/f"{name}.{extension}",facecolor="white")
    registry.append({"figure":name,"width_mm":170,"height_mm":round(fig.get_figheight()/MM,2),
                     "font_pt":9,"minimum_tick_font_pt":8,"png_dpi":600})
    plt.close(fig)
