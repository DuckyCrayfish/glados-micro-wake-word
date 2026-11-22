import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Training evaluation

    Graphs for judging a training run from its TensorBoard logs (`output/<run>/logs/`) and the ROC file written by the final streaming test. Edit `RUNS` and `ROC_FILES` in the first code cell to compare other runs.

    | Graph | Question it answers | Healthy | Warning sign |
    |---|---|---|---|
    | Loss, train vs validation | Is it still learning? Is it overfitting? | both fall; validation levels off | validation rises while train keeps falling |
    | Validation metrics | Is the model improving on what `best_weights` is chosen by? | climbs, then flattens | slow downward drift; frequent collapses |
    | Validation loss ÷ train loss | How big is the generalization gap, and is it growing? | flat or slowly rising | steady climb |
    | Wall-clock time | How fast is it training? Did it stall? | straight line | plateaus (pauses), slope drops |
    | Summary table | Final vs best checkpoint, per metric | final close to best | best was long ago and final is much worse |
    | False rejects vs false activations | The trade-off you get when picking a detection cutoff | curve hugging the lower left | curve sitting above the other model's |

    Faint lines are the raw value at every eval; bold lines are a rolling median (`SMOOTH` evals), which hides one-off spikes so the trend is readable.

    Not included: confusion matrix, precision-recall curve and score histograms. Those need the model run on test clips, and the logs only hold summary numbers.

    The logs and ROC files are read again on a timer, so the graphs follow a run that is still training, and a run's ROC curve appears as soon as its final test writes the file. The control below sets the interval, and its button reloads right away.
    """)
    return


@app.cell
def _():
    # Every cell that takes `refresh` runs again on each tick, and so does everything
    # downstream of those cells (all the graphs and tables).
    refresh = mo.ui.refresh(options=["10s", "30s", "1m", "5m"], default_interval="30s")
    refresh
    return (refresh,)


@app.cell
def _():
    # --- What to compare ---------------------------------------------------------
    # name -> TensorBoard log dir (contains train/ and validation/)
    RUNS = {
        "run 5": "output/05-fast-loader/logs",
        "run 4": "output/04-near-miss/logs",
        "run 3": "output/03-staged-lr-neg-ramp/logs",
        "run 2": "output/02-single-stage-repeat/logs",
    }
    # name -> ROC file from the quantized streaming test. Use the same names as RUNS
    # so each run keeps its color.
    ROC_FILES = {
        "run 5": "output/05-fast-loader/tflite_stream_state_internal_quant/tflite_streaming_roc.txt",
        "run 4": "output/04-near-miss/tflite_stream_state_internal_quant/tflite_streaming_roc.txt",
        "run 3": "output/03-staged-lr-neg-ramp/tflite_stream_state_internal_quant/tflite_streaming_roc.txt",
        "run 2": "output/02-single-stage-repeat/tflite_stream_state_internal_quant/tflite_streaming_roc.txt",
    }
    SMOOTH = 9  # evals in the (centered) rolling-median window

    # --- Look --------------------------------------------------------------------
    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    # Categorical slots 1-4 (blue, orange, aqua, yellow), validated as a set for line
    # charts and kept in this fixed order. Orange and yellow are the closest pair, so
    # runs are also named in legends and at line ends. Beyond four runs, compare in
    # separate notebooks instead.
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    if len(RUNS) > len(SERIES):
        raise ValueError("At most 4 runs; more would need hues that are not colorblind-safe")
    COLORS = dict(zip(RUNS, SERIES))

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.titlesize": 12, "axes.titleweight": "semibold",
        "lines.linewidth": 1.6, "legend.frameon": False,
    })
    return COLORS, INK_2, ROC_FILES, RUNS, SMOOTH, SURFACE


@app.cell(hide_code=True)
def _(RUNS, refresh):
    refresh  # run again on every tick

    def load_run(logdir):
        # Every scalar in train/ and validation/ as one long table.
        rows = []
        for split in ("train", "validation"):
            path = Path(logdir) / split
            if not path.exists():
                continue
            ea = EventAccumulator(str(path), size_guidance={"tensors": 0})
            ea.Reload()
            for tag in ea.Tags()["tensors"]:
                for e in ea.Tensors(tag):
                    value = float(tensor_util.make_ndarray(e.tensor_proto))
                    rows.append((split, tag, e.step, value, e.wall_time))
        df = pd.DataFrame(rows, columns=["split", "tag", "step", "value", "wall_time"])
        # A log dir can hold several event files (e.g. a resumed run). Keep the latest per step.
        return (df.sort_values("wall_time")
                  .drop_duplicates(["split", "tag", "step"], keep="last")
                  .sort_values(["split", "tag", "step"]))


    data = {name: load_run(path) for name, path in RUNS.items()}
    for _name, _df in data.items():
        if _df.empty:
            print(f"{_name}: no logs found in {RUNS[_name]!r}, skipping")
        else:
            _steps = _df[(_df["split"] == "validation") & (_df["tag"] == "loss")]["step"]
            print(f"{_name}: {len(_steps)} evals, steps {_steps.min():,} to {_steps.max():,}")
    data = {name: df for name, df in data.items() if not df.empty}
    return (data,)


@app.cell(hide_code=True)
def _(INK_2, SMOOTH, SURFACE, data):
    def get(name, split, tag):
        df = data[name]
        return df[(df["split"] == split) & (df["tag"] == tag)].set_index("step")["value"]


    def smooth(s):
        return s.rolling(SMOOTH, center=True, min_periods=1).median()


    def end_labels(ax, items, min_gap=13):
        # Names at the right end of each line, nudged apart so they never overlap.
        # items: (x, y, text, color). Call after axis limits and scales are final.
        ax.figure.canvas.draw()
        # min_gap and the offsets are pixels at the figure's dpi, passed on as points so
        # they scale with the text. marimo renders figures at twice the figure's dpi,
        # where pixel offsets would crowd labels that are sized in points.
        to_pt = 72 / ax.figure.dpi
        disp = [ax.transData.transform((x, y))[1] for x, y, _, _ in items]
        placed, last = {}, -1e9
        for i in np.argsort(disp):
            placed[i] = max(disp[i], last + min_gap)
            last = placed[i]
        for i, (x, y, text, color) in enumerate(items):
            ax.plot([x], [y], "o", color=color, ms=7, mec=SURFACE, mew=1.5, clip_on=False, zorder=5)
            ax.annotate(text, (x, y), xytext=(10 * to_pt, (placed[i] - disp[i]) * to_pt),
                        textcoords="offset points", va="center", ha="left", fontsize=9,
                        color=INK_2, annotation_clip=False)


    def room_on_right(ax, xmax, frac=0.2):
        lo, _ = ax.get_xlim()
        ax.set_xlim(lo, xmax + frac * (xmax - lo))

    return end_labels, get, room_on_right, smooth


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Loss: train vs validation

    The basic learning curve. Both should fall. Overfitting shows as validation loss turning up while train loss keeps going down. Train loss is one batch of 128 with class weights applied, validation loss is the full set, so compare the *shapes*, not the absolute levels.
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, INK_2, data, end_labels, get, room_on_right, smooth):
    def plot_loss():
        fig, ax = plt.subplots(figsize=(10, 4.8))
        items, xmax = [], 0
        for name in data:
            c = COLORS[name]
            for split, ls, short in (("train", (0, (4, 3)), "train"), ("validation", "-", "val")):
                s = get(name, split, "loss")
                if s.empty:
                    continue
                sm = smooth(s)
                ax.plot(s.index, s.values, color=c, alpha=0.18, lw=1)
                ax.plot(sm.index, sm.values, color=c, ls=ls, lw=2)
                items.append((sm.index[-1], sm.values[-1], f"{name} · {short}", c))
                xmax = max(xmax, sm.index[-1])
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss (log scale)")
        ax.set_title("Loss: train vs validation")
        handles = [Line2D([], [], color=COLORS[n], lw=2, label=n) for n in data]
        handles += [Line2D([], [], color=INK_2, lw=2, ls=(0, (4, 3)), label="train (one batch)"),
                    Line2D([], [], color=INK_2, lw=2, label="validation (full set)")]
        ax.legend(handles=handles, loc="upper right", fontsize=9)
        room_on_right(ax, xmax, 0.2)
        end_labels(ax, items)
        return fig


    plot_loss()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validation metrics

    These are what `best_weights` is chosen by (`average_viable_recall`) plus the metrics next to it. Rings mark each run's best eval. A ▼ on the bottom edge means the raw value fell below the axis there, which is how `recall_at_no_faph` collapses to 0 when one stretch of background audio scores high.
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, SURFACE, data, get, smooth):
    METRICS = [
        ("average_viable_recall", "Average viable recall (0–2 false activations/h)", 0.95),
        ("recall_at_no_faph", "Recall at 0 false activations/h", 0.95),
        ("recall", "Recall", 0.97),
        ("auc", "AUC", 0.995),
    ]


    def plot_metrics():
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle("Validation metrics", x=0.01, ha="left", fontsize=13, fontweight="semibold")
        for ax, (tag, title, lo) in zip(axes.flat, METRICS):
            hi = 1.0 + 0.04 * (1 - lo)
            for name in data:
                c = COLORS[name]
                s = get(name, "validation", tag)
                if s.empty:
                    continue
                sm = smooth(s)
                ax.plot(s.index, s.values, color=c, alpha=0.2, lw=1)
                ax.plot(sm.index, sm.values, color=c, lw=2)
                below = s[s < lo]
                ax.plot(below.index, [lo + 0.015 * (hi - lo)] * len(below), "v", color=c, ms=5, alpha=0.7)
                if tag in ("average_viable_recall", "recall_at_no_faph") and s.max() >= lo:  # ring only if on the axis
                    ax.plot([s.idxmax()], [s.max()], "o", mfc=SURFACE, mec=c, mew=2, ms=9, zorder=6, clip_on=False)
            ax.set_ylim(lo, hi)
            ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= 1.0 + 1e-9])  # nothing above 1.0
            ax.set_ylim(lo, hi)
            ax.set_title(title, fontsize=10.5)
            ax.set_xlabel("training step")
        fig.legend(handles=[Line2D([], [], color=COLORS[n], lw=2, label=n) for n in data],
                   loc="upper right", ncol=len(data), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig


    plot_metrics()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Overfitting check: validation loss ÷ train loss

    A single number for the generalization gap over time. The level isn't meaningful (the two losses are measured differently), but the **trend** is. A ratio that flattens means the model generalizes about as well as it did earlier. A steady climb means validation is falling behind train.
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, data, end_labels, get, room_on_right, smooth):
    def plot_gap():
        fig, ax = plt.subplots(figsize=(10, 4.2))
        items, xmax, ymax = [], 0, 0
        for name in data:
            c = COLORS[name]
            tr, va = get(name, "train", "loss"), get(name, "validation", "loss")
            steps = tr.index.intersection(va.index)
            ratio = (va.loc[steps] / tr.loc[steps])
            sm = smooth(ratio)
            ax.plot(ratio.index, ratio.values, color=c, alpha=0.18, lw=1)
            ax.plot(sm.index, sm.values, color=c, lw=2)
            items.append((sm.index[-1], sm.values[-1], name, c))
            xmax, ymax = max(xmax, sm.index[-1]), max(ymax, sm.max())
        ax.set_ylim(0, 1.5 * ymax)  # raw spikes run off the top; the trend is what matters
        ax.set_xlabel("training step")
        ax.set_ylabel("validation loss ÷ train loss")
        ax.set_title("Generalization gap")
        ax.legend(handles=[Line2D([], [], color=COLORS[n], lw=2, label=n) for n in data],
                  loc="upper left", fontsize=9)
        room_on_right(ax, xmax, 0.2)
        end_labels(ax, items)
        return fig


    plot_gap()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Wall-clock time

    Hours since each run's first eval. A straight line is steady speed, a flat stretch is a pause or stall, and a slope change is a slowdown. The label gives the average steps per second.
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, data, end_labels, room_on_right):
    def plot_time():
        fig, ax = plt.subplots(figsize=(10, 4.2))
        items, xmax = [], 0
        for name in data:
            c = COLORS[name]
            df = data[name]
            v = df[(df["split"] == "validation") & (df["tag"] == "loss")].sort_values("step")
            hours = (v["wall_time"] - v["wall_time"].iloc[0]) / 3600
            rate = (v["step"].iloc[-1] - v["step"].iloc[0]) / max(hours.iloc[-1] * 3600, 1)
            ax.plot(v["step"], hours, color=c, lw=2)
            items.append((v["step"].iloc[-1], hours.iloc[-1], f"{name} · {rate:.1f} steps/s", c))
            xmax = max(xmax, v["step"].iloc[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("training step")
        ax.set_ylabel("hours since first eval")
        ax.set_title("Training speed")
        ax.legend(handles=[Line2D([], [], color=COLORS[n], lw=2, label=n) for n in data],
                  loc="upper left", fontsize=9)
        room_on_right(ax, xmax, 0.3)
        end_labels(ax, items)
        return fig


    plot_time()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary table

    Last eval vs best eval per validation metric. Best is the maximum, except for loss (minimum). The best value is the optimistic one: it's the best of every eval, picked on the same validation set. The final streaming test is the unbiased number.
    """)
    return


@app.cell(hide_code=True)
def _(data, get):
    _rows = []
    for _name in data:
        for _tag in ["loss", "recall", "precision", "auc", "recall_at_no_faph", "average_viable_recall"]:
            _s = get(_name, "validation", _tag)
            if _s.empty:
                continue
            _best_step = _s.idxmin() if _tag == "loss" else _s.idxmax()
            _rows.append({"run": _name, "metric": _tag, "last step": _s.index[-1], "last": _s.iloc[-1],
                         "best": _s.loc[_best_step], "best step": _best_step})
    summary = pd.DataFrame(_rows).set_index(["run", "metric"])
    mo.ui.table(summary.round(4), pagination=False, selection=None, show_column_summaries=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## False rejects vs false activations (ROC)

    From the quantized streaming test, the model you would actually deploy. Each point is one detection cutoff (labeled). Moving right lets through more false activations per hour, moving down means fewer missed wake words. **Lower left is better.** The ROC file only exists after a run has finished and exported, so a run that is still training is skipped. The degenerate top cutoff (nothing triggers, FRR 100%) is left out.
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, ROC_FILES, refresh):
    refresh  # run again on every tick

    ROC_LINE = re.compile(r"Cutoff ([\d.]+): frr=([\d.]+); faph=([\d.]+)")


    def read_roc(path):
        rows = [tuple(map(float, m.groups())) for m in map(ROC_LINE.match, Path(path).read_text().splitlines()) if m]
        df = pd.DataFrame(rows, columns=["cutoff", "frr", "faph"])
        return df[df["frr"] < 0.999]


    rocs = {}
    for _name, _path in ROC_FILES.items():
        if _name not in COLORS:
            continue
        if Path(_path).exists():
            rocs[_name] = read_roc(_path)
        else:
            print(f"{_name}: {_path} not found yet, skipping")
    return (rocs,)


@app.cell(hide_code=True)
def _(COLORS, INK_2, SURFACE, end_labels, rocs, room_on_right):
    def plot_roc():
        fig, ax = plt.subplots(figsize=(10, 4.8))
        items = []
        for i, (name, roc) in enumerate(rocs.items()):
            c = COLORS[name]
            ax.plot(roc["faph"], roc["frr"] * 100, "-o", color=c, lw=2, ms=7, mec=SURFACE, mew=1.5)
            for _, r in roc.iterrows():
                ax.annotate(f"{r['cutoff']:.2f}", (r["faph"], r["frr"] * 100), xytext=(0, 9 if i % 2 == 0 else -15),
                            textcoords="offset points", ha="center", fontsize=8, color=INK_2)
            items.append((roc["faph"].iloc[-1], roc["frr"].iloc[-1] * 100, name, c))
        ax.set_xlabel("false activations per hour")
        ax.set_ylabel("missed wake words (%)")
        ax.set_title("False rejects vs false activations, by cutoff")
        ax.set_ylim(0, 1.15 * max(r["frr"].max() for r in rocs.values()) * 100)
        ax.set_xlim(left=-0.05)
        ax.legend(handles=[Line2D([], [], color=COLORS[n], lw=2, marker="o", label=n) for n in rocs],
                  loc="upper right", fontsize=9)
        xmax = max(r["faph"].max() for r in rocs.values())
        room_on_right(ax, xmax, 0.18)
        end_labels(ax, items)
        return fig


    mo.stop(not rocs)
    plot_roc()
    return


@app.cell(hide_code=True)
def _(rocs):
    mo.stop(not rocs)
    _roc_table = pd.concat(rocs, names=["run"]).droplevel(1).reset_index().set_index(["run", "cutoff"])
    mo.ui.table(_roc_table, pagination=False, selection=None, show_column_summaries=False)
    return


if __name__ == "__main__":
    app.run()
