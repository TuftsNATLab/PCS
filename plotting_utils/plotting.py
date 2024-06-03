from pathlib import Path
import matplotlib.pyplot as plt
import data_loading as dl
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
from matplotlib.gridspec import GridSpec
import sys
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
matplotlib.rcParams["hatch.linewidth"] = 2.0



output_folder_name = "figures"



SYSTEM = "PCS"


def GET_COLORS():
    colors = sns.color_palette("tab10").as_hex()

    names = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "gold",
        "lightblue",
    ]
    output = {n: c for n, c in zip(names, colors)}
    output.update({"black": "#000000", "midblue": "#1798cf"})

    ###fizzah code####


    output.update(
        {"gray1": "#333333",
        "gray2": "#808080",
        "gray3": "#bfbfbf",
        "green1": "#B6DA9F",
        "blue1": "#3e6fb4",
        "red1": "#de5152",
        # "red1": "#FF7A7A",
        "orange1": "#eb7e3b",
        "purple1": "#9372b2"
        }
    )

    ##################

    return output


COLORS = GET_COLORS()

FIG1A_JOB_COLORS = {
    "J1": COLORS["blue1"],
    "J2": COLORS["red1"],
    "J3": COLORS["gray1"],
    "J4": COLORS["orange1"],
}

FIG1A_JOB_TEXT_COLORS = defaultdict(lambda: "black", {"J3": "white"})

# second thing to modify
PARETO_COLORS = {
    "AFS": COLORS["red1"],
    "Tiresias": COLORS["blue1"],
    "TIRESIAS": COLORS["blue1"],
    "SRSF": COLORS["blue1"],
    "Themis": COLORS["green1"],
    "THEMIS": COLORS["green1"],
    "FIFO": COLORS["orange1"],
    "Max-Min": COLORS["purple1"],
    SYSTEM: COLORS["blue"],
    f"{SYSTEM}-JCT": COLORS["gray1"],
    f"{SYSTEM}-jct": COLORS["gray1"],
    f"{SYSTEM}_jct": COLORS["gray1"],
    f"{SYSTEM}-bal": COLORS["gray2"],
    f"{SYSTEM}_bal": COLORS["gray2"],
    f"{SYSTEM}-pred": COLORS["gray3"],
    f"{SYSTEM}_pred": COLORS["gray3"],
}

HATCHES = {
    "AFS": None,
    "Tiresias": None,
    "TIRESIAS": None,
    "SRSF": None,
    "FIFO": None,
    "Themis": None,
    "THEMIS": None,
    f"{SYSTEM}-JCT": None,
    f"{SYSTEM}-jct": None,
    f"{SYSTEM}_jct": None,
    f"{SYSTEM}-bal": None,  # "x" * 3,
    f"{SYSTEM}_bal": None,  # "x" * 3,
    f"{SYSTEM}-pred": None,  # "o" * 3,
    f"{SYSTEM}_pred": None,  # "o" * 3,
    # https://pythonmatplotlibtips.blogspot.com/2017/12/change-hatch-density-barplot-matplotlib-pyplot.html
}

MORE_COLORS = list(sns.color_palette("light:b").as_hex())  # len = 6

SYSTEM_BLUE = MORE_COLORS[3]
OTHER_BLUE = MORE_COLORS[1]

CDF_PLOT_INFO = {
    "PCS-bal": {
        "label": f"{SYSTEM}-bal",
        # "color": COLORS["midblue"],
        "color": COLORS["gray2"],
        "linestyle": "-",
        "marker": "o",
    },


    "PCS_bal": {
        "label": f"{SYSTEM}-bal",
        # "color": COLORS["midblue"],
        "color": COLORS["gray2"],
        "linestyle": "-",
        "marker": "o",
    },

    "PCS-pred": {
        "label": f"{SYSTEM}-pred",
        # "color": COLORS["lightblue"],
        "color": COLORS["gray3"],
        "linestyle": "-",
        "marker": "o",
    },


    "PCS_pred": {
        "label": f"{SYSTEM}-pred",
        # "color": COLORS["lightblue"],
        "color": COLORS["gray3"],
        "linestyle": "-",
        "marker": "o",
    },

    "PCS-JCT": {
        "label": f"{SYSTEM}-JCT",
        # "color": COLORS["blue"],
        "color": COLORS["gray1"],
        "linestyle": "-",
        "marker": "o",
    },

    "PCS_jct": {
        "label": f"{SYSTEM}-JCT",
        # "color": COLORS["blue"],
        "color": COLORS["gray1"],
        "linestyle": "-",
        "marker": "o",
    },

    "PCS-jct": {
        "label": f"{SYSTEM}-JCT",
        # "color": COLORS["blue"],
        "color": COLORS["gray1"],
        "linestyle": "-",
        "marker": "o",
    },
    "TIRESIAS": {
        "label": "Tiresias",
        # "color": COLORS["pink"],
        "color": COLORS["blue1"],
        "linestyle": "-",
        "marker": "^",
    },

    "SRSF": {
        "label": "Tiresias",
        # "color": COLORS["pink"],
        "color": COLORS["blue1"],
        "linestyle": "-",
        "marker": "^",
    },

    "AFS": {"label": "AFS", "color": COLORS["red1"], "linestyle": "-", "marker": "+"},

    "Themis": {
        "label": "Themis",
        "color": COLORS["green1"],
        "linestyle": "-",
        "marker": "^",
    },


    "THEMIS": {
        "label": "Themis",
        "color": COLORS["green1"],
        "linestyle": "-",
        "marker": "^",
    },

    "FIFO": {
        "label": "FIFO",
        "color": COLORS["orange1"],
        "linestyle": "-",
        "marker": None,
    },
    "FS": {
        "label": "Max-Min",
        # "color": COLORS["purple"],
        "color": COLORS["purple1"],
        "linestyle": None,
        "marker": None,
    },
}


def _save_image(folder, filename):
    full_folder = os.path.join(os.path.dirname(__file__), "..", output_folder_name)
    Path(full_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(full_folder, filename), bbox_inches="tight", format="pdf")
    plt.close()


def _make_axis_label(scheduler):
    return "          " + (scheduler.title() if scheduler != "yarn_fifo" else "FIFO")


def _get_closest(arr, target):
    min_dist = np.inf
    min_idx = None
    for i, e in enumerate(arr):
        curr_dist = abs(target - e)
        if curr_dist < min_dist:
            min_idx = i
            min_dist = curr_dist
    return min_idx


def _sample_cdf_data(X, Y):
    TOP_KEEP_COUNT = 5
    CDF_THRESHOLDS = np.linspace(0, 0.9, 10)
    sorted_idx = np.argsort(Y)
    top_idx = sorted_idx[-TOP_KEEP_COUNT:]
    sampledX, sampledY = X[top_idx].tolist(), Y[top_idx].tolist()
    for t in CDF_THRESHOLDS:
        closest = _get_closest(Y, t)
        sampledX.append(X[closest])
        sampledY.append(Y[closest])
    return sampledX, sampledY


def _pred_job(ax, j, i, jct, idx, data, scheduler):
    sj, sjct = j, str(jct)
    alpha_val = 0.3 + (0.7 * (jct > data["arrivals"][idx]["time"]))
    p_bbox = {
        "facecolor": FIG1A_JOB_COLORS[j],
        "edgecolor": "none",
        "pad": 3,
        "alpha": alpha_val,
    }
    ax.text(
        i,
        3 - (scheduler == "yarn_fifo"),
        sj,
        ha="center",
        va="center",
        bbox=p_bbox,
        fontsize=24,
        color=FIG1A_JOB_TEXT_COLORS[j],
        alpha=alpha_val,
    )
    ax.text(
        i + 1.5,
        3 - (scheduler == "yarn_fifo"),
        sjct,
        ha="right",
        va="center",
        fontsize=24,
        alpha=alpha_val,
    )
    if idx > 0:
        prev_pred = data[scheduler][idx - 1]["predictions"]
        if j in prev_pred:
            change = jct - prev_pred[j]
            if change > 0:
                ax.text(
                    i + 1,
                    2,
                    f"+{change}",
                    ha="center",
                    va="center",
                    color="red",
                    weight="bold",
                    fontsize=24,
                )


def create_custom_legend_lines(lines, linewidth=4):

    # Create legend with custom legend lines

    legend_lines = list()
    labels = list()
    for line in lines:
        legend_lines.append(Line2D([0], [0], color=line.get_color(), linewidth=linewidth))

        labels.append(line.get_label())

    return legend_lines, labels


def plot_fig4():
    data = dl.get_fig4_data()
    _sample_cdf_data(data[0]["X"], data[0]["Y"])
    fig = plt.figure(figsize=(8, 3.5))
    ax = plt.subplot()
    fig.add_subplot(ax)

    line_handles = list()

    for e in data:
        pi = CDF_PLOT_INFO[e["policy"]]
        line_handle, *_ = ax.plot(
            e["X"],
            e["Y"],
            c=pi["color"],
            label=pi["label"],
            linestyle=pi["linestyle"],
            linewidth=2,
        )

        line_handles.append(line_handle)

        x, y = _sample_cdf_data(e["X"], e["Y"])
        ax.scatter(x, y, c=pi["color"], s=50, marker=pi["marker"])
        
    custom_legend_lines, custom_legend_labels = create_custom_legend_lines(line_handles)

    ax.legend(custom_legend_lines, custom_legend_labels, fontsize=20, ncol=2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([0, 300])
    ax.tick_params("y", labelsize=20)
    ax.tick_params("x", labelsize=20)
    ax.set_ylabel("Fraction of Jobs", fontsize=20)
    ax.set_xlabel("Prediction Error (%)", fontsize=20)
    _save_image("graphs", "fig4.pdf")
    return ax


def plot_fig5():
    ax1 = plot_fig5a()
    ax2 = plot_fig5b()
    return ax1,ax2


def plot_fig5a(
    bar_width=0.24,
    bar_border_width=1,
    jct_font_size=20,
    statistic_font_size=20,
    bar_font_size=20,
    tick_font_size=20,
):
    data = dl.get_fig5a_data()
    labels = ["Avg", "p90"]
    x = np.arange(len(labels))
    fig = plt.figure()
    fig.set_figheight(4.8)
    fig.set_figwidth(6.4)
    ax = plt.subplot()
    fig.add_subplot(ax)
    rects1 = ax.bar(
        x - bar_width * 1.1,
        data["afs"],
        bar_width,
        label="AFS",
        color=CDF_PLOT_INFO["AFS"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )
    rects2 = ax.bar(
        x,
        data["tiresias"],
        bar_width,
        label="Tiresias",
        color=CDF_PLOT_INFO["TIRESIAS"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )
    rects3 = ax.bar(
        x + bar_width * 1.1,
        data["flex_jct"],
        bar_width,
        label=f"{SYSTEM}-JCT",
        color=CDF_PLOT_INFO["PCS-JCT"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )
    ax.set_ylabel("JCT (minutes)", fontsize=jct_font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=statistic_font_size)
    for label, rect in zip(
        ["AFS", "AFS", "Tiresias", "Tiresias", f"{SYSTEM}\nJCT", f"{SYSTEM}\nJCT"],
        rects1 + rects2 + rects3,
    ):
        height = rect.get_height()
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=bar_font_size,
        )
    ax.set_ylim(top=max(data["afs"] + data["tiresias"] + data["flex_jct"]) * 1.2)
    ax.tick_params(axis="x", which="major", length=0)
    ax.tick_params("y", labelsize=tick_font_size)
    ax.tick_params("x", labelsize=tick_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_image("graphs", "fig5a.pdf")
    return ax

def plot_fig5b(
    bar_width=0.24,
    bar_border_width=1,
    err_pred_font_size=20,
    statistic_font_size=20,
    bar_font_size=20,
    tick_font_size=20,
):
    data = dl.get_fig5b_data()
    labels = ["Avg", "p90"]
    x = np.arange(len(labels))
    fig = plt.figure()
    fig.set_figheight(4.8)
    fig.set_figwidth(6.4)
    ax = plt.subplot()
    fig.add_subplot(ax)
    rects1 = ax.bar(
        x - bar_width * 1.1,
        data["afs"],
        bar_width,
        label="AFS",
        color=CDF_PLOT_INFO["AFS"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )
    rects2 = ax.bar(
        x,
        data["tiresias"],
        bar_width,
        label="Tiresias",
        color=CDF_PLOT_INFO["TIRESIAS"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )
    rects3 = ax.bar(
        x + bar_width * 1.1,
        data["flex_jct"],
        bar_width,
        label=f"{SYSTEM}-JCT",
        color=CDF_PLOT_INFO["PCS-JCT"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )
    ax.set_ylabel("Prediction Error (%)", fontsize=err_pred_font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=statistic_font_size)
    for label, rect in zip(
        ["AFS", "AFS", "Tiresias", "Tiresias", f"{SYSTEM}\nJCT", f"{SYSTEM}\nJCT"],
        rects1 + rects2 + rects3,
    ):
        height = rect.get_height()
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=bar_font_size,
        )
    ax.tick_params(axis="x", which="major", length=0)
    ax.set_ylim(top=max(data["afs"] + data["tiresias"] + data["flex_jct"]) * 1.2)
    ax.tick_params("y", labelsize=tick_font_size)
    ax.tick_params("x", labelsize=tick_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_image("graphs", "fig5b.pdf")
    return ax

def plot_fig6():
    ax1 = plot_fig6a()
    ax2 = plot_fig6b()
    return ax1,ax2


def plot_fig6a():
    data = dl.get_fig6a_data()
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot()
    fig.add_subplot(ax)
    for e in data:
        pi = CDF_PLOT_INFO[e["policy"]]
        ax.plot(
            e["X"],
            e["Y"],
            c=pi["color"],
            label=pi["label"],
            linestyle=pi["linestyle"],
            linewidth=6,
        )
        x, y = _sample_cdf_data(e["X"], e["Y"])
        ax.scatter(x, y, c=pi["color"], s=250, marker=pi["marker"])
    ax.legend(fontsize=28, loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([0, 125])
    ax.set_ylim([0.3, 1.03])
    ax.tick_params("y", labelsize=28)
    ax.tick_params("x", labelsize=28)
    ax.set_ylabel("Fraction of Jobs", fontsize=28)
    ax.set_xlabel("Unfairness", fontsize=28)
    _save_image("graphs", "fig6a.pdf")
    return ax


def plot_fig6b(
    marker="o",
    axis_label_font_size=28,
    tick_font_size=28,
    marker_size=300,
    value_font_size=28,
):
    points = dl.get_fig6b_data()
    points = {CDF_PLOT_INFO[k]["label"]: v for k, v in points.items()}
    _, ax = plt.subplots(figsize=(8, 5))
    for label, point in points.items():
        if SYSTEM in label:
            ax.scatter(
                point["mean_err_pred"],
                point["normalized_mean_jct"],
                marker=marker,
                label=label,
                s=marker_size,
                c=PARETO_COLORS[label],
            )
        else:
            ax.scatter(
                point["mean_err_pred"],
                point["normalized_mean_jct"],
                marker=marker,
                label=label,
                s=marker_size,
                c=PARETO_COLORS[label],
            )
        if label == "Themis":
            ax.annotate(
                label,
                xy=(point["mean_err_pred"], point["normalized_mean_jct"]),
                xytext=(10, -5),
                textcoords="offset points",
                fontsize=value_font_size,
            )
        elif label == "Tiresias":
            ax.annotate(
                label,
                xy=(point["mean_err_pred"], point["normalized_mean_jct"]),
                xytext=(-120, -5),
                textcoords="offset points",
                fontsize=value_font_size,
            )
        elif label == f"{SYSTEM}-bal":
            ax.annotate(
                label,
                xy=(point["mean_err_pred"], point["normalized_mean_jct"]),
                xytext=(-20, 18),
                textcoords="offset points",
                fontsize=value_font_size,
            )
        elif label == f"{SYSTEM}-pred":
            ax.annotate(
                label,
                xy=(point["mean_err_pred"], point["normalized_mean_jct"]),
                xytext=(-11, 15),
                textcoords="offset points",
                fontsize=value_font_size,
            )
        else:
            ax.annotate(
                label,
                xy=(point["mean_err_pred"], point["normalized_mean_jct"]),
                xytext=(13, -1),
                textcoords="offset points",
                fontsize=value_font_size,
            )
    ax.set_ylabel("Normalized Avg JCT      ", fontsize=axis_label_font_size)
    ax.set_xlabel("Avg Prediction Error (%)", fontsize=axis_label_font_size)
    ax.tick_params("y", labelsize=tick_font_size)
    ax.tick_params("x", labelsize=tick_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_image("graphs", "fig6b.pdf")
    return ax


def plot_fig7():
    return plot_fig7a(), plot_fig7b()
    

def plot_fig7a(
    data_file="avg_jct_results.csv",
    ylabel="Norm. Avg JCT",
    out_file="fig7a.pdf",
    y_min=0,
    y_limit=9,
    legend=True,
    yticks=[0, 2, 4, 6],
    yticklabels=["0", "2", "4", "6"],
    logscale=False,
    xtick_labelsize=16,
    max_y=6.5,
    data=None
):
    return _plot_fig7a(
        data_file,
        ylabel,
        out_file,
        y_min,
        y_limit,
        legend,
        logscale=logscale,
        yticks=yticks,
        yticklabels=yticklabels,
        xtick_labelsize=xtick_labelsize,
        max_y=max_y,
        data=data,
    )


def _plot_fig7a(
    data_file,
    ylabel,
    out_file,
    y_min,
    y_limit,
    legend,
    yticks=None,
    yticklabels=None,
    logscale=False,
    xtick_labelsize=24,
    max_y=float('inf'),
    data=None
):
    data_table = data or dl.get_trace_data(data_file)
    x_labels = data_table["traces"]
    y = data_table["data"]
    bar_width = 0.02
    spacing = 0.15
    bar_indices = _get_bar_indices(len(x_labels), len(y), bar_width, spacing)
    x_end = np.max(bar_indices) + (bar_width)
    x_start = np.min(bar_indices) - (bar_width)
    _, ax = plt.subplots(figsize=(10, 3))
    ax.set_xticks(np.arange(0, len(x_labels) * spacing, spacing))
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for xticks, (policy, metrics) in zip(bar_indices, y.items()):

        if logscale:
            metrics = np.add(metrics, 1)
            ax.set_yscale("log")
        ax.bar(
            xticks,
            [min(m,max_y) for m in metrics],
            width=bar_width,
            label=policy,
            color=PARETO_COLORS[policy],
            hatch=HATCHES[policy],
            # https://stackoverflow.com/a/59389823
            alpha=0.99,
            zorder=3
        )
        for x, m in zip(xticks, metrics):
            if m > y_limit:
                if logscale:
                    m = np.round(np.log10(m), 1)


                # draw_broken_indicator(x,min(max_y,y_limit)-0.5,bar_width,0.2,ax)

                ax.text(
                    x,
                    min(max_y,y_limit)+0.8,
                    str(round(m)),
                    ha="center",
                    va="top",
                    fontsize=xtick_labelsize,
                    rotation=90 if m > 99 else 0,
                )
            elif m == 0:
                ax.axhline(
                    0,
                    ((x - (bar_width / 2)) / (x_end - x_start))
                    - (x_start / (x_end - x_start)),
                    ((x + (bar_width / 2)) / (x_end - x_start))
                    - (x_start / (x_end - x_start)),
                    linewidth=0.5,
                    color=PARETO_COLORS[policy],
                )

    if legend:
        ax.legend(
            fontsize=14,
            # loc="upper left",
            labelspacing=0.1,
            columnspacing=1,
            handlelength=0.5,
            borderaxespad=0.1,
            ncol=len(y),
            frameon=False,
            loc='upper left', bbox_to_anchor=(0, 1.0)
            # bbox_to_anchor=(1.3, 1.05),
        )
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)

    ax.tick_params("y", labelsize=20)
    ax.tick_params("x", labelsize=xtick_labelsize)
    ax.set_ylabel(ylabel, fontsize=20,y=0.4)
    ax.set_xlabel("Trace", fontsize=20)
    ax.set_ylim([y_min, y_limit])
    ax.set_xlim([x_start, x_end])
    _save_image("graphs", out_file)
    return ax



def plot_fig7b(
    data_file1="avg_error_results.csv",
    data_file2="p99_error_results.csv",
    ylabel="Pred Error %",
    out_file="fig7b.pdf",
    y_min=0.8,
    y_limit=1000,
    legend=False,
    yticks=[1, 10, 100, 1000],
    yticklabels=["0", "10", "$10^2$", "$10^3$"],
    logscale=True,
    xtick_labelsize=16,
    data1=None,
    data2=None,
):
    return _plot_fig7b(
        data_file1,
        data_file2,
        ylabel,
        out_file,
        y_min,
        y_limit,
        legend,
        yticks=yticks,
        yticklabels=yticklabels,
        logscale=logscale,
        xtick_labelsize=xtick_labelsize,
        data1=data1,
        data2=data2,
    )



def _plot_fig7b(
    data_file1,
    data_file2,
    ylabel,
    out_file,
    y_min,
    y_limit,
    legend,
    yticks=None,
    yticklabels=None,
    logscale=False,
    xtick_labelsize=24,
    data1=None,
    data2=None,
):
    data_table = data1 or dl.get_trace_data(data_file1)
    tail_data_table = data2 or dl.get_trace_data(data_file2)

    x_labels = data_table["traces"]
    y = data_table["data"]
    tail_y = tail_data_table["data"]

    bar_width = 0.02
    spacing = 0.15
    bar_indices = _get_bar_indices(len(x_labels), len(y), bar_width, spacing)
    x_end = np.max(bar_indices) + (bar_width)
    x_start = np.min(bar_indices) - (bar_width)
    _, ax = plt.subplots(figsize=(10, 3))
    ax.set_xticks(np.arange(0, len(x_labels) * spacing, spacing))
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for xticks, (policy, metrics) in zip(bar_indices, y.items()):
        if logscale:
            metrics = np.add(metrics, 1)
            ax.set_yscale("log")
        ax.bar(
            xticks,
            metrics,
            width=bar_width,
            label=policy,
            color=PARETO_COLORS[policy],
            hatch=HATCHES[policy],
            # https://stackoverflow.com/a/59389823
            alpha=0.99,
            zorder=3
        )


        tail_y[policy] = np.add(tail_y[policy], 1)
        for x, m, t in zip(xticks, metrics, tail_y[policy]):
            draw_error_line(x,m,t,bar_width,ax)


        for x, m in zip(xticks, metrics):

            if m > y_limit:
                if logscale:
                    m = np.round(np.log10(m), 1)
                ax.text(
                    x,
                    y_limit,
                    str(m),
                    ha="center",
                    va="top",
                    fontsize=22,
                    rotation=90 if m > 99 else 0,
                )
            elif m == 0:
                ax.axhline(
                    0,
                    ((x - (bar_width / 2)) / (x_end - x_start))
                    - (x_start / (x_end - x_start)),
                    ((x + (bar_width / 2)) / (x_end - x_start))
                    - (x_start / (x_end - x_start)),
                    linewidth=0.5,
                    color=PARETO_COLORS[policy],
                )

    if legend:
        ax.legend(
            fontsize=24,
            loc="upper left",
            labelspacing=0.1,
            handlelength=0.5,
            borderaxespad=0.1,
            ncol=1,
            frameon=False,
            # bbox_to_anchor=(1.3, 1.05),
        )
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, x=0.01)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    ax.tick_params("y", labelsize=20)
    ax.tick_params("x", labelsize=xtick_labelsize)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=-1, y=0.5)
    ax.set_xlabel("Trace", fontsize=20)
    ax.set_ylim([y_min, y_limit])
    ax.set_xlim([x_start, x_end])
    _save_image("graphs", out_file)
    return ax



def plot_fig8():
    return plot_fig8a(), plot_fig8b()



def plot_fig8a(
    bar_width=0.15,
    bar_border_width=1,
    jct_font_size=28,
    scheduler_font_size=28,
    tick_font_size=28,
    data=None,
):
    data = data or dl.get_fig8a_data()
    _, ax = plt.subplots(figsize=(8, 5))

    inter_bar_gap = 1.5

    ax.bar(
        0,
        data["afs"][0],
        bar_width,
        color=CDF_PLOT_INFO["AFS"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )
    ax.bar(
        bar_width * inter_bar_gap,
        data["tiresias"][0],
        bar_width,
        color=CDF_PLOT_INFO["TIRESIAS"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )
    ax.bar(
        bar_width * 2 * inter_bar_gap,
        data["themis"][0],
        bar_width,
        color=CDF_PLOT_INFO["THEMIS"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )

    ax.bar(
        bar_width * 3 * inter_bar_gap,
        data["flex_jct"][0],
        bar_width,
        color=CDF_PLOT_INFO["PCS-JCT"]["color"],
        edgecolor=None,
        linewidth=bar_border_width,
    )

    ax.set_xticks([0, bar_width * inter_bar_gap, bar_width * 2 * inter_bar_gap, bar_width * 3 *inter_bar_gap])
    ax.set_xticklabels(
        ["AFS", "Tiresias", "Themis", f"{SYSTEM}-JCT"], fontsize=scheduler_font_size
    )
    ax.tick_params("y", labelsize=tick_font_size)
    ax.tick_params("x", labelsize=tick_font_size)
    ax.set_ylabel("Avg Pred Error (%)", fontsize=jct_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(
        "   ",
        fontsize=jct_font_size,
    )
    ax.set_ylim([0, 20])
    _save_image("graphs", "fig8a.pdf")
    return ax

def plot_fig8b(
    marker="o",
    axis_label_font_size=28,
    marker_size=300,
    tick_font_size=28,
    point_font_size=28,
    data=None,
):
    points = data or dl.get_fig8b_data()
    points = {CDF_PLOT_INFO[k]["label"]: v for k, v in points.items()}
    _, ax = plt.subplots(figsize=(8, 5))
    for label, point in points.items():
        if SYSTEM in label:
            ax.scatter(
                point["p99_err_pred"],
                point["normalized_mean_jct"],
                marker=marker,
                label=label,
                s=marker_size,
                c=PARETO_COLORS[label],
            )
        elif label == "FIFO":
            ax.scatter(
                point["p99_err_pred"],
                point["normalized_mean_jct"],
                marker=marker,
                label=label,
                s=marker_size,
                c=PARETO_COLORS[label],
            )
        elif label == "Max-Min":
            continue
        else:
            ax.scatter(
                point["p99_err_pred"],
                point["normalized_mean_jct"],
                marker=marker,
                label=label,
                s=marker_size,
                c=PARETO_COLORS[label],
            )
        if label == "AFS":
            ax.annotate(
                label,
                xy=(point["p99_err_pred"], point["normalized_mean_jct"]),
                xytext=(-70, -10),
                textcoords="offset points",
                fontsize=point_font_size,
            )
        elif label == "Tiresias":
            ax.annotate(
                label,
                xy=(point["p99_err_pred"], point["normalized_mean_jct"]),
                xytext=(-50, 40),
                textcoords="offset points",
                fontsize=point_font_size,
            )
        elif label == "Max-Min":
            # ax.annotate(
            #     label,
            #     xy=(point["mean_err_pred"], point["normalized_mean_jct"]),
            #     xytext=(-65, -30),
            #     textcoords="offset points",
            #     fontsize=point_font_size,
            # )
            continue
        else:
            ax.annotate(
                label,
                xy=(point["p99_err_pred"], point["normalized_mean_jct"]),
                xytext=(10, -7),
                textcoords="offset points",
                fontsize=point_font_size,
            )
    ax.arrow(
        343, 1.32, 0, -0.12, head_width=8, width=2, head_length=0.06, fc="k", ec="k"
    )
    ax.tick_params("y", labelsize=tick_font_size)
    ax.tick_params("x", labelsize=tick_font_size)
    ax.set_ylabel("Normalized Avg JCT", fontsize=axis_label_font_size)
    ax.set_xlabel("p99 Prediction Error (%)", fontsize=axis_label_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_image("graphs", "fig8b.pdf")
    return ax

def plot_fig9():
    return plot_fig9a(), plot_fig9b(), plot_fig9c()


def plot_fig9a(data=None):
    data = data or dl.get_fig9a_data()
    x_labels = data["gpus"]
    sizes = ["512", "1024"]
    bar_values = [data[key]["values"] for key in sizes]
    std_values = [data[key]["stds"] for key in sizes]
    _, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar(
        np.arange(len(x_labels)) - 0.15,
        bar_values[0],
        width=0.3,
        yerr=std_values[0],
        capsize=5,
        label=sizes[0] + " Jobs",
        color=COLORS["green1"],
    )
    ax.bar(
        np.arange(len(x_labels)) + 0.15,
        bar_values[1],
        width=0.3,
        yerr=std_values[1],
        capsize=5,
        label=sizes[1] + " Jobs",
        color=COLORS["orange1"],
    )
    ax.legend(
        fontsize=24,
        loc="upper left",
        labelspacing=0.1,
        handlelength=0.5,
        borderaxespad=0.1,
        frameon=False,
    )
    ax.tick_params("y", labelsize=24)
    ax.tick_params("x", labelsize=24)
    ax.set_ylabel("Time (min)", fontsize=24)
    ax.set_xlabel("Number of GPUs", fontsize=24)
    ax.set_ylim([0, 8])
    _save_image("graphs", "fig9a.pdf")
    return ax

def plot_fig9b():
    data = dl.get_fig9b_data()
    x_labels = data["gpus"]
    sizes = ["512", "1024"]
    bar_values = [np.divide(data[key]["values"], 60) for key in sizes]
    _, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar(
        np.arange(len(x_labels)) - 0.15,
        bar_values[0],
        width=0.3,
        label=sizes[0] + " Jobs",
        color=COLORS["green1"],
    )
    ax.bar(
        np.arange(len(x_labels)) + 0.15,
        bar_values[1],
        width=0.3,
        capsize=5,
        label=sizes[1] + " Jobs",
        color=COLORS["orange1"],
    )
    ax.legend(
        fontsize=22,
        loc="upper left",
        labelspacing=0.1,
        handlelength=0.5,
        borderaxespad=0.1,
        frameon=False,
    )
    ax.tick_params("y", labelsize=24)
    ax.tick_params("x", labelsize=24)
    ax.set_ylabel("Time (min)", fontsize=24)
    ax.set_xlabel("Number of GPUs", fontsize=24)
    ax.set_ylim([0, 80])
    _save_image("graphs", "fig9b.pdf")
    return ax
def plot_fig9c():
    data = dl.get_fig9c_data()
    keys = ["heuristics", "without_heuristics"]
    x_labels = data["labels"]
    bar_values = [data[key] for key in keys]
    _, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar(
        np.arange(len(x_labels)) + 0.15,
        bar_values[0],
        width=0.3,
        capsize=5,
        label="Heuristics",
        color=COLORS["orange1"],
    )
    ax.bar(
        np.arange(len(x_labels)) - 0.15,
        bar_values[1],
        width=0.3,
        capsize=5,
        label="No Heuristics",
        color=COLORS["green1"],
    )
    ax.legend(
        fontsize=24,
        loc="upper left",
        labelspacing=0.1,
        handlelength=0.5,
        borderaxespad=0.1,
        frameon=False,
    )
    ax.tick_params("y", labelsize=24)
    ax.tick_params("x", labelsize=24)
    ax.set_ylabel("Pareto Optimal Points (%)      ", fontsize=24)
    ax.set_xlabel("Number of Evaluations", fontsize=24)
    ax.set_ylim([0, 100])
    _save_image("graphs", "fig9c.pdf")
    return ax

def plot_fig10():    
    return plot_fig10a(), plot_fig10b()

def plot_fig10a(data=None):
    data = data or dl.get_fig10_data()
    x_labels = list(data["PCS"].keys())
    schedulers = ["PCS", "FIFO"]
    bar_values = [
        [data[key][k]["avg pred error"] for k in x_labels] for key in schedulers
    ]
    _, ax = plt.subplots()
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar(
        np.arange(len(x_labels)) + 0.2,
        bar_values[0],
        width=0.4,
        label=SYSTEM,
        color=CDF_PLOT_INFO["PCS-JCT"]["color"],
    )
    ax.bar(
        np.arange(len(x_labels)) - 0.2,
        bar_values[1],
        width=0.4,
        label="FIFO",
        color=CDF_PLOT_INFO["FIFO"]["color"],
    )
    ax.legend(fontsize=22, frameon=False)
    ax.tick_params("y", labelsize=22)
    ax.tick_params("x", labelsize=22)
    ax.set_ylabel("Avg Prediction Error (%)", fontsize=22)
    ax.set_xlabel("Job Size Estimation Error", fontsize=22)
    _save_image("graphs", "fig10a.pdf")
    return ax


def plot_fig10b(data=None):
    data = data or dl.get_fig10_data()
    x_labels = list(data["PCS"].keys())
    schedulers = ["PCS", "AFS"]
    bar_values = [[data[key][k]["avg JCT"] for k in x_labels] for key in schedulers]
    _, ax = plt.subplots()
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar(
        np.arange(len(x_labels)) - 0.2,
        bar_values[0],
        width=0.4,
        color=CDF_PLOT_INFO["PCS-JCT"]["color"],
        label=SYSTEM,
    )
    ax.bar(
        np.arange(len(x_labels)) + 0.2,
        bar_values[1],
        width=0.4,
        color=CDF_PLOT_INFO["AFS"]["color"],
        label="AFS",
    )
    ax.tick_params("y", labelsize=22)
    ax.tick_params("x", labelsize=22)
    ax.set_ylabel("Normalized JCT", fontsize=22)
    ax.set_xlabel("Job Size Estimation Error", fontsize=22)
    ax.set_ylim([0.9, 1.1])
    ax.legend(loc="upper right", fontsize=22, frameon=False)
    _save_image("graphs", "fig10b.pdf")
    return ax


def _get_bar_indices(bar_groups, bars_per_group, bar_width, spacing):
    center = np.arange(0, spacing * bar_groups, spacing)
    bar_indices = list()
    if bars_per_group % 2 == 0:
        for n in range((bars_per_group // 2) - 1, -1, -1):
            bar_indices.append((center - (bar_width / 2)) - (n * bar_width))
        for p in range(0, (bars_per_group // 2)):
            bar_indices.append((center + (bar_width / 2)) + (p * bar_width))
    else:
        for n in range((bars_per_group // 2), 0, -1):
            bar_indices.append(center - n * bar_width)
        bar_indices.append(center)
        for p in range(1, (bars_per_group // 2) + 1):
            bar_indices.append(center + p * bar_width)
    return bar_indices



def draw_broken_indicator(x,y,w,h,ax):

    x = x-(w/2)

    break_patch = Rectangle((x, y), w, h, color='white')
    ax.add_patch(break_patch)




def draw_error_line(x, start_y, end_y, bar_width, ax):
    error_line = Line2D([x, x], [start_y, end_y], color='black', linestyle='-', linewidth=1)


    T_marker = Line2D([x-(bar_width/4), x+(bar_width/4)], [end_y, end_y], color='black', linestyle='-', linewidth=1)
    # T_marker = Rectangle((x-(bar_width/2), end_y-1), bar_width, 1, color='black', linestyle='-', linewidth=1)


    ax.add_line(error_line)

    if start_y != end_y:
        # ax.add_patch(T_marker)
        ax.add_line(T_marker)



