import pickle
from wfq_tuner import *
import matplotlib.pyplot as plt
import os, sys
from mpl_toolkits.mplot3d import Axes3D
from jmetal.util.solution import get_non_dominated_solutions
from functools import partial
from matplotlib.widgets import Button
import seaborn as sns
from matplotlib.patches import Ellipse

import argparse


def fix_label(label):
    if "jct" in label:
        label = "normalized " + label

    if "error" in label:
        label = label + " %"

    label = label.replace("_", " ")
    label = label.replace("estimation", "prediction")
    label = label.title()
    label = label.replace("Jct", "JCT")

    return label


def set_canvas(
    ax,
    x_label=None,
    y_label=None,
    x_lim=None,
    y_lim=None,
    y_ticks=None,
    x_ticks=None,
    legend_loc="best",
    legend_ncol=1,
    showgrid=True,
    legendfsize=15,
    showlegend=False,
):
    ax.set_facecolor(("#c8cbcf"))

    x_label = fix_label(x_label)
    y_label = fix_label(y_label)

    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["left"].set_color("white")

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=15)

    if showlegend:
        ax.legend(
            loc=legend_loc,
            ncol=legend_ncol,
            prop={"family": "monospace", "size": legendfsize},
        )

    if x_label:
        ax.set_xlabel(x_label, fontsize=20, family="monospace")
    if y_label:
        ax.set_ylabel(y_label, fontsize=20, family="monospace")

    if x_ticks:
        ax.set_xticks(x_ticks[0])
        ax.set_xticklabels(x_ticks[1])

    if y_ticks:
        ax.set_yticks(y_ticks[0])
        ax.set_yticklabels(y_ticks[1])

    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])

    if showgrid:
        ax.grid(alpha=0.5, color="white", linestyle="-", zorder=0)


def normalize(D):
    D = [round(d / min(D), 3) for d in D]
    return D[:]


def get_closest_config(point1):
    distances = list()

    for solution in pareto_front_configs.values():
        point2 = [solution["mean_error"], solution["mean_jct"]]
        distances.append(
            [np.linalg.norm(np.asarray(point1) - np.asarray(point2)), solution]
        )

    distances = sorted(distances, key=lambda e: e[0])

    closest_config = distances[0][1]

    print(
        "closest_config: mean_error: {}, mean_jct: {} num_classes: {} thresolds: {} rates: {}".format(
            closest_config["mean_error"],
            closest_config["mean_jct"],
            closest_config["num_classes"],
            closest_config["class_thresholds"],
            closest_config["class_rates"],
        )
    )


class Callback(object):
    """docstring for Callback"""

    def __init__(self, plot_handle, fig, checkpoint, out_fname):
        self.plot_handle = plot_handle
        self.fig = fig
        self.checkpoint = checkpoint
        self.selected_config = None
        self._addons = None
        ###############################################################################
        # Added out_fname as an attribute because it was querying the global variable
        # args which was declared in the __main__ block. With the new setup (i.e. the
        # main( ) function) this has to be explicitly passed here and then used in
        # create_config( ).
        # 11/29/23
        ###############################################################################
        self._output_fname = out_fname

    def on_pick(self, event):
        solutions = get_non_dominated_solutions(self.checkpoint["SOLUTIONS"])
        ###############################################################################
        # Fixed to use self.checkpoint not the global checkpoint variable in old
        # __main__ block.
        # 11/29/23
        ###############################################################################
        problem = self.checkpoint["PROBLEM"]

        for ind in range(len(solutions)):
            self.plot_handle._facecolors[ind, :] = (1, 1, 1, 1)
            self.plot_handle._edgecolors[ind, :] = tuple(list(BLUE) + [1])

        ind, *_ = event.ind

        # turning selected point red
        self.plot_handle._facecolors[event.ind, :] = (1, 1, 1, 1)
        self.plot_handle._edgecolors[event.ind, :] = tuple(list(RED) + [1])

        d = self.plot_handle.get_offsets().data[event.ind][0]
        x, y = d

        closest_solution = solutions[ind]

        # closest_solution.variables.append(1.0)
        # print("warning not handling config delta")

        config = problem.solution_transformer(closest_solution)

        self.selected_config = config
        print(closest_solution.objectives)
        print(config)

        if self._addons:
            self._addons.remove()

        string = "<"
        for i in range(len(problem.obj_labels)):
            string += f"{problem.obj_labels[i]}: {round(d[i], 2)}, "
        string = string[:-2]
        string += ">"
        ###############################################################################
        # Fixed to use self.fig not the global fix variable in old __main__ block.
        # 11/29/23
        ###############################################################################
        self._addons = self.fig.axes[0].text(x + 0.1, y, string)

        # print(config)
        # print(closest_solution.objectives)

        self.fig.canvas.draw()
        plt.draw()

    def create_config(self, event):
        class_detail = self.selected_config

        """
        file_name = "{}_{}_{}_{}.pkl".format(
            class_detail["num_classes"],
            ",".join(list(map(str,class_detail["class_thresholds"]))),
            ",".join(list(map(lambda r: str(r.numerator),class_detail["class_rates"]))),
            round(class_detail["clip_demand_factor"], 2))


        print(f"Saving config to {file_name}")
        """

        ###############################################################################
        # Changed to use new attribute (see __init__( ) comment)
        # 11/29/23
        ###############################################################################
        with open(self._output_fname, "wb") as fp:
            pickle.dump(class_detail, fp)

        ###############################################################################
        # Added so that the interactive window closes upon clicking create config.
        # Clicking on multiple configs just overwrites the previous choices anyway.
        # 11/29/23
        ###############################################################################
        plt.close(self.fig)


colors = sns.color_palette("tab10")
BLUE = colors[0]
RED = colors[3]


def sol_dis(s1, s2):
    return np.linalg.norm(np.asarray(s1.objectives) - np.asarray(s2.objectives))


def draw_tri(offset_x, offset_y, ax, scalex=1, scaley=1):
    tri_x = np.multiply([0, 1, 0], scalex)
    tri_y = np.multiply([0, 0, 1], scaley)

    tri_x = np.add(tri_x, offset_x)
    tri_y = np.add(tri_y, offset_y)

    ax.fill(tri_x, tri_y, color="k", transform=ax.transAxes, zorder=3)


def filter_solutions(solutions, to_filter=False):
    if not to_filter:
        return solutions

    N = 20000

    print(f"WARNING: MANUAL DISTANCE SET AS {N}")

    filtered_solutions = [solutions[0]]
    for solution in solutions:
        s1 = filtered_solutions[-1]
        if sol_dis(s1, solution) > N:
            filtered_solutions.append(solution)

    return filtered_solutions


def draw_better_marker(aspect_ratio, ax, x=0.5, y=0.5):
    degree = np.rad2deg(np.arctan(aspect_ratio[1] / aspect_ratio[0]))

    ellipse = Ellipse(
        xy=(x, y),
        width=0.2 * sum(aspect_ratio) / aspect_ratio[1],
        height=0.1 * sum(aspect_ratio) / aspect_ratio[0],
        edgecolor=colors[1],
        fc=colors[1],
        lw=2,
        angle=60,
        transform=ax.transAxes,
        zorder=2,
    )

    ax.add_patch(ellipse)

    # ax.text(2,2.5,"Better", rotation=45.0, ha='center', va='center', size="xx-large")
    ax.text(
        x + 0.005,
        y - 0.02,
        "Better",
        rotation=45,
        ha="center",
        va="center",
        size=25,
        transform=ax.transAxes,
        zorder=3,
    )

    draw_tri(
        offset_x=x - 0.13,
        offset_y=y - 0.2,
        ax=ax,
        scalex=0.025 * sum(aspect_ratio) / aspect_ratio[0],
        scaley=0.025 * sum(aspect_ratio) / aspect_ratio[1],
    )


def main(args):
    fname = args.fname
    savefig = not args.interactive

    loaded_obj = []

    # with open("learnt_configs/"+fname, 'rb') as fp:
    with open(fname, "rb") as fp:
        while True:
            try:
                loaded_obj.append(pickle.load(fp))
            except Exception as e:
                break

    header = loaded_obj[0]

    objectives = header["objectives"]

    # print("workload: {}".format(header["workload"]))

    print([objective.get_name() for objective in objectives])

    # checkpoint?
    checkpoint = loaded_obj[-1]

    solutions = get_non_dominated_solutions(checkpoint["SOLUTIONS"])

    print("workload: {}".format(header["workload"]))
    print(f"num_solutions: {len(solutions)}")

    solutions = filter_solutions(sorted(solutions, key=lambda s: s.objectives[0]))
    # solutions = (sorted(solutions, key=lambda s: s.objectives[0]))

    checkpoint["SOLUTIONS"] = solutions
    solutions = get_non_dominated_solutions(checkpoint["SOLUTIONS"])

    # solutions = checkpoint["SOLUTIONS"]
    evaluations = checkpoint["EVALUATIONS"]
    computing_time = checkpoint["COMPUTING_TIME"]

    # print(solutions[0])

    # sys.exit(1)

    points = [solution.objectives[:] for solution in solutions]

    # points = sorted(points, key=lambda p: p[0])

    # fig, axs = plt.subplots(1,figsize=(8,6))

    # normalize jct

    data = list()
    for didex in range(len(objectives)):
        D = list(map(lambda p: p[didex], points))

        if "jct" in objectives[didex].get_name():
            D = normalize(D)

        data.append(D)

    aspect_ratio = (6, 4)

    fig = plt.figure(figsize=aspect_ratio)

    plot_handle = None

    if len(objectives) == 1:
        print("Just one objective, nothing to plot")
        config = checkpoint["PROBLEM"].solution_transformer(solutions[0])

        print(config)

        with open(args.output_fname, "wb") as fp:
            pickle.dump(config, fp)

        sys.exit(1)
    elif len(objectives) == 2:
        ax = fig.add_subplot(111)
        plot_handle = ax.scatter(
            data[0],
            data[1],
            color=["w"] * len(data[0]),
            edgecolors=[colors[0]] * len(data[0]),
            s=[400] * len(data[0]),
            linewidth=1.75,
            label="Pareto-optimal WFQ configs",
            picker=True,
            zorder=3,
        )
        ax.plot(data[0], data[1], color=BLUE, linewidth=2, zorder=2)

    elif len(objectives) == 3:
        ax = fig.add_subplot(111, projection="3d")
        plot_handle = ax.scatter(
            data[0],
            data[1],
            data[2],
            color=[colors[0]] * len(data[0]),
            s=[200] * len(data[0]),
            label="Pareto-optimal WFQ configs",
            picker=True,
        )
        ax.set_zlabel(objectives[2].get_name())

        # tuple(list(BLUE) + [1])

    draw_better_marker(aspect_ratio, ax, x=0.55, y=0.55)

    # ax.set_title(f"{evaluations} Configurations Evaluated in {round(computing_time/60.0, 2)} Minutes")

    set_canvas(
        ax,
        x_label=objectives[0].get_name(),
        y_label=objectives[1].get_name(),
        showgrid=True,
        showlegend=True,
    )

    fig.subplots_adjust(left=0.12, right=1, bottom=0.18, top=1)

    if savefig:
        plt.savefig(fname.replace(".pkl", ".pdf"), dpi=300, format="pdf")
    else:
        interact = Callback(plot_handle, fig, checkpoint, args.output_fname)

        fig.canvas.mpl_connect("pick_event", interact.on_pick)

        ax_button = fig.add_axes([0.75, 0.8, 0.148, 0.075])
        create_config_button = Button(ax_button, "Create Config")
        create_config_button.on_clicked(interact.create_config)

        plt.show(block=True)

    """
    problem = checkpoint["PROBLEM"]
    solution = solutions[2]

    class_detail = problem.solution_transformer(solution)
    
    print(class_detail["num_classes"], class_detail["clip_demand_factor"])
    
    print(solution.objectives)

    eval_solution = problem.evaluate(solution)
    print(eval_solution)
    """

    # for rates in class_detail["class_rates"]:
    #     print(float(rates))

    # ax.set_zlabel("unfairness")

    # axs.plot(X,Y, lw=5, linestyle="--")
    # plt.xlabel(, )
    # plt.ylabel()

    # plt.rcParams.update({'font.size': 22})

    """
    S_X = []
    S_Y = []

    for i in range(len(X)):

        if i%10==0:
            S_X.append(X[i])
            S_Y.append(Y[i])


    S_X.append(X[-1])
    S_Y.append(Y[-1])

    axs.scatter(S_X, S_Y, s=200, label="FLEX configurations")
    """

    # set_canvas(axs, "Avg Pred Error %", "Normalized Avg JCT", showlegend=True)

    # plt.savefig("pareto.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw pareto")
    parser.add_argument(
        "-fname", type=str, help="name of input pkl file", default=None, required=True
    )
    parser.add_argument(
        "-interactive", type=int, help="interactive mode (1/0)", default=0
    )
    parser.add_argument(
        "-output_fname", type=str, help="output_fname", default="MCS_config.pkl"
    )
    args = parser.parse_args()
    main(args)
