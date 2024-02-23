"""
This file is used for generating phase portrait from tensorboard event files.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class ListFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def write_lists_to_file(self, list1, list2=None):
        with open(self.file_path, 'w') as file:
            if list2 is None:
                for val1 in list1:
                    file.write(f"{val1}\n")
            else:
                for val1, val2 in zip(list1, list2):
                    file.write(f"{val1} {val2}\n")

    def read_lists_from_file(self):
        list1 = []
        list2 = []
        with open(self.file_path, 'r') as file:
            for line in file:
                values = list(map(float, line.strip().split()))
                if len(values) == 1:
                    list1.append(values[0])
                elif len(values) == 2:
                    list1.append(values[0])
                    list2.append(values[1])
        return list1, list2


# pylint: disable=too-many-arguments
def get_xy_from_event_file(
    event_file,
    plot1,
    plot2=None,
    tf_size_guidance=None,
    sanity_check=False,
    verbose=True,
):
    """
    extract x and y values from a tensorboard event file
    """
    if tf_size_guidance is None:
        # settings for which/how much data is loaded from the
        # tensorboard event files
        tf_size_guidance = {
            "compressedHistograms": 0,
            "images": 0,
            "scalars": 1e10,  # keep unlimited number
            "histograms": 0,
        }
    # load event file
    event = EventAccumulator(event_file, tf_size_guidance)
    event.Reload()
    # print names of available plots
    if verbose:
        print(f"Event file {event_file} -- available plots:")
        print(event.Tags()["scalars"])
    if plot2:
        # extract the plot2 values (e.g., reg/dyn0)
        y_event = event.Scalars(plot2)
        y = [s.value for s in y_event]
        x_int = [s.step for s in y_event]
        # the .step data are saved as ints in tensorboard,
        # (so, in case of phase portrait, we re-extact from 'task')
    else:
        y = None
    # extract the corresponding plot1 values (e.g., 'task')
    x_event = event.Scalars(plot1)
    x = [s.value for s in x_event]
    # sanity check (originally added for the reg/dyn0 vs. task phase portrait;
    # shouldn't be needed if plot1 and plot2 represent something else):
    if sanity_check:
        for i in range(len(x)):
            assert int(x[i]) == x_int[i]
    return x, y


# pylint: disable=too-many-arguments, too-many-locals, redefined-outer-name, unused-argument
def phase_portrait_combined(
    event_files,
    colors,
    plot1,
    plot2,
    legend1=None,
    legend2=None,
    plot_len=None,
    output_dir=".",
):
    """
    combined phase portait for multiple (at least one) Tensorboard
    event files in the same plot
    """
    plt.figure()

    for event_i in range(len(event_files)):
        x, y = get_xy_from_event_file(event_files[event_i], plot1=plot1, plot2=plot2)

        assert len(x) == len(y)
        if plot_len is None:
            plot_len = len(x)
        # truncate x and y to the desired length:
        x, y = x[:plot_len], y[:plot_len]

        head_w_glob = min((max(x) - min(x)) / 100.0, (max(y) - min(y)) / 100.0)
        for i in range(plot_len - 1):
            xy_dist = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
            head_l = xy_dist / 30.0
            head_w = min(head_l, head_w_glob)
            plt.arrow(
                x[i],
                y[i],
                (x[i + 1] - x[i]),
                (y[i + 1] - y[i]),
                head_width=head_w,
                head_length=head_l,
                length_includes_head=True,
                fc=colors[event_i],
                ec=colors[event_i],
                alpha=0.8,
            )
        # the combination of head_width and head_length make the arrow
        # more visible.
        # length_includes_head=False makes the arrow stick too far out
        # beyond of the point, which let; so, True is used.
        colors = ["red", "green", "blue", "yellow", "purple"]
        plt.plot(x[0], y[0], "ko")

        list_color = [colors[i % len(colors)] for i, h in enumerate(x)]
        plt.scatter(x, y, s=1, c=np.array(list_color))

        if legend1 is None:
            legend1 = plot1
        if legend2 is None:
            legend2 = plot2
        plt.xlabel(legend1)
        plt.ylabel(legend2)
        plt.title("phase portrait")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    legend22 = legend2.split(os.sep)[-1]
    plt.savefig(
        os.path.join(output_dir, f"phase_portrait_combined_{legend22}.png"), dpi=300
    )


def two_curves_combined(
    event_files,
    colors,
    plot1,
    plot2,
    legend1=None,
    legend2=None,
    output_dir=".",
    title=None,
):
    """
    FIXME: colors parameter is not used
    """
    plt.figure()
    for event_i in range(len(event_files)):
        x, y = get_xy_from_event_file(event_files[event_i], plot1=plot1, plot2=plot2)
        plt.plot(x, color="blue")
        plt.plot(y, color="red")
        plt.xlabel("epoch")
        # plt.ylabel("loss")
        if title is not None:
            plt.title(title)
        if legend1 is None:
            legend1 = plot1
        if legend2 is None:
            legend2 = plot2
        plt.legend([legend1, legend2])

    legend11 = legend1.replace(os.sep, "_")
    legend22 = legend2.replace(os.sep, "_")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(
        os.path.join(output_dir, f"timecourse_{legend11}_{legend22}.png"), dpi=300
    )


def plot_single_curve(event_files, colors, plot1, legend1=None, output_dir="."):
    """
    FIXME: colors parameter is not used
    """
    plt.figure()
    for event_i in range(len(event_files)):
        x, _ = get_xy_from_event_file(event_files[event_i], plot1=plot1)
        plt.plot(x)
        plt.xlabel("time")
        if legend1 is None:
            legend1 = plot1
        plt.ylabel(legend1)
        # plt.title("timecourse")

    legend11 = legend1.replace(os.sep, "_")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"timecourse_{legend11}.png"), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot")
    parser.add_argument("-plot1", "--plot1", default=None, type=str)
    parser.add_argument("-plot2", "--plot2", default=None, type=str)
    parser.add_argument("-legend1", "--legend1", default=None, type=str)
    parser.add_argument("-legend2", "--legend2", default=None, type=str)
    parser.add_argument("-plot_len", "--plot_len", default=None, type=int)
    parser.add_argument("-title", "--title", default=None, type=str)
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument("--runs_dir", default="runs", type=str)
    parser.add_argument(
        "--phase_portrait",
        action="store_true",
        help="if True plots a phase portrait,\
                        otherwise a curve (default)",
    )
    args = parser.parse_args()

    # get event files from all available runs
    # Tensorboard: * could be the date information, this intermediate directory
    # always exist
    # events* means all the event folders
    event_files = glob.glob(f"{args.runs_dir}/*/events*")
    print(
        "Using the following tensorboard event files:\n{}".format(
            "\n".join(event_files)
        )
    )

    # Different colors for the different runs
    cmap = plt.get_cmap("tab10")  # Choose a colormap
    colors = [cmap(i) for i in range(len(event_files))]

    if args.phase_portrait:
        phase_portrait_combined(
            event_files,
            colors,
            plot1=args.plot1,
            plot2=args.plot2,
            legend1=args.legend1,
            legend2=args.legend2,
            plot_len=args.plot_len,
            output_dir=args.output_dir,
        )
    else:
        if args.plot2:
            # two curves per plot
            two_curves_combined(
                event_files,
                colors,
                plot1=args.plot1,
                plot2=args.plot2,
                legend1=args.legend1,
                legend2=args.legend2,
                output_dir=args.output_dir,
                title=args.title,
            )
        else:
            # one curve per plot
            plot_single_curve(
                event_files,
                colors,
                plot1=args.plot1,
                legend1=args.legend1,
                output_dir=args.output_dir,
            )
