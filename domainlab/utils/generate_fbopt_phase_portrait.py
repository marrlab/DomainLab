import glob
import os
import argparse

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator


def get_xy_from_event_file(event_file, str1, str2,
                           tf_size_guidance=None,
                           sanity_check=False):
    if tf_size_guidance is None:
        # settings for which/how much data is loaded from the
        # tensorboard event files
        tf_size_guidance = {
            'compressedHistograms': 0,
            'images': 0,
            'scalars': 1e10,  # keep unlimited number
            'histograms': 0
        }
    # load event file
    event = EventAccumulator(event_file, tf_size_guidance)
    event.Reload()
    # extract the reg/dyn0 values
    y_event = event.Scalars(str1)
    y = [s.value for s in y_event]
    x_int = [s.step for s in y_event]
    # the .step data are saved as ints in tensorboard,
    # so we will re-extact from 'task'
    # extract the corresponding 'task' values
    x_event = event.Scalars(str2)
    x = [s.value for s in x_event]
    # sanity check:
    if sanity_check:
        for i in range(len(x)):
            assert int(x[i]) == x_int[i]
    return x, y


def phase_portrain_combined(event_files, colors, str1, str2, output_dir="."):
    plt.figure()

    for event_i in range(len(event_files)):
        x, y = get_xy_from_event_file(event_files[event_i],
                                      str1=str1, str2=str2)

        assert len(x) == len(y)
        for i in range(len(x) - 1):
            plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]),
                      head_width=0.2, head_length=0.2,
                      length_includes_head=True,
                      fc=colors[event_i], ec=colors[event_i], alpha=0.4)

        plt.plot(x[0], y[0], 'ko')
        plt.scatter(x, y, s=1, c='black')

        plt.xlabel(str1)
        plt.ylabel(str2)
        plt.title("phase portrait")

    plt.savefig(os.path.join(output_dir,
                             'phase_portrain_combined.png'), dpi=300)


def curve_combined(event_files, colors, str1, str2, output_dir="."):
    plt.figure()
    for event_i in range(len(event_files)):
        x, y = get_xy_from_event_file(event_files[event_i],
                                      str1=str1, str2=str2)
        plt.plot(x)
        plt.plot(y)
        plt.xlabel("time")
        plt.ylabel("loss")
        plt.title("timecourse")
        plt.legend([str1, str2])

    plt.savefig(os.path.join(output_dir,
                             f'timecourse_{str1}_{str2}.png'), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot')
    parser.add_argument('-str1', "--str1", default=None, type=str)
    parser.add_argument('-str2', "--str2", default=None, type=str)
    args = parser.parse_args()

    event_files = glob.glob("runs/*/events*")
    print("Using the following tensorboard event files:\n{}".format(
        "\n".join(event_files)))
    cmap = plt.get_cmap('tab10')  # Choose a colormap
    colors = [cmap(i) for i in range(len(event_files))]
    # Different colors for the different runs
    phase_portrain_combined(event_files, colors,
                            str1=args.str1, str2=args.str2)
    curve_combined(event_files, colors, str1=args.str1, str2=args.str2)
