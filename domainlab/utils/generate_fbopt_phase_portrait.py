import glob
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# FIXME: maybe adjust the output path where the png is saved
output_dir = "../.."

def get_xy_from_event_file(event_file, tf_size_guidance=None):
    if tf_size_guidance is None:
        # settings for which/how much data is loaded from the tensorboard event files
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
    y_event = event.Scalars('x-axis=task vs y-axis=reg/dyn0')
    y = [s.value for s in y_event]
    x_int = [s.step for s in y_event]  # the .step data are saved as ints in tensorboard, so we will re-extact from 'task'
    # extract the corresponding 'task' values
    x_event = event.Scalars('task')
    x = [s.value for s in x_event]
    # sanity check:
    for i in range(len(x)):
        assert int(x[i]) == x_int[i]
    return x, y

def phase_portrain_combined(event_files, colors):
    plt.figure()

    for event_i in range(len(event_files)):
        x, y = get_xy_from_event_file(event_files[event_i])

        assert len(x) == len(y)
        for i in range(len(x)-1):
            plt.arrow(x[i], y[i], (x[i+1]-x[i]), (y[i+1]-y[i]),
                      head_width=0.2, head_length=0.2, length_includes_head=True,
                      fc=colors[event_i], ec=colors[event_i], alpha=0.4)

        plt.plot(x[0], y[0], 'ko')
        plt.scatter(x, y, s=1, c='black')

        plt.xlabel("task")
        plt.ylabel("reg/dyn0")
        plt.title("x-axis=task vs y-axis=reg/dyn0")

    plt.savefig(os.path.join(output_dir, 'phase_portrain_combined.png'), dpi=300)


if __name__ == "__main__":
    event_files = glob.glob("../../runs/*/events*")
    print("Using the following tensorboard event files:\n{}".format("\n".join(event_files)))
    cmap = plt.get_cmap('tab10')  # Choose a colormap
    colors = [cmap(i) for i in range(len(event_files))]  # Different colors for the different runs
    phase_portrain_combined(event_files, colors)

