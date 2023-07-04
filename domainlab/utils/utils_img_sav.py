import os
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from domainlab.utils.logger import Logger


def mk_fun_sav_img(path=".", nrow=8, folder_na=""):
    """
    create torchvision.utils image saver
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    def my_sav_img(comparison_tensor_stack, name, title=None):
        f_p = os.path.join(path, folder_na, name)
        Path(os.path.dirname(f_p)).mkdir(parents=True, exist_ok=True)
        logger = Logger.get_logger()
        logger.info(f"saving to {f_p}")
        # works also if tensor is already in cpu
        tensor = comparison_tensor_stack.cpu()
        if title is None:
            save_image(tensor=tensor, nrow=nrow, fp=f_p)
        else:
            img_grid = make_grid(tensor=tensor, nrow=nrow)
            sav_add_title(img_grid, path=f_p, title="hi")
    return my_sav_img


def sav_add_title(grid_img, path, title):
    """
    add title and save image as matplotlib.pyplot
    """
    fig = plt.gcf()   # get current figure
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title)
    fig.savefig(path)
    fig.clf()  # clear figure
