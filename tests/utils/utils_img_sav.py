import os
import matplotlib.pyplot as plt
from pathlib import Path
from domainlab.utils.test_img import mk_img
from domainlab.utils.utils_img_sav import mk_fun_sav_img

def test_fun():
    imgs = mk_img(28, batch_size=16)
    fun = mk_fun_sav_img()
    fun(imgs, name="hi.png", title="hw")
