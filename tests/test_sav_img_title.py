'''
Code coverage issues:
    https://app.codecov.io/gh/marrlab/DomainLab/blob/master/domainlab/utils/utils_img_sav.py
    - lines 22-23
    - lines 31-35
'''

import torch
from domainlab.utils.utils_img_sav import mk_fun_sav_img, sav_add_title

def test_save_img():
    """
    test sav_img function
    """
    imgs = torch.randn(1, 3, 28, 28)
    tt_sav_img = mk_fun_sav_img()
    tt_sav_img(imgs, name='rand_img.png', title='random_img')
    
def test_add_title():
    """
    test sav_add_title
    """
    img = torch.randn(3, 28, 28)
    sav_add_title(img, path='.', title='random_img')
