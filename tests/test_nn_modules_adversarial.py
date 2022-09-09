'''
Code coverage issues:
    https://app.codecov.io/gh/marrlab/DomainLab/blob/master/domainlab/compos/nn_zoo/net_adversarial.py
    - lines 6-7
    - lines 29-30, 34-35
'''

import torch
from domainlab.compos.nn_zoo.net_adversarial import Flatten, AutoGradFunMultiply

def test_mflatten():
    input = torch.randn(1,3,28,28)
    m_flatten = Flatten()
    out = m_flatten(input)
    

def test_autograd_multiply():
    x = torch.randn(1,3,28,28)
    alpha = torch.randn(1,3,28,28)
    
    mm = AutoGradFunMultiply()
    out = mm(x, alpha)
    out.backward()