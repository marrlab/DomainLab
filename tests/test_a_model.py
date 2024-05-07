"""
Test a model functionality
"""

import pytest
from io import StringIO
import sys
from torch import nn
from domainlab.models.a_model import AModel

class ModelTest(AModel):
    """
    A test model class conforming to model naming
    """

    def __init__(self):
        super().__init__()
        self.test_param = 42

    def cal_task_loss(self, tensor_x, tensor_y):
        return 0

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        return 0


class InvalidTest(AModel):
    """
    A test model class that does not conform to the "Model" prefix naming convention
    """

    def cal_task_loss(self, tensor_x, tensor_y):
        return 0

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        return 0


def test_model_name_valid():
    """
    Test a valid model name
    """
    model = ModelTest()
    assert model.name == "test", f"Expected 'test' but got '{model.name}'"


def test_model_name_invalid():
    """
    Test an invalid model name
    """
    model = InvalidTest()
    with pytest.raises(RuntimeError, match="Model builder node class must start with"):
        model.name


def test_print_parameters(capsys):
    """
    Test the printing of parameters
    """
    model = ModelTest()
    model.print_parameters()
    captured = capsys.readouterr()
    assert "Parameters of ModelTest:" in captured.out
    assert "'test_param': 42" in captured.out    