"""
Test a trainer functionality
"""

from domainlab.algos.trainers.a_trainer import AbstractTrainer

class TrainerTest(AbstractTrainer):
    """
    A test trainer class conforming to model naming
    """

    def __init__(self):
        super().__init__()
        self.test_param = 42

    def tr_epoch(self, epoch):
        """
        :param epoch:
        """

    def before_tr(self):
        """
        before training, probe model performance
        """

def test_print_parameters(capsys):
    """
    Test the printing of parameters
    """
    trainer = TrainerTest()
    trainer.print_parameters()
    captured = capsys.readouterr()
    assert "Parameters of TrainerTest:" in captured.out
    assert "'test_param': 42" in captured.out
