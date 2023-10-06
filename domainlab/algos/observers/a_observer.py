"""
interface for observer + visitor
"""
import abc


class AObVisitor(metaclass=abc.ABCMeta):
    """
    Observer + Visitor pattern for model selection
    """
    def __init__(self):
        self.exp = None
        self.task = None
        self.epo_te = None
        self.str_msel = None
        self.keep_model = None
        self.loader_te = None
        self.loader_tr = None
        self.loader_val = None

    def set_exp(self, exp):
        """
        couple observer with exp
        """
        self.exp = exp
        self.task = self.exp.task
        self.epo_te = self.exp.args.epo_te
        self.str_msel = self.exp.args.msel  # FIXME: consistent with self.model_sel?
        self.keep_model = self.exp.args.keep_model
        self.loader_te = self.task.loader_te
        self.loader_tr = self.task.loader_tr
        # Note loader_tr behaves/inherit different properties than loader_te
        self.loader_val = self.task.loader_val

    @abc.abstractmethod
    def update(self, epoch) -> bool:
        """
        return True/False whether the trainer should stop training
        """

    @abc.abstractmethod
    def accept(self, trainer):
        """
        accept invitation as a visitor
        """

    @abc.abstractmethod
    def after_all(self):
        """
        After training is done
        """

    @abc.abstractmethod
    def clean_up(self):
        """
        to be called by a decorator
        """
