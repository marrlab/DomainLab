"""
experiment
"""
import datetime
import os
import warnings


from domainlab.algos.zoo_algos import AlgoBuilderChainNodeGetter
from domainlab.exp.exp_utils import AggWriter
from domainlab.tasks.zoo_tasks import TaskChainNodeGetter
from domainlab.utils.sanity_check import SanityCheck
from domainlab.utils.logger import Logger

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # debug


class Exp():
    """
    Exp is combination of Task, Algorithm, and Configuration (including random seed)
    """
    def __init__(self, args, task=None, model=None, observer=None, visitor=AggWriter):
        """
        :param args:
        :param task: default None
        :param model: default None
        """
        self.task = task
        self.curr_dir = os.getcwd()
        if task is None:
            self.task = TaskChainNodeGetter(args)()

        self.args = args
        algo_builder = AlgoBuilderChainNodeGetter(self.args.aname, self.args.apath)()  # request
        # the critical logic below is to avoid circular dependence between task initialization
        # and trainer initialization:

        # jigen algorithm builder has method dset_decoration_args_algo, which could AOP
        # into the task intilization process
        if args.san_check:
            sancheck = SanityCheck(args, self.task)
            sancheck.dataset_sanity_check()

        self.trainer, self.model, observer_default, device = algo_builder.init_business(self)
        if model is not None:
            self.model = model
        self.epochs = self.args.epos
        self.epoch_counter = 1
        if observer is None:
            observer = observer_default
        if not self.trainer.flag_initialized:
            self.trainer.init_business(self.model, self.task, observer, device, args)
        self.visitor = visitor(self)  # visitor depends on task initialization first
        # visitor must be initialized last after trainer is initialized
        self.model.set_saver(self.visitor)

    def execute(self, num_epochs=None):
        """
        train model
        check performance by loading persisted model
        """
        if num_epochs is None:
            num_epochs = self.epochs + 1
        t_0 = datetime.datetime.now()
        logger = Logger.get_logger()
        logger.info(f'\n Experiment start at: {str(t_0)}')
        t_c = t_0
        self.trainer.before_tr()
        for epoch in range(1, num_epochs):
            t_before_epoch = t_c
            flag_stop = self.trainer.tr_epoch(epoch)
            t_c = datetime.datetime.now()
            logger.info(f"after epoch: {epoch},"
                        f"now: {str(t_c)},"
                        f"epoch time: {t_c - t_before_epoch},"
                        f"used: {t_c - t_0},"
                        f"model: {self.visitor.model_name}")
            logger.info(f"working direcotry: {self.curr_dir}")
            # current time, time since experiment start, epoch time
            if flag_stop:
                self.epoch_counter = epoch
                logger.info("early stop trigger")
                break
            if epoch == self.epochs:
                self.epoch_counter = self.epochs
            else:
                self.epoch_counter += 1
        logger.info(f"Experiment finished at epoch: {self.epoch_counter} "
                    f"with time: {t_c - t_0} at {t_c}")
        self.trainer.post_tr()

    def clean_up(self):
        """
        to be called by a decorator
        """
        try:
            # oracle means use out-of-domain test accuracy to select the model
            self.visitor.remove("oracle")  # pylint: disable=E1101
        except FileNotFoundError:
            pass

        try:
            # the last epoch:
            # have a model to evaluate in case the training stops in between
            self.visitor.remove("epoch")  # pylint: disable=E1101
        except FileNotFoundError:
            logger = Logger.get_logger()
            logger.warn("failed to remove model_epoch: file not found")
            warnings.warn("failed to remove model_epoch: file not found")

        try:
            # without suffix: the selected model
            self.visitor.remove()  # pylint: disable=E1101
        except FileNotFoundError:
            logger = Logger.get_logger()
            logger.warn("failed to remove model")
            warnings.warn("failed to remove model")

        try:
            # for matchdg
            self.visitor.remove("ctr")  # pylint: disable=E1101
        except FileNotFoundError:
            pass
