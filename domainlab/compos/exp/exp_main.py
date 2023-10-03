"""
experiment
"""
import datetime
import os


from domainlab.algos.zoo_algos import AlgoBuilderChainNodeGetter
from domainlab.compos.exp.exp_utils import AggWriter
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
        if task is None:
            self.task = TaskChainNodeGetter(args)()
            if args.san_check:
                sancheck = SanityCheck(args, self.task)
                sancheck.dataset_sanity_check()

        self.args = args
        algo_builder = AlgoBuilderChainNodeGetter(self.args.aname, self.args.apath)()  # request
        # the critical logic below is to avoid circular dependence between task initialization
        # and trainer initialization:
        self.task.init_business(node_algo_builder=algo_builder, args=args)
        # jigen algorithm builder has method dset_decoration_args_algo, which could AOP
        # into the task intilization process
        self.trainer, self.model, observer_default, device = algo_builder.init_business(self)
        if model is not None:
            self.model = model
        self.visitor = visitor(self)  # visitor depends on task initialization first
        self.epochs = self.args.epos
        self.epoch_counter = 1
        if observer is None:
            observer = observer_default
        observer.set_exp(self)
        if not self.trainer.flag_initialized:
            # for matchdg
            self.trainer.init_business(self.model, self.task, observer, device, args)

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
            logger.info(f"epoch: {epoch},"
                        f"now: {str(t_c)},"
                        f"epoch time: {t_c - t_before_epoch},"
                        f"used: {t_c - t_0},"
                        f"model: {self.visitor.model_name}")
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
