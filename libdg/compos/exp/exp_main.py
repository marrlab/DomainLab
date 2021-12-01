import os
import datetime
from libdg.tasks.zoo_tasks import TaskChainNodeGetter
from libdg.compos.exp.exp_utils import AggWriter
from libdg.algos.zoo_algos import AlgoBuilderChainNodeGetter
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # debug


class Exp():
    """
    Exp is combination of Task, Algorithm, and Configuration (including random seed)
    """
    def __init__(self, args, task=None):
        """
        :param args:
        :param task:
        """
        self.task = task
        if task is None:
            self.task = TaskChainNodeGetter(args)()
        self.task.init_business(args)
        self.args = args
        self.visitor = AggWriter(self)
        algo_builder = AlgoBuilderChainNodeGetter(self.args)()  # request
        self.trainer = algo_builder.init_business(self)
        self.epochs = self.args.epos
        self.epoch_counter = 1

    def execute(self):
        """
        train model
        check performance by loading persisted model
        """
        t_0 = datetime.datetime.now()
        print('\n Experiment start at :', str(t_0))
        t_c = t_0
        self.trainer.before_tr()
        for epoch in range(1, self.epochs + 1):
            t_before_epoch = t_c
            flag_stop = self.trainer.tr_epoch(epoch)
            t_c = datetime.datetime.now()
            print("now: ", str(t_c), "epoch time: ", t_c - t_before_epoch, "used: ", t_c - t_0)
            # current time, time since experiment start, epoch time
            if flag_stop:
                self.epoch_counter = epoch
                break
            elif epoch == self.epochs:
                self.epoch_counter = self.epochs
            else:
                self.epoch_counter += 1
        print("Experiment finished at epoch:", self.epoch_counter,
              "with time:", t_c - t_0, "at", t_c)
        self.trainer.post_tr()

def test_exp():
    from libdg.utils.arg_parser import mk_parser_main
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "2", "--task", "mnistcolor4", "--debug"])
    exp = Exp(args)
    exp.execute()
