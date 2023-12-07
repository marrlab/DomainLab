"""
observer and visitor pattern, responsible train, validation, test
dispatch performance evaluation to model,
dispatch model selection to model selection object
"""
import os

from domainlab.algos.observers.a_observer import AObVisitor
from domainlab.tasks.task_folder_mk import NodeTaskFolderClassNaMismatch
from domainlab.tasks.task_pathlist import NodeTaskPathListDummy
from domainlab.utils.logger import Logger


class ObVisitor(AObVisitor):
    """
    Observer + Visitor pattern for model selection
    """
    def __init__(self, model_sel):
        """
        observer trainer
        """
        super().__init__()
        self.host_trainer = None
        self.model_sel = model_sel
        self.epo = None
        self.metric_te = None
        self.metric_val = None
        self.perf_metric = None

    @property
    def str_metric4msel(self):
        """
        string representing the metric used for persisting models on the disk
        """
        return self.host_trainer.str_metric4msel

    def reset(self):
        """
        reset observer via reset model selector
        """
        self.model_sel.reset()

    def update(self, epoch):
        logger = Logger.get_logger()
        logger.info(f"epoch: {epoch}")
        self.epo = epoch
        if epoch % self.epo_te == 0:
            logger.info("---- Training Domain: ")
            self.host_trainer.model.cal_perf_metric(
                self.loader_tr, self.device)
            if self.loader_val is not None:
                logger.info("---- Validation: ")
                self.metric_val = self.host_trainer.model.cal_perf_metric(
                    self.loader_val, self.device)
            if self.loader_te is not None:
                logger.info("---- Test Domain (oracle): ")
                metric_te = self.host_trainer.model.cal_perf_metric(
                    self.loader_te, self.device)
                self.metric_te = metric_te
        if self.model_sel.update():
            logger.info("better model found")
            self.host_trainer.model.save()
            logger.info("persisted")
        flag_stop = self.model_sel.if_stop()
        flag_enough = epoch > self.host_trainer.aconf.epos_min
        return flag_stop & flag_enough

    def accept(self, trainer):
        """
        accept invitation as a visitor
        """
        self.host_trainer = trainer
        self.model_sel.accept(trainer, self)
        self.set_task(trainer.task, args=trainer.aconf,  device=trainer.device)
        self.perf_metric = self.host_trainer.model.create_perf_obj(self.task)

    def after_all(self):
        """
        After training is done
        """
        model_ld = None
        try:
            model_ld = self.host_trainer.model.load()
        except FileNotFoundError as err:
            # if other errors/exceptions occur, we do not catch them
            # other exceptions will terminate the python script
            # this can happen if loss is increasing, model never get selected
            logger = Logger.get_logger()
            logger.warning(err)
            logger.warning("this error can occur if model selection criteria \
                           is worsening, "
                           "model never get persisted, \
                           no performance metric is reported")
            return
        model_ld = model_ld.to(self.device)
        model_ld.eval()
        logger = Logger.get_logger()
        logger.info("persisted model performance metric: \n")
        metric_te = model_ld.cal_perf_metric(self.loader_te, self.device)
        dict_2add = self.cal_oracle_perf()
        if dict_2add is not None:
            metric_te.update(dict_2add)
        else:
            metric_te.update({"acc_oracle": -1})
        if hasattr(self, "model_sel"):
            metric_te.update({"acc_val": self.model_sel.best_val_acc})
        else:
            metric_te.update({"acc_val": -1})
        self.dump_prediction(model_ld, metric_te)
        # save metric to one line in csv result file
        self.host_trainer.model.visitor(metric_te)
        # prediction dump of test domain is essential
        # to verify the prediction results

    def cal_oracle_perf(self):
        """
        calculate oracle performance
        """
        try:
            model_or = self.host_trainer.model.load("oracle")
            # @FIXME: name "oracle is a strong dependency
            model_or = model_or.to(self.device)
            model_or.eval()
        except FileNotFoundError:
            return {"acc_oracle": -1}
        logger = Logger.get_logger()
        logger.info("oracle model performance metric: \n")
        metric_te = model_or.cal_perf_metric(self.loader_te, self.device)
        return {"acc_oracle": metric_te["acc"]}

    def dump_prediction(self, model_ld, metric_te):
        """
        given the test domain loader, use the loaded model \
            model_ld to predict each instance
        """
        flag_task_folder = isinstance(
            self.host_trainer.task, NodeTaskFolderClassNaMismatch)
        flag_task_path_list = isinstance(
            self.host_trainer.task, NodeTaskPathListDummy)
        if flag_task_folder or flag_task_path_list:
            fname4model = self.host_trainer.model.visitor.model_path  # pylint: disable=E1101
            file_prefix = os.path.splitext(fname4model)[0]  # remove ".model"
            dir4preds = os.path.join(self.host_trainer.aconf.out, "saved_predicts")
            if not os.path.exists(dir4preds):
                os.mkdir(dir4preds)
            file_prefix = os.path.join(dir4preds,
                                       os.path.basename(file_prefix))
            file_name = file_prefix + "_instance_wise_predictions.txt"
            model_ld.pred2file(
                self.loader_te, self.device,
                filename=file_name,
                metric_te=metric_te)

    def clean_up(self):
        """
        to be called by a decorator
        """
        if not self.keep_model:
            self.host_trainer.model.visitor.clean_up()
