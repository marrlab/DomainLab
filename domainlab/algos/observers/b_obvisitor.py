"""
observer and visitor pattern, responsible train, validation, test
dispatch performance evaluation to model, dispatch model selection to model selection object
"""
import os
import warnings

from domainlab.algos.observers.a_observer import AObVisitor
from domainlab.tasks.task_folder_mk import NodeTaskFolderClassNaMismatch
from domainlab.tasks.task_pathlist import NodeTaskPathListDummy
from domainlab.utils.logger import Logger


class ObVisitor(AObVisitor):
    """
    Observer + Visitor pattern for model selection
    """
    def __init__(self, task, visitor, model_sel, out, epo_te, str_msel, keep_model, device):
        """
        observer trainer
        """
        self.out = out
        self.host_trainer = None
        self.model_sel = model_sel
        self.visitor = visitor
        self.device = device
        self.task = task
        self.loader_te = self.task.loader_te
        self.loader_tr = self.task.loader_tr
        self.loader_val = self.task.loader_val
        # Note loader_tr behaves/inherit different properties than loader_te
        self.epo_te = epo_te
        self.str_msel = str_msel
        self.epo = None
        self.metric_te = None
        self.metric_val = None
        self.keep_model = keep_model
        self.perf_metric = None

    def update(self, epoch):
        logger = Logger.get_logger()
        logger.info(f"epoch: {epoch}")
        self.epo = epoch
        if epoch % self.epo_te == 0:
            logger.debug("---- Training Domain: ")
            self.host_trainer.model.cal_perf_metric(self.loader_tr, self.device)
            if self.loader_val is not None and self.str_msel == "val":
                logger.debug("---- Validation: ")
                self.metric_val = self.host_trainer.model.cal_perf_metric(
                    self.loader_val, self.device)
            logger.debug("---- Test Domain (oracle): ")
            metric_te = self.host_trainer.model.cal_perf_metric(self.loader_te, self.device)
            self.metric_te = metric_te
        if self.model_sel.update():
            logger.info("better model found")
            self.visitor.save(self.host_trainer.model)
            logger.info("persisted")
        flag_stop = self.model_sel.if_stop()
        return flag_stop

    def accept(self, trainer):
        """
        accept invitation as a visitor
        """
        self.host_trainer = trainer
        self.perf_metric = self.host_trainer.model.create_perf_obj(self.task)
        self.model_sel.accept(trainer, self)

    def after_all(self):
        """
        After training is done
        """
        model_ld = None
        try:
            model_ld = self.visitor.load()
        except FileNotFoundError as err:  # if other errors/exceptions occur, we do not catch them
            # other exceptions will terminate the python script
            # this can happen if loss is increasing, model never get selected
            logger = Logger.get_logger()
            logger.error(err)
            logger.error("this error can occur if model selection criteria is worsening, "
                         "model never get persisted")
            return

        model_ld = model_ld.to(self.device)
        model_ld.eval()
        logger = Logger.get_logger()
        logger.info("persisted model performance metric: \n")
        metric_te = model_ld.cal_perf_metric(self.loader_te, self.device)
        self.dump_prediction(model_ld, metric_te)
        self.visitor(metric_te)
        # prediction dump of test domain is essential to verify the prediction results

    def dump_prediction(self, model_ld, metric_te):
        """
        given the test domain loader, use the loaded model model_ld to predict each instance
        """
        flag_task_folder = isinstance(self.task, NodeTaskFolderClassNaMismatch)
        flag_task_path_list = isinstance(self.task, NodeTaskPathListDummy)
        if flag_task_folder or flag_task_path_list:
            fname4model = self.visitor.model_path  # pylint: disable=E1101
            file_prefix = os.path.splitext(fname4model)[0]  # remove ".model"
            dir4preds = os.path.join(self.out, "saved_predicts")
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
