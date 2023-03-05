import os
import warnings


from domainlab.algos.observers.a_observer import AObVisitor
from domainlab.tasks.task_folder_mk import NodeTaskFolderClassNaMismatch
from domainlab.tasks.task_pathlist import NodeTaskPathListDummy


class ObVisitor(AObVisitor):
    """
    Observer + Visitor pattern for model selection
    """
    def __init__(self, exp, model_sel, device):
        """
        observer trainer
        """
        self.host_trainer = None
        self.exp = exp
        self.model_sel = model_sel
        self.device = device
        self.task = self.exp.task
        self.loader_te = self.exp.task.loader_te
        self.loader_tr = self.exp.task.loader_tr
        self.loader_val = self.exp.task.loader_val
        # Note loader_tr behaves/inherit different properties than loader_te
        self.epo_te = self.exp.args.epo_te
        self.epo = None
        self.metric_te = None
        self.keep_model = self.exp.args.keep_model
        self.perf_metric = None

    def update(self, epoch):
        print("epoch:", epoch)
        self.epo = epoch
        if epoch % self.epo_te == 0:
            metric_te = self.host_trainer.model.cal_perf_metric(
                self.loader_tr, self.device, self.loader_te)
            self.metric_te = metric_te
        if self.model_sel.update():
            print("model selected")
            self.exp.visitor.save(self.host_trainer.model)
            print("persisted")
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
            model_ld = self.exp.visitor.load()
        except FileNotFoundError:
            # this can happen if loss is increasing, model never get selected
            return

        model_ld = model_ld.to(self.device)
        model_ld.eval()
        print("persisted model performance metric: \n")
        metric_te = model_ld.cal_perf_metric(self.loader_tr, self.device, self.loader_te)
        self.dump_prediction(model_ld, metric_te)
        self.exp.visitor(metric_te)
        # prediction dump of test domain is essential to verify the prediction results

    def dump_prediction(self, model_ld, metric_te):
        """
        given the test domain loader, use the loaded model model_ld to predict each instance
        """
        flag_task_folder = isinstance(self.exp.task, NodeTaskFolderClassNaMismatch)
        flag_task_path_list = isinstance(self.exp.task, NodeTaskPathListDummy)
        if flag_task_folder or flag_task_path_list:
            fname4model = self.exp.visitor.model_path  # pylint: disable=E1101
            file_prefix = os.path.splitext(fname4model)[0]  # remove ".model"
            dir4preds = os.path.join(self.exp.args.out, "saved_predicts")
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
                self.exp.visitor.remove("oracle")  # pylint: disable=E1101
            except FileNotFoundError:
                pass

            try:
                # the last epoch:
                # have a model to evaluate in case the training stops in between
                self.exp.visitor.remove("epoch")  # pylint: disable=E1101
            except FileNotFoundError:
                warnings.warn("failed to remove model_epoch: file not found")

            try:
                # without suffix: the selected model
                self.exp.visitor.remove()  # pylint: disable=E1101
            except FileNotFoundError:
                warnings.warn("failed to remove model")

            try:
                # for matchdg
                self.exp.visitor.remove("ctr")  # pylint: disable=E1101
            except FileNotFoundError:
                pass
