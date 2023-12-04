from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.utils.flows_gen_img_model import fun_gen
from domainlab.utils.logger import Logger


class ObVisitorGen(ObVisitor):
    """
    For Generative Models
    """
    def after_all(self):
        super().after_all()
        logger = Logger.get_logger()
        logger.info("generating images for final model at last epoch")
        fun_gen(subfolder_na=self.host_trainer.model.visitor.model_name+"final",
                args=self.host_trainer.aconf, node=self.host_trainer.task, model=self.host_trainer.model,
                device=self.device)

        logger.info("generating images for oracle model")
        model_or = self.host_trainer.model.load("oracle")  # @FIXME: name "oracle is a strong dependency
        model_or = model_or.to(self.device)
        model_or.eval()
        fun_gen(subfolder_na=self.host_trainer.model.visitor.model_name+"oracle",
                args=self.host_trainer.aconf, node=self.host_trainer.task, model=model_or, device=self.device)
        logger.info("generating images for selected model")
        model_ld = self.host_trainer.model.load()
        model_ld = model_ld.to(self.device)
        model_ld.eval()
        fun_gen(subfolder_na=self.host_trainer.model.visitor.model_name+"selected",
                args=self.host_trainer.aconf, node=self.host_trainer.task, model=model_ld, device=self.device)
