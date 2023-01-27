from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.utils.flows_gen_img_model import fun_gen


class ObVisitorGen(ObVisitor):
    """
    For Generative Models
    """
    def after_all(self):
        super().after_all()
        print("generating images for final model at last epoch")
        fun_gen(subfolder_na=self.exp.visitor.model_name+"final",
                args=self.exp.args, node=self.exp.task, model=self.host_trainer.model, device=self.device)

        print("generating images for oracle model")
        model_or = self.exp.visitor.load("oracle")  # @FIXME: name "oracle is a strong dependency
        model_or = model_or.to(self.device)
        model_or.eval()
        fun_gen(subfolder_na=self.exp.visitor.model_name+"oracle",
                args=self.exp.args, node=self.exp.task, model=model_or, device=self.device)
        print("generating images for selected model")
        model_ld = self.exp.visitor.load()
        model_ld = model_ld.to(self.device)
        model_ld.eval()
        fun_gen(subfolder_na=self.exp.visitor.model_name+"selected",
                args=self.exp.args, node=self.exp.task, model=model_ld, device=self.device)
