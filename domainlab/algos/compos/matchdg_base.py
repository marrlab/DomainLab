"""
interface class for matchdg by defining auxilliary operations
"""

import torch
from domainlab.algos.compos.matchdg_match import MatchPair
from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.utils.logger import Logger


class MatchAlgoBase(AbstractTrainer):
    """
    interface class for matchdg by defining auxilliary operations
    """
    def before_tr(self):
        """
        override abstract method
        """

    @property
    def model_path_ctr(self):
        """
        property: model storage path for ctr phase model
        """
        if self.exp is not None:
            return self.exp.visitor.model_path + "_ctr"
        # import uuid
        # filename = str(uuid.uuid4())
        # import time
        # timestr = time.strftime("%Y%m%d_%H%M%S")
        # print timestr
        return "model_ctr"

    def init_erm_phase(self):
        """
        loade model from disk after training
        the ctr(contrastive learning) phase
        """
        # the keys of :attr:`state_dict` must exactly match the
        # keys returned by this module's
        # :meth:`~torch.nn.Module.state_dict` function
        self.model.load_state_dict(
            torch.load(self.model_path_ctr), strict=False)
        # load the model network trained during the
        # ctr(contrastive learning) phase
        self.model = self.model.to(self.device)
        # len((ctr_phi.state_dict()).keys()): 122,
        # extra fields are fc.weight, fc.bias
        self.model.eval()  # @FIXME
        self.mk_match_tensor(epoch=0)

    def save_model_ctr_phase(self):
        # Store the weights of the model
        # dirname = os.path.dirname(self.ctr_mpath)
        # Path(dirname).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path_ctr)

    def mk_match_tensor(self, epoch):
        """
        initialize or update match tensor
        """
        obj_match = MatchPair(self.task.dim_y,
                              self.task.isize.i_c,
                              self.task.isize.i_h,
                              self.task.isize.i_w,
                              self.aconf.bs,
                              virtual_ref_dset_size=self.base_domain_size,
                              num_domains_tr=len(self.task.list_domain_tr),
                              list_tr_domain_size=self.list_tr_domain_size)

        # @FIXME: what is the usefulness of (epoch > 0) as argument
        self.tensor_ref_domain2each_domain_x, self.tensor_ref_domain2each_domain_y = \
        obj_match(
            self.device,
            self.task.loader_tr,
            self.model.extract_semantic_feat,
            (epoch > 0))


def get_base_domain_size4match_dg(task):
    """
    Base domain is a dataset where each class
    set come from one of the nominal domains
    """
    # @FIXME: base domain should be calculated only on training domains
    # instead of all the domains!
    # domain_keys = task.get_list_domains()
    domain_keys = task.list_domain_tr
    base_domain_size = 0
    classes = task.list_str_y
    for mclass in classes:
        num = 0
        ref_domain = -1
        for _, domain_key in enumerate(domain_keys):
            if task.dict_domain_class_count[domain_key][mclass] > num:
                ref_domain = domain_key
                num = task.dict_domain_class_count[domain_key][mclass]
        logger = Logger.get_logger()
        logger.info(f"for class {mclass} bigest sample size is {num} "
                    f"ref domain is {ref_domain}")
        base_domain_size += num
    return base_domain_size
