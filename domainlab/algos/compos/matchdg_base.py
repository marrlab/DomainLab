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
