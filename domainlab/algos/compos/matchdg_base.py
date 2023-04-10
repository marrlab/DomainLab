import torch
from domainlab.algos.compos.matchdg_match import MatchPair


class MatchAlgoBase():
    def __init__(self, task, phi, args, device, exp, opt):
        self.bs_match = args.bs  # use the same batch size for match tensor
        self.exp = exp
        self.task = task
        self.num_domain_tr = len(self.task.list_domain_tr)
        train_domains = self.task.list_domain_tr
        self.list_tr_domain_size = [len(self.task.dict_dset[key]) \
            for key in train_domains]
        # so that order is kept!
        self.base_domain_size = get_base_domain_size4match_dg(self.task)
        self.dim_y = task.dim_y

        self.args = args
        # @FIXME: training loader always drop the last batch,
        # so inside matchdg, for the data storage tensor,
        # loader is re-initialized by disabling drop
        self.loader = task.loader_tr
        self.device = device
        self.phi = phi.to(self.device)
        #
        self.opt = opt
        self.ctr_mpath = self.exp.visitor.model_path + "_ctr"
        #
        self.tensor_ref_domain2each_domain_x = None
        self.tensor_ref_domain2each_domain_y = None

    def init_erm_phase(self):
        """
        loade model from disk after training
        the ctr(contrastive learning) phase
        """
        # the keys of :attr:`state_dict` must exactly match the
        # keys returned by this module's
        # :meth:`~torch.nn.Module.state_dict` function
        self.phi.load_state_dict(torch.load(self.ctr_mpath), strict=False)
        # load the phi network trained during the
        # ctr(contrastive learning) phase
        self.phi = self.phi.to(self.device)
        # len((ctr_phi.state_dict()).keys()): 122,
        # extra fields are fc.weight, fc.bias
        self.phi.eval()  # @FIXME
        self.mk_match_tensor(epoch=0)

    def save_model_ctr_phase(self):
        # Store the weights of the model
        # dirname = os.path.dirname(self.ctr_mpath)
        # Path(dirname).mkdir(parents=True, exist_ok=True)
        torch.save(self.phi.state_dict(), self.ctr_mpath)

    def save_model_erm_phase(self):
        torch.save(self.phi, self.exp.visitor.model_path)

    def mk_match_tensor(self, epoch):
        """
        initialize or update match tensor
        """
        obj_match = MatchPair(self.dim_y,
                              self.task.isize.i_c,
                              self.task.isize.i_h,
                              self.task.isize.i_w,
                              self.bs_match,
                              virtual_ref_dset_size=self.base_domain_size,
                              num_domains_tr=self.num_domain_tr,
                              list_tr_domain_size=self.list_tr_domain_size)

        # @FIXME: what is the usefulness of (epoch > 0) as argument
        self.tensor_ref_domain2each_domain_x, self.tensor_ref_domain2each_domain_y = \
        obj_match(
            self.device,
            self.loader,
            lambda x: self.phi.extract_semantic_feat(x),
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
        print("for class ", mclass, " bigest sample size is ",
              num, "ref domain is", ref_domain)
        base_domain_size += num
    return base_domain_size
