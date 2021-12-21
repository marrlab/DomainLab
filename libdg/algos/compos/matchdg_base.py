import torch
from torch import optim

from libdg.algos.compos.matchdg_match import MatchPair


class MatchAlgoBase():
    def __init__(self, task, phi, args, device, exp):
        self.bs_match = args.bs  # use the same batch size for match tensor
        self.exp = exp
        self.task = task
        self.num_domain_tr = len(self.task.list_domain_tr)
        train_domains = self.task.list_domain_tr
        self.list_tr_domain_size = [len(self.task.dict_dset[key]) for key in train_domains]
        # so that order is kept!
        self.base_domain_size = get_base_domain_size4match_dg(self.task)
        self.dim_y = task.dim_y

        self.args = args
        self.loader = task.loader_tr
        self.device = device
        self.phi = phi.to(self.device)
        #
        self.opt = self.get_opt_sgd()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25)
        self.ctr_mpath = self.exp.visitor.model_path + "_ctr"
        #

        self.tensor_ref_domain2each_domain_x = None
        self.tensor_ref_domain2each_domain_y = None

    def init_erm_phase(self):
        """
        loade model from disk after training the ctr(contrastive learning) phase
        """
        # the keys of :attr:`state_dict` must exactly match the keys returned by this module's :meth:`~torch.nn.Module.state_dict` function
        self.phi.load_state_dict(torch.load(self.ctr_mpath), strict=False)
        # load the phi network trained during the ctr(contrastive learning) phase
        self.phi = self.phi.to(self.device)
        # len((ctr_phi.state_dict()).keys()): 122, extra fields are fc.weight, fc.bias
        self.phi.eval()  #  FIXME
        self.mk_match_tensor(epoch=0)

    def get_opt_sgd(self):
        opt = optim.SGD([{'params': filter(lambda p: p.requires_grad, self.phi.parameters())}, ],
                        lr=self.args.lr, weight_decay=5e-4,
                        momentum=0.9, nesterov=True)
        return opt

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
        obj_match = MatchPair(self.dim_y, self.task.isize.i_c, self.task.isize.i_h,
                              self.task.isize.i_w,
                              self.bs_match,
                              virtual_ref_dset_size=self.base_domain_size,
                              num_domains_tr=self.num_domain_tr,
                              list_tr_domain_size=self.list_tr_domain_size)

        self.tensor_ref_domain2each_domain_x, self.tensor_ref_domain2each_domain_y = \
            obj_match(self.device, self.loader, self.phi, (epoch > 0))


def get_base_domain_size4match_dg(task):
    """
    Base domain is a dataset where each class set come from one of the nominal domains
    """
    domain_keys = task.get_list_domains()
    base_domain_size = 0
    classes = task.list_str_y
    for mclass in classes:
        num = 0
        for domain_key in domain_keys:
            if task.dict_domain_class_count[domain_key][mclass] > num:
                num = task.dict_domain_class_count[domain_key][mclass]
        print("for class ", mclass, " bigest sample size is ", num)
        base_domain_size += num
    return base_domain_size
