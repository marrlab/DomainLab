"""
Color MNIST with palette
"""
from torch.utils.data.dataset import ConcatDataset
from libdg.tasks.b_task import NodeTaskDict
from libdg.tasks.utils_task import DsetDomainVecDecorator, mk_onehot, mk_loader, ImSize
from libdg.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from libdg.dsets.utils_color_palette import default_rgb_palette   # FIXME
from libdg.utils.utils_classif import mk_dummy_label_list_str


class NodeTaskMNISTColor4(NodeTaskDict):
    """
    Use the first 4 colors of the deafult palette
    """
    @property
    def list_str_y(self):
        return mk_dummy_label_list_str("digit", 10)

    @property
    def isize(self):
        """image channel, height, width"""
        return ImSize(3, 28, 28)

    def get_list_domains(self):
        """
        1. get list of domain names
        2. better use method than property so new domains can be added
        """
        list_domains = []
        for rgb_list in default_rgb_palette[:4]:   # FIXME:
            domain = "_".join([str(c) for c in rgb_list])
            domain = "rgb_" + domain
            list_domains.append(domain)
        return list_domains

    def get_dset_by_domain(self, args, na_domain):
        ind_global = self.get_list_domains().index(na_domain)
        dset = DsetMNISTColorSoloDefault(ind_global, args.dpath)
        return dset

def test_fun():
    from libdg.utils.arg_parser import mk_parser_main
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "1", "--dpath", "zout"])
    node = NodeTaskMNISTColor4()
    node.get_list_domains()
    node.list_str_y
    node.init_business(args)
