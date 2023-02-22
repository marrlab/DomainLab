from domainlab.compos.pcr.p_chain_handler import AbstractChainNodeHandler


class NodeAlgoBuilder(AbstractChainNodeHandler):
    """
    Base class for Algorithm Builder
    """
    na_prefix = "NodeAlgoBuilder"

    def dset_decoration_args_algo(self, args, ddset):
        """
        most algorithms do not need re-organization of data feed flow like JiGen and MatchDG
        """
        print("processing dataset for ", args.aname)
        return ddset

    @property
    def name(self):
        """
        get the name of the algorithm
        """
        na_prefix = NodeAlgoBuilder.na_prefix
        len_prefix = len(na_prefix)
        na_class = type(self).__name__
        if na_class[:len_prefix] != na_prefix:
            raise RuntimeError(
                "algorithm builder node class must start with ", na_prefix,
                "the current class is named: ", na_class)
        return type(self).__name__[len_prefix:].lower()

    def is_myjob(self, request):
        """
        :param request: string
        """
        return request == self.name

    def init_business(self, exp):
        raise NotImplementedError
