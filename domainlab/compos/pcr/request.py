from domainlab.utils.utils_class import store_args


class RequestVAEBuilderCHW():
    @store_args
    def __init__(self, i_c, i_h, i_w, args):
        pass

class RequestVAEBuilderNN():
    """creates request when input does not come from command-line (args) but from test_exp file"""
    @store_args
    def __init__(self, net_class_d, net_x, net_class_y, i_c, i_h, i_w):
        """net_class_d, net_x and net_class_y are neural networks defined by the user"""

class RequestTask():
    """
    Isolate args from Request object of chain of responsibility node for task
    """
    def __init__(self, args):
        self.args = args

    def __call__(self):
        return self.args.task


class RequestArgs2ExpCmd():
    """
    Isolate args from Request object of chain of responsibility node for experiment
    For example, args has field names which will couple with experiment class, this
    request class also serves as isolation class or adaptation class
    """
    @store_args
    def __init__(self, args):
        self.args = args

    def __call__(self):
        return self.args.aname
