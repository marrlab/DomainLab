"""
request for chain of responsibility pattern builder
"""
from domainlab.utils.utils_class import store_args


class ConfVAEBuilder():
    def __init__(self, isize, args):
        self.i_c = isize[0]
        self.i_h = isize[1]
        self.i_w = isize[2]
        self.args = args


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
