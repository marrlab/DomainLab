"""
all available tasks for domainlab
"""

from domainlab.arg_parser import mk_parser_main
from domainlab.compos.pcr.request import RequestTask
from domainlab.tasks.task_mnist_color import NodeTaskMNISTColor10
from domainlab.utils.u_import import import_path
from domainlab.utils.logger import Logger
from domainlab.tasks.task_mini_vlcs import addtask2chain


class TaskChainNodeGetter(object):
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, args):
        self.args = args
        tpath = args.tpath
        self.tpath = tpath
        self.request = RequestTask(args)()
        if tpath is not None:
            self.task_module = import_path(self.tpath)

    def __call__(self):
        """
        1. construct the chain, filter out responsible node, create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = NodeTaskMNISTColor10(None)
        chain = addtask2chain(chain)

        if self.tpath is not None:
            node = self.task_module.get_task(self.args.task)
            chain.set_parent(node)
            chain = node
            if self.args.task is None:
                logger = Logger.get_logger()
                logger.info("")
                logger.info(f"overriding args.task {self.args.task} "
                            f"to {node.task_name}")
                logger.info("")
                self.request = node.task_name  # @FIXME
        node = chain.handle(self.request)
        return node


def get_task(name=None):
    """
    get build in task from DomainLab
    """
    args = mk_parser_main
    parser = mk_parser_main()
    args = parser.parse_args("")
    args.task = name
    task_getter = TaskChainNodeGetter(args)
    return task_getter()
