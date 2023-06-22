import os

from torchvision import transforms

from domainlab.compos.pcr.request import RequestTask
from domainlab.tasks.task_folder_mk import mk_task_folder
from domainlab.tasks.task_mnist_color import NodeTaskMNISTColor10
from domainlab.tasks.utils_task import ImSize
from domainlab.utils.u_import import import_path

path_this_file = os.path.dirname(os.path.realpath(__file__))


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

        chain = mk_task_folder(extensions={"caltech": "jpg", "sun": "jpg", "labelme": "jpg"},
                               list_str_y=["chair", "car"],
                               dict_domain_folder_name2class={
                                   "caltech": {"auto": "car",
                                               "stuhl": "chair"},
                                   "sun": {"vehicle": "car",
                                           "sofa": "chair"},
                                   "labelme": {"drive": "car",
                                               "sit": "chair"}
                               },
                               dict_domain_img_trans={
                                   "caltech": transforms.Compose(
                                       [transforms.Resize((224, 224)),
                                        transforms.ToTensor()]),
                                   "sun": transforms.Compose(
                                       [transforms.Resize((224, 224)),
                                        transforms.ToTensor()]),
                                   "labelme": transforms.Compose(
                                       [transforms.Resize((224, 224)),
                                        transforms.ToTensor()]),
                               },
                               img_trans_te=transforms.Compose(
                                   [transforms.Resize((224, 224)),
                                    transforms.ToTensor()]),
                               isize=ImSize(3, 224, 224),
                               dict_domain2imgroot={
                                   "caltech": os.path.join(
                                       path_this_file,
                                       "../../data/vlcs_mini/caltech/"),
                                   "sun": os.path.join(
                                       path_this_file,
                                       "../../data/vlcs_mini/sun/"),
                                   "labelme": os.path.join(
                                       path_this_file,
                                       "../../data/vlcs_mini/labelme/")},
                               taskna="mini_vlcs",
                               succ=chain)

        if self.tpath is not None:
            node = self.task_module.get_task(self.args.task)
            chain.set_parent(node)
            chain = node
            if self.args.task is None:
                print("")
                print("overriding args.task ",
                      self.args.task, " to  ",
                      node.task_name)
                print("")
                self.request = node.task_name  # @FIXME
        node = chain.handle(self.request)
        return node
