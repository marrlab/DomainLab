"""
When class names and numbers does not match across different domains
"""
from domainlab.tasks.task_folder import NodeTaskFolderClassNaMismatch


def mk_task_folder(extensions,
                   list_str_y,
                   dict_domain_folder_name2class,
                   dict_domain_img_trans,
                   img_trans_te,
                   isize,
                   dict_domain2imgroot,
                   taskna,
                   succ=None):
    """
    Make task by specifying each domain with folder structures
    :param extensions: Different Options: 1. a python dictionary with key as the domain name
    and value (str or tuple[str]) as the file extensions of the image. 2. a str or tuple[str]
    with file extensions for all domains. 3. None: in each domain all files with an extension
    in ('jpg', 'jpeg', 'png') are loaded.
    :param 
    list_str_y: a python list with user defined class name where
    the order of the list matters.
    :param dict_domain_folder_name2class: a python dictionary, with key
    as the user specified domain name, value as a dictionary to map the
    sub-folder name of each domain's class folder into the user defined
    common class name.
    :param dict_domain_img_trans: a python dictionary with keys as the user
    specified domain name, value as a user defined torchvision transform.
    This feature allows carrying out different transformation (composition) to different
    domains at training time.
    :param img_trans_te: at test or inference time, we do not have knowledge
    of domain information so only a unique transform (composition) is allowed.
    :isize: domainlab.tasks.ImSize(image channel, image height, image width)
    :dict_domain2imgroot: a python dictionary with keys as user specified domain
    names and values as the absolute path to each domain's data.
    :taskna: user defined task name
    """
    class NodeTaskFolderDummy(NodeTaskFolderClassNaMismatch):
        @property
        def task_name(self):
            """
            The basic name of the task, without configurations
            """
            return taskna

        def conf_without_args(self):
            self.extensions = extensions
            self.list_str_y = list_str_y
            self._dict_domain_folder_name2class = dict_domain_folder_name2class
            self.isize = isize
            self.set_list_domains(list(self._dict_domain_folder_name2class.keys()))
            self.dict_domain2imgroot = dict_domain2imgroot
            self._dict_domain_img_trans = dict_domain_img_trans
            self.img_trans_te = img_trans_te
    return NodeTaskFolderDummy(succ=succ)
