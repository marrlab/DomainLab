import os
from importlib.machinery import SourceFileLoader


def import_path(path):
    path = os.path.expanduser(path)
    if path[0] == "/":
        full_path = path
    else:
        dir_path = os.getcwd()
        full_path = os.path.join(dir_path, path)
    return SourceFileLoader(fullname=full_path, path=full_path).load_module()
