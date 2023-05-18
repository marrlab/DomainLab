"""
import external neural network implementation
"""

from domainlab.utils.u_import import import_path
from domainlab.utils.logger import Logger


def build_external_obj_net_module_feat_extract(mpath, dim_y,
                                               remove_last_layer):
    """ The user provide a function to initiate an object of the neural network,
    which is fine for training but problematic for persistence of the trained
    model since it is created externally.
    :param mpath: path of external python file where the neural network
    architecture is defined
    :param dim_y: dimension of features
    """
    # other possibility
    # na_external_module = "name_external_module"   # the dummy module name
    # spec = importlib.util.spec_from_file_location(
    #    name=na_external_module,
    #    location=path_net_feat_extract)
    # module_external = importlib.util.module_from_spec(spec)
    # sys.modules[na_external_module] = module_external
    # register the name of the external module
    # spec.loader.exec_module(module_external)

    net_module = import_path(mpath)
    name_signature = "build_feat_extract_net(dim_y, \
        remove_last_layer)"
    # @FIXME: hard coded, move to top level __init__ definition in domainlab
    name_fun = name_signature[:name_signature.index("(")]
    if hasattr(net_module, name_fun):
        try:
            net = getattr(net_module, name_fun)(dim_y, remove_last_layer)
        except Exception:
            logger = Logger.get_logger()
            logger.error(f"function {name_signature} should return a neural network "
                         f"(pytorch module) that that extract features from an image")
            raise
        if net is None:
            raise RuntimeError("the pytorch module returned by %s is None"
                               % (name_signature))
    else:
        raise RuntimeError("Please implement a function %s \
                            in your external python file"
                           % (name_signature))
    return net
