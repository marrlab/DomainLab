import sys
import importlib.util
from domainlab.utils.u_import import import_path


def import_net_module_from_path(path_net_feat_extract):
    """
    :param path_net_feat_extract: path of the python file which contains the definition of the custom neural network
    """
    name_signature = "__init__(self, dim_y, i_c, i_h, i_w)"  # FIXME: hard coded
    name_custom_net = "NetFeatExtract"
    na_external_module = "name_external_module"   # the dummy module name
    spec = importlib.util.spec_from_file_location(
        name=na_external_module,
        location=path_net_feat_extract)
    module_external = importlib.util.module_from_spec(spec)
    sys.modules[na_external_module] = module_external
    # register the name of the external module
    spec.loader.exec_module(module_external)
    if not hasattr(module_external, name_custom_net):
        raise RuntimeError("the specified python file should contain the \
                           definition a neural network (as pytorch module) \
                           that can extract features from an image. The name \
                           of this pytorch module must be %s and the module \
                           must contain signature %s"
                           % (name_custom_net, name_signature))
    net = getattr(module_external, name_custom_net)
    return net
    # assert "dim_y" in str(inspect.signature(net.__init__))
    # assert "i_c" in str(inspect.signature(net.__init__))
    # assert "i_h" in str(inspect.signature(net.__init__))
    # assert "i_w" in str(inspect.signature(net.__init__))


def build_external_obj_net_module_feat_extract(mpath, dim_y,
                                               remove_last_layer):
    """ The user provide a function to initiate an object of the neural network,
    which is fine for training but problematic for persistence of the trained
    model since it is created externally.
    :param mpath: path of external python file where the neural network
    architecture is defined
    :param dim_y: dimension of features
    :param i_c: number of channels of image
    :param i_h: height of image
    :param i_w: width of image
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
    # FIXME: hard coded, move to top level __init__ definition in domainlab
    name_fun = name_signature[:name_signature.index("(")]
    if hasattr(net_module, name_fun):
        try:
            net = getattr(net_module, name_fun)(dim_y, remove_last_layer)
        except Exception:
            print("function %s should return a neural network (pytorch module) that \
                   that extract features from an image" % (name_signature))
            raise
        if net is None:
            raise RuntimeError("the pytorch module returned by %s is None"
                               % (name_signature))
    else:
        raise RuntimeError("Please implement a function %s \
                            in your external python file"
                           % (name_signature))
    return net
