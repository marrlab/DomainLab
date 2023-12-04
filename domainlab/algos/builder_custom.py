from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter
from domainlab.utils.utils_cuda import get_device


def make_basic_trainer(class_name_model):
    """make_basic_trainer.
    :param class_name_model:
    """
    class NodeAlgoBuilderCustom(NodeAlgoBuilder):
        """NodeAlgoBuilderCustom."""

        def get_trainer(self, args):
            """
            chain of responsibility pattern for fetching trainer from commandline parsed arguments
            """
            trainer = TrainerChainNodeGetter(args.trainer)(default="basic")
            return trainer

        def _set_args(self, args, val_arg_na, prefix, argname):
            """_set_args.
            Since we could not hard code all possible strings in the argparser,
            we add attributes to args name space according to the user
            specification from the custom python file they provide

            the custom model the user wrote should have
            model.dict_net_module_na2arg_na with something like
            {"net1":"name1", "net2":"name2"}
            val_arg_na below will be filled with "name1" for instance

            python main_out.py –te_d=caltech –task=mini_vlcs –debug –bs=3
            –apath=examples/algos/demo_custom_model.py
            –aname=custom –nname_argna2val net1 –nname_argna2val alexnet

            :param args: the namespace of command line arguemnts
            :param val_arg_na: the custom argument name the user specified
            :param prefix: nname or npath to be consistent with the rest of
            the package
            :param argname: nname_argna2val or "npath_argna2val", hard coded
            """
            if getattr(args, argname) is None:
                setattr(args, prefix+val_arg_na, None)
                return
            list_args = getattr(args, argname)
            ind = list_args.index(val_arg_na)
            if ind+1 >= len(list_args):  # list of args always even length
                raise RuntimeError("\n nname_argna2val or npath_argna2val should \
                                   \n always be specified in pairs instead of \
                                   odd number:\
                                   \n %s" % (
                                       str(list_args)))
            val = list_args[ind+1]
            # add attributes to namespaces args, the attributes are provided by
            # user in the custom model file
            setattr(args, prefix+val_arg_na, val)

        def set_nets_from_dictionary(self, args, task, model):
            """set_nets_from_dictionary.
            the custom model the user wrote should have
            model.dict_net_module_na2arg_na with something like
            {"net1":"name1", "net2":"name2"}
            python main_out.py –te_d=caltech –task=mini_vlcs –debug –bs=3
            –apath=examples/algos/demo_custom_model.py
            –aname=custom –nname_argna2val net1 –nname_argna2val alexnet
            """
            for key_module_na, val_arg_na in \
                    model.dict_net_module_na2arg_na.items():
                #
                if args.nname_argna2val is None and \
                        args.npath_argna2val is None:
                    raise RuntimeError("either specify nname_argna2val or \
                                        npath_argna2val")
                self._set_args(args, val_arg_na, "nname", "nname_argna2val")
                self._set_args(args, val_arg_na, "npath", "npath_argna2val")
                #
                builder = FeatExtractNNBuilderChainNodeGetter(
                    args, arg_name_of_net="nname"+val_arg_na,
                    arg_path_of_net="npath"+val_arg_na)()
                net = builder.init_business(
                    flag_pretrain=True, dim_out=task.dim_y,
                    remove_last_layer=False, args=args,
                    i_c=task.isize.i_c, i_h=task.isize.i_h, i_w=task.isize.i_w)
                model.add_module("%s" % (key_module_na), net)

        def init_business(self, exp):
            """
            return trainer, model, observer
            """
            task = exp.task
            args = exp.args
            device = get_device(args)
            model_sel = MSelOracleVisitor(MSelValPerf(max_es=args.es))
            observer = ObVisitor(model_sel)
            model = class_name_model(list_str_y=task.list_str_y)
            self.set_nets_from_dictionary(args, task, model)
            trainer = self.get_trainer(args)
            trainer.init_business(model, task, observer, device, args)
            return trainer, model, observer, device
    return NodeAlgoBuilderCustom
