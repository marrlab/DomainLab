"""
This module contains 3 classes inheriting:
    ExpProtocolAggWriter(AggWriter(ExpModelPersistVisitor))
"""
import copy
import datetime
import os
import numpy as np
from pathlib import Path

import torch
from sklearn.metrics import ConfusionMatrixDisplay

from domainlab.utils.get_git_tag import get_git_tag
from domainlab.utils.logger import Logger


class ExpModelPersistVisitor():
    """
    This class couples with Task class attributes
    """
    model_dir = "saved_models"
    model_suffix = ".model"

    def __init__(self, host):
        """
        1. create new attributes like model names
        2. all dependencies in contructor
        3. do not change the sequence of the lines!
        since later lines depends on earlier definitions
        """
        self.host = host
        self.out = host.args.out
        self.model_dir = os.path.join(self.out,
                                      ExpModelPersistVisitor.model_dir)
        self.git_tag = get_git_tag()
        self.task_name = self.host.task.get_na(self.host.args.tr_d,
                                               self.host.args.te_d)
        self.algo_name = self.host.args.aname
        self.seed = self.host.args.seed
        self.model_name = self.mk_model_na(self.git_tag)
        self.model_path = os.path.join(self.model_dir,
                                       self.model_name +
                                       ExpModelPersistVisitor.model_suffix)

        Path(os.path.dirname(self.model_path)).mkdir(parents=True, exist_ok=True)
        self.model = copy.deepcopy(self.host.trainer.model)
        # although deepcopy in contructor is expensive, but
        # execute copy.deepcopy(self.host.trainer.model) after training will cause thread lock
        # if self.host.trainer has tensorboard writer, see
        # https://github.com/marrlab/DomainLab/issues/673

    def mk_model_na(self, tag=None, dd_cut=19):
        """
        :param tag: for git commit hash for example
        """
        if tag is None:
            tag = "tag"
        suffix_t = str(datetime.datetime.now())[:dd_cut].replace(" ", "_")
        suffix_t = suffix_t.replace("-", "md_")
        suffix_t = suffix_t.replace(":", "_")
        list4mname = [self.task_name,
                      self.algo_name,
                      tag, suffix_t,
                      "seed",
                      str(self.seed)]
        # the sequence of components (e.g. seed in the last place)
        # in model name is not crutial
        model_name = "_".join(list4mname)
        if self.host.args.debug:
            model_name = "debug_" + model_name
        slurm = os.environ.get('SLURM_JOB_ID')
        if slurm:
            model_name = model_name + '_' + slurm
        logger = Logger.get_logger()
        logger.info(f"model name: {model_name}")
        return model_name

    def save(self, model, suffix=None):
        """
        :param model:
        """
        file_na = self.model_path
        if suffix is not None:
            file_na = "_".join([file_na, suffix])
        torch.save(copy.deepcopy(model.state_dict()), file_na)
        # checkpoint = {'model': Net(), '
        # state_dict': model.state_dict(),
        # 'optimizer' :optimizer.state_dict()}
        # torch.save(checkpoint, 'Checkpoint.pth')

    def remove(self, suffix=None):
        """
        remove model after use
        """
        file_na = self.model_path
        if suffix is not None:
            file_na = "_".join([file_na, suffix])
        os.remove(file_na)

    def load(self, suffix=None):
        """
        load pre-defined model name from disk
        the save function is the same class so to ensure load will ways work
        """
        path = self.model_path
        if suffix is not None:
            path = "_".join([self.model_path, suffix])
        # due to tensorboard writer in trainer.scheduler,
        # copy.deepcopy(self.host.trainer.model) will cause thread lock
        # see https://github.com/marrlab/DomainLab/issues/673
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        # without separate self.model and self.model_suffixed,
        # it will cause accuracy inconsistent problems since the content of self.model
        # can be overwritten when the current function is called another time and self.model
        # is not deepcopied
        # However, deepcopy is also problematic when it is executed too many times
        return copy.deepcopy(self.model)
        # instead of deepcopy, one could also have multiple copies of model in constructor, but this
        # does not adhere the lazy principle.

    def clean_up(self):
        self.host.clean_up()


class AggWriter(ExpModelPersistVisitor):
    """
    1. aggregate results to text file.
    2. all dependencies are in the constructor
    """
    def __init__(self, host):
        super().__init__(host)
        self.agg_tag = self.host.args.aggtag
        self.exp_tag = self.host.args.exptag
        self.debug = self.host.args.debug
        self.has_first_line = False
        self.list_cols = None

    def first_line(self, dict_cols):
        """
        generate header of the results aggregation file
        """
        self.list_cols = list(dict_cols.keys())
        # @FIXME: will be list be the same order each time?
        str_line = ", ".join(self.list_cols)
        if not os.path.isfile(self.get_fpath()):
            self.to_file(str_line)
        self.has_first_line = True

    def __call__(self, dict_metric):
        line, confmat, confmat_filename = self._gen_line(dict_metric)
        self.to_file(line)
        if not self.host.args.no_dump:
            self.confmat_to_file(confmat, confmat_filename)

    def get_cols(self):
        """
        call the same function to always get the same columns configuration
        """
        epos_name = "epos"
        dict_cols = {
                     "algo": self.algo_name,
                     epos_name: None,
                     "seed": self.seed,
                     "aggtag": self.agg_tag,
                     # algorithm configuration for instance
                     "mname": "mname_" + self.model_name,
                     "commit": "commit_" + self.git_tag}
        return dict_cols, epos_name

    def _gen_line(self, dict_metric):
        dict_cols, epos_name = self.get_cols()
        dict_cols.update(dict_metric)
        confmat = dict_cols.pop("confmat")
        confmat_filename = dict_cols.get("mname", None)  # return None if not found
        # @FIXME: strong dependency on host attribute name
        dict_cols.update({epos_name: self.host.epoch_counter})
        if not self.has_first_line:
            self.first_line(dict_cols)
        list_str = [str(dict_cols[key]) for key in self.list_cols]
        str_line = ", ".join(list_str)
        return str_line, confmat, confmat_filename

    def get_fpath(self, dirname="aggrsts"):
        """
        for writing and reading, the same function is called to ensure name
        change in the future will not break the software
        """
        list4fname = [self.task_name,
                      self.exp_tag,
                      ]
        fname = "_".join(list4fname) + ".csv"
        if self.debug:
            fname = "_".join(["debug_agg", fname])
        file_path = os.path.join(self.out, dirname, fname)
        return file_path

    def to_file(self, str_line):
        """
        :param str_line:
        """
        file_path = self.get_fpath()
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        logger = Logger.get_logger()
        logger.info(f"results aggregation path: {file_path}")
        with open(file_path, 'a', encoding="utf8") as f_h:
            print(str_line, file=f_h)

    def confmat_to_file(self, confmat, confmat_filename):
        """Save confusion matrix as a figure

        Args:
            confmat: confusion matrix.
        """
        disp = ConfusionMatrixDisplay(confmat)
        disp = disp.plot(cmap="gray")
        file_path = self.get_fpath()
        # @FIXME: although removesuffix is safe when suffix does not exist,
        # we would like to have ".csv" live in some configuraiton file in the future.
        file_path = file_path.removesuffix(".csv")
        # if prefix does not exist, string remain unchanged.
        # @FIXME: still we want to have mname_ as a variable defined in some
        # configuration file in the future.
        confmat_filename = confmat_filename.removeprefix("mname_")
        file_path = os.path.join(os.path.dirname(file_path), f"{confmat_filename}_conf_mat.png")
        logger = Logger.get_logger()
        logger.info(f"confusion matrix saved in file: {file_path}")
        disp.figure_.savefig(file_path)


class ExpProtocolAggWriter(AggWriter):
    """
    AggWriter tailored to experimental protocol
    Output contains additionally index, exp task, te_d and params.
    """
    def get_cols(self):
        """columns"""
        epos_name = "epos"
        dict_cols = {
            "param_index": self.host.args.param_index,
            "method": self.host.args.benchmark_task_name,
            "mname": "mname_" + self.model_name,
            "commit": "commit_" + self.git_tag,
            "algo": self.algo_name,
            epos_name: None,
            "te_d": self.host.args.te_d,
            "seed": self.seed,
            "params": f"\"{self.host.args.params}\"",
        }
        return dict_cols, epos_name

    def get_fpath(self, dirname=None):
        """filepath"""
        if dirname is None:
            return self.host.args.result_file
        return super().get_fpath(dirname)

    def confmat_to_file(self, confmat, confmat_filename):
        """Save confusion matrix as a figure

        Args:
            confmat: confusion matrix.
        """
        path4file = self.get_fpath()
        index = os.path.basename(path4file)
        path4file = os.path.dirname(os.path.dirname(path4file))
        # if prefix does not exist, string remain unchanged.
        confmat_filename = confmat_filename.removeprefix("mname_")
        path4file = os.path.join(path4file, "confusion_matrix")
        os.makedirs(path4file, exist_ok=True)
        file_path = os.path.join(path4file,
                                 f"{index}.txt")
        with open(file_path, 'a', encoding="utf8") as f_h:
            print(confmat_filename, file=f_h)
            for line in np.matrix(confmat):
                np.savetxt(f_h, line, fmt='%.2f')
        logger = Logger.get_logger()
        logger.info(f"confusion matrix saved in file: {file_path}")
