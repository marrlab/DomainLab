import abc
import os
import datetime
from pathlib import Path

import torch

from libdg.utils.get_git_tag import get_git_tag


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
        self.model_dir = os.path.join(self.out, ExpModelPersistVisitor.model_dir)
        self.git_tag = get_git_tag()
        self.task_name = self.host.task.get_na(self.host.args.tr_d, self.host.args.te_d)
        self.algo_name = self.host.args.aname
        self.seed = self.host.args.seed
        self.model_name = self.mk_model_na(self.git_tag)
        self.model_path = os.path.join(self.model_dir,
                                       self.model_name + ExpModelPersistVisitor.model_suffix)

        Path(os.path.dirname(self.model_path)).mkdir(parents=True, exist_ok=True)

    def mk_model_na(self, tag=None, dd_cut=19):
        """
        :param tag: for git commit hash for example
        """
        if tag is None:
            tag = "tag"
        suffix_t = str(datetime.datetime.now())[:dd_cut].replace(" ", "_")
        suffix_t = suffix_t.replace("-", "md_")
        suffix_t = suffix_t.replace(":", "_")
        list4mname = [self.task_name, self.algo_name, tag, suffix_t, "seed", str(self.seed)]
        # the sequence of components (e.g. seed in the last place) in model name is not crutial
        model_name = "_".join(list4mname)
        if self.host.args.debug:
            model_name = "debug_" + model_name
        print("model name:", model_name)
        return model_name

    def save(self, model, suffix=None):
        """
        :param model:
        """
        file_na = self.model_path
        if suffix is not None:
            file_na = "_".join([file_na, suffix])
        torch.save(model, file_na)

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
        """
        path = self.model_path
        if suffix is not None:
            path = "_".join([self.model_path, suffix])
        model = torch.load(path, map_location="cpu")
        return model


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
        dict_cols, *_ = self.get_cols()
        self.list_cols = list(dict_cols.keys())
        # FIXME: will be list be the same order each time?
        str_line = ", ".join(self.list_cols)
        if not os.path.isfile(self.get_fpath()):
            self.to_file(str_line)

    def __call__(self, acc):
        self.to_file(self._gen_line(acc))

    def get_cols(self):
        acc_name = "acc"
        epos_name = "epos"
        dict_cols = {acc_name: None,
                     "algo":self.algo_name,
                     epos_name:None,
                     "seed": self.seed,
                     "aggtag": self.agg_tag,   # algorithm configuration for instance
                     "mname": "mname_" + self.model_name,
                     "commit": "commit_" + self.git_tag}
        return dict_cols, acc_name, epos_name

    def _gen_line(self, acc):
        dict_cols, acc_name, epos_name = self.get_cols()
        dict_cols.update({acc_name: acc})
        dict_cols.update({epos_name: self.host.epoch_counter})  # FIXME: strong dependency on host attribute name
        list_str = [str(dict_cols[key]) for key in self.list_cols]
        str_line = ", ".join(list_str)
        return str_line

    def get_fpath(self, dirname="aggrsts"):
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
        print("results aggregation path:", file_path)
        with open(file_path, 'a') as f_h:
            print(str_line, file=f_h)
