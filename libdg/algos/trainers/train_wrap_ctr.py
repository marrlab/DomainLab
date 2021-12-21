"""
Trainer for TopicVAE
"""
import torch
from train.train_interface import AbstractTrainer
from train.train_match_dg import TrainerMatchDG

def get_wrap_ctr_trainer(base):
    class TrainerWrapCTR(base):
        """"""

        def __init__(self, model, perf, scenario, device, conf):
            """__init__.

            :param model:
            :param perf:
            :param device:
            :param conf:
            """
            self.device = None
            self.model = None
            self.ctr_model = None
            super().__init__(model=model, perf=perf, scenario=scenario,
                             conf=conf, device=device)

        def probe_scenario(self, scenario):
            """probe_scenario. configure trainer accoding to properties of
            scenario as well according to algorithm configurations
            :param scenario:
            """
            from experiment.fac_deepall import FactoryMatchDG
            fun_create_ctr, fun_build_erm = \
                FactoryMatchDG.get_ctr_model_erm_creator(scenario)
            self.ctr_model = fun_create_ctr()
            args, base_domain_size, total_domains, training_list_size, \
                train_domains = TrainerMatchDG.get_conf_from_scenario(scenario)
            path = TrainerMatchDG.init_ctr(
                scenario, self.ctr_model, self.device,
                fun_build_erm, args,
                base_domain_size,
                total_domains,
                training_list_size,
                train_domains)
            self.model.init_classif_part(path)
    return TrainerWrapCTR
