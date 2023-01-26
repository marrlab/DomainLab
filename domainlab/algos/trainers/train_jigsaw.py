import torch.optim as optim
from torch import nn
from train.optimizer_helper import get_optim_and_scheduler
# from utils.Logger import Logger
from train.train_interface import AbstractTrainer

class TrainerJigsaw(AbstractTrainer):
    def __init__(self, model, perf, conf, device):
        super().__init__(model, perf, conf, device)
        # print(self.model)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=conf.lr)
        self.jig_weight = 0.9  # FIXME: previously it was 0.1

    def train_one_epoch(self, scenario, epoch):
        criterion = nn.CrossEntropyLoss()
        train_loader = scenario.tr_loader
        self.model.train()
        itr_per_epoch = len(train_loader)
        print("itr_per_epoch:", itr_per_epoch)
        running_loss = 0.0
        #for it, ((data, jig_l, class_l), d_idx) in enumerate(source_loader):
        # jig_l is the ground truth permutation index, class_l is the ground truth class label index
        # NOTE: class_l must be before jig_l to ensure other function like performance.get_accuracy works!
        for it, (data, class_l, jig_l) in enumerate(train_loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            #data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            # absolute_iter_count = it + self.current_epoch * self.len_dataloader
            # p = float(absolute_iter_count) / self.args.epochs / self.len_dataloader
            # lambda_val = 2. / (1. + np.exp(-10 * p)) - 1
            # if domain_error > 2.0:
            #     lambda_val  = 0
            # print("Shutting down LAMBDA to prevent implosion")

            self.optimizer.zero_grad()

            jigsaw_logit, class_logit = self.model(data)  # , lambda_val=lambda_val)
            jigsaw_loss = criterion(jigsaw_logit, jig_l)  # jig_l is in range 0 to 100, in total 101 values
            _, c_target = class_l.max(dim=1)   # one hot to index
            class_loss = criterion(class_logit, c_target)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            loss = class_loss + jigsaw_loss * self.jig_weight  # + 0.1 * domain_loss
            loss.backward()
            self.optimizer.step()
            running_loss += class_loss.detach().item()
            #self.logger.log(it, len(self.source_loader),
            #                {"jigsaw": jigsaw_loss.item(), "class": class_loss.item()  # , "domain": domain_loss.item()
            #                 },
            #                # ,"lambda": lambda_val},
            #                {"jigsaw": torch.sum(jig_pred == jig_l.data).item(),
            #                 "class": torch.sum(cls_pred == class_l.data).item(),
            #                 # "domain": torch.sum(domain_pred == d_idx.data).item()
            #                 },
            #                data.shape[0])
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit
        flag_stop = self.perf.after_epoch(running_loss, running_loss, epoch)
        return flag_stop


class TrainerJigsawCalmDown(TrainerJigsaw):
    def __init__(self, model, perf, conf, device):
        super().__init__(model, perf, conf, device)
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, conf.epochs, conf.lr, nesterov=True)   # overwrite self.optimizer

    def train_one_epoch(self, scenario, epoch):
        self.scheduler.step()
        print("learning rate:", self.scheduler.get_lr())
        super().train_one_epoch(scenario, epoch)
