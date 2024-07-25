"""
simple exponential moving average of each layers, after each epoch,
trainer=ma_trainer2_trainer3
always set ma to be outer most

Paper:
Ensemble of Averages: Improving Model Selection and
Boosting Performance in Domain Generalization
Devansh Arpit, Huan Wang, Yingbo Zhou, Caiming Xiong
Salesforce Research, USA
"""

import torch
from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerMA(TrainerBasic):
    """
    initializer of this class goes to one block/section in the abstract class
    initializer, otherwise it will break the class inheritance.
    """
    def move_average(self, dict_data, epoch):
        """
        for each epoch, convex combine the weights for each layer
        Paper:
        Ensemble of Averages: Improving Model Selection and
        Boosting Performance in Domain Generalization
        Devansh Arpit, Huan Wang, Yingbo Zhou, Caiming Xiong
        Salesforce Research, USA
        """
        self.ma_weight_previous_model_params = epoch / (epoch + 1)
        # 1/2, 2/3, 3/4, 4/5, 
        # weight on previous model converges to 1 as training goes on
        dict_ema_para_curr_iter = {}
        for key, data in dict_data.items():
            # data = data.view(1, -1)  # make it rank 1 tensor (a.k.a. vector)
            if self._ma_iter == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self._ma_dict_para_persist[key]

            local_data_convex = \
                self.ma_weight_previous_model_params * previous_data + \
                (1 - self.ma_weight_previous_model_params) * data
            # correction by 1/(1 - self.rho)
            # so that the gradients amplitude backpropagated in data is
            # independent of self.rho
            dict_ema_para_curr_iter[key] = \
                local_data_convex / (1 - self.ma_weight_previous_model_params)
            self._ma_dict_para_persist[key] = \
                local_data_convex.clone().detach()  # used as previous data

        self._ma_iter += 1
        return dict_ema_para_curr_iter

    def after_epoch(self, epoch):
        torch_model = self.get_model()
        dict_para = torch_model.state_dict()  # only for trainable parameters
        new_dict_para = self.move_average(dict_para, epoch)
        # without deepcopy, this seems to work
        torch_model.load_state_dict(new_dict_para)
        super().after_epoch(epoch)
