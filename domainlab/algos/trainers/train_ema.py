import torch


class MovingAverage:
    def __init__(self, rho):
        self.rho = rho
        self.dict_para_ema_persist = {}
        self._iter = 0

    def update(self, dict_data):
        dict_ema_para_curr_iter = {}
        for key, data in dict_data.items():
            data = data.view(1, -1)  # make it rank 1 tensor (a.k.a. vector)
            if self._iter == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.dict_para_ema_persist[key]

            local_data_convex = self.rho * previous_data + (1 - self.rho) * data
            # correction by 1/(1 - self.rho)
            # so that the gradients amplitude backpropagated in data is independent of self.rho
            dict_ema_para_curr_iter[key] = local_data_convex / (1 - self.rho)
            self.dict_para_ema_persist[key] = local_data_convex.clone().detach()  # used as previous data

        self._iter += 1
        return dict_ema_para_curr_iter
