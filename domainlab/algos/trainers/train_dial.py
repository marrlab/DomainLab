"""
use random start to generate adversarial images
"""
import torch
from torch.autograd import Variable

from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerDIAL(TrainerBasic):
    """
    Trainer Domain Invariant Adversarial Learning
    """
    def gen_adversarial(self, device, img_natural, vec_y):
        """
        use naive trimming to find optimize img in the direction of adversarial gradient,
        this is not necessarily constraint optimal due to nonlinearity,
        as the constraint epsilon is only considered ad-hoc
        """
        # @FIXME: is there better way to initialize adversarial image?
        # ensure adversarial image not in computational graph
        steps_perturb = self.aconf.dial_steps_perturb
        scale = self.aconf.dial_noise_scale
        step_size = self.aconf.dial_lr
        epsilon = self.aconf.dial_epsilon
        img_adv_ini = img_natural.detach()
        img_adv_ini = img_adv_ini + scale * torch.randn(img_natural.shape).to(device).detach()
        img_adv = img_adv_ini
        for _ in range(steps_perturb):
            img_adv.requires_grad_()
            loss_gen_adv = self.model.cal_loss_gen_adv(img_natural, img_adv, vec_y)
            grad = torch.autograd.grad(loss_gen_adv, [img_adv])[0]
            # instead of gradient descent, we gradient ascent here
            img_adv = img_adv_ini.detach() + step_size * torch.sign(grad.detach())
            img_adv = torch.min(torch.max(img_adv, img_natural - epsilon), img_natural + epsilon)
            img_adv = torch.clamp(img_adv, 0.0, 1.0)
        return img_adv

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        Let trainer behave like a model, so that other trainer could use it
        """
        _ = tensor_d
        _ = others
        tensor_x_adv = self.gen_adversarial(self.device, tensor_x, tensor_y)
        tensor_x_batch_adv_no_grad = Variable(tensor_x_adv, requires_grad=False)
        loss_dial = self.model.cal_task_loss(tensor_x_batch_adv_no_grad, tensor_y)
        return [loss_dial], [self.aconf.gamma_reg]
