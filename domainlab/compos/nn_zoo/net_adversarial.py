from torch import nn
from torch.autograd import Function


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class AutoGradFunReverseMultiply(Function):
    """
    https://pytorch.org/docs/stable/autograd.html
    https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class AutoGradFunMultiply(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        return output, None
