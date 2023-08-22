import torch
from torch.nn import CrossEntropyLoss, Linear
from backpack.utils.examples import load_one_batch_mnist
from backpack import extend, backpack
from backpack.extensions import Variance

X, y = load_one_batch_mnist(flat=True)


X.shape
y.shape
y


# def exp_reducer(x):
#    return x.exp().sum(dim=1)


# inputs = torch.rand(2, 2)
#inputs.exp()
#inputs.sum(dim=0)
#inputs.sum(dim=1)

#inputs.shape
# jacobian w.r.t. to data
# rst = torch.autograd.functional.jacobian(exp_reducer, inputs)
#rst.shape


model0 = Linear(784, 10)
model = extend(Linear(784, 10))
lossfunc = extend(CrossEntropyLoss())
dir(lossfunc)
loss = lossfunc(model(X), y)

with backpack(Variance()):
    loss.backward()

list_param_grad = []
list_param_var = []
for param in model.parameters():
    list_param_grad.append(param.grad)
    list_param_var.append(param.variance)

for (name, param) in model.named_parameters():
    list_param_grad.append(param.grad_batch)
    list_param_grad.append(param.grad)
    list_param_var.append(param.variance)



# list_para = list(model.parameters())
# result = torch.autograd.functional.jacobian(model, list_para[0])

result.shape
X.shape
list_para[0].shape

dir(model)
list(model.named_parameters())

list_par = list(model.parameters())

len(list_par)
list_par[0].shape   # weight
list_par[1].shape   # bias

X.shape

list_param_grad[0].shape
list_param_grad[1].shape

list_param_var[0].shape

list_param_var[1]
list_param_grad[1]


