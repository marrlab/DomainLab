import torch


def mk_img(i_h, i_ch=3, batch_size=5):
    img = torch.rand(i_h, i_h)  # uniform distribution [0,1]
    # x = torch.clamp(x, 0, 1)
    img.unsqueeze_(0)
    img = img.repeat(i_ch, 1, 1) # RGB image
    img.unsqueeze_(0)
    img = img.repeat(batch_size, 1, 1, 1)
    return img

def mk_rand_label_onehot(target_dim=10, batch_size=5):
    label_scalar = torch.randint(high=target_dim, size=(batch_size, ))
    label_scalar2 = label_scalar.unsqueeze(1)
    label_zeros = torch.zeros(batch_size, target_dim)
    label_onehot = torch.scatter(input=label_zeros, dim=1, index=label_scalar2, value=1.0)
    return label_onehot

def mk_rand_xyd(ims, y_dim, d_dim, batch_size):
    imgs = mk_img(i_h=ims, batch_size=batch_size)
    ys = mk_rand_label_onehot(target_dim=y_dim, batch_size=batch_size)
    ds = mk_rand_label_onehot(target_dim=d_dim, batch_size=batch_size)
    return imgs, ys, ds
