import numpy as np
import torch
import scipy.misc
import torch.autograd as autograd
from scipy.misc.pilutil import imsave



def calc_gradient_penalty(args, model, real_data, gen_data):
    datashape = model.shape
    alpha = torch.rand(args.batch_size, 1)
    real_data = real_data.view(args.batch_size, -1)
    if args.dataset == 'mnist':
        alpha = alpha.expand(real_data.size()).cuda()
    else:
        alpha = alpha.expand(args.batch_size, real_data.nelement()//args.batch_size)
        alpha = alpha.contiguous().view(args.batch_size, -1).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = model(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, 
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),      
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True)[0]

    if args.dataset != 'mnist':
        gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty



