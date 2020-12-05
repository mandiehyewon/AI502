import torch
from torch.nn import functional as F

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD #reconstruction loss + KL divergence

def optiminitializer(args, model, lr):
    name = args.optim.upper()
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if name == 'ADAM':
        optim = torch.optim.Adam(params=params, eps=args.eps,
                     weight_decay=args.weight_decay, lr = lr)

    elif name == 'RMSPROP':
        optim = torch.optim.RMSprop(params=params, lr=args.lr, 
                        eps=args.eps, weight_decay=args.weight_decay)

    elif name == 'SGD':
        optim = torch.optim.SGD(params=params, lr=args.lr, weight_decay=args.weight_decay)

    return optim