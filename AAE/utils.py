import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)    

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


# Useful functions adopted from : https://github.com/neale/Adversarial-Autoencoder/blob/master/utils.py
def save_model(net, optim, epoch, path):
    state_dict = net.state_dict()
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer': optim.state_dict(),
        }, path)


def generate_ae_image(iter, Enc, Gen, save_path, args, real_data):
    batch_size = args.batch_size
    datashape = Enc.shape
    encoding = Enc(real_data)
    samples = Gen(encoding)
    samples = samples.view(batch_size, 28, 28)

    samples = samples.cpu().data.numpy()
    save_images(samples, save_path+'/ae_samples_{}.jpg'.format(iter))


def generate_image(iter, model, save_path, args):
    batch_size = args.batch_size
    datashape = model.shape
    if model._name == 'mnistG':
        fixed_noise_128 = torch.randn(batch_size, args.dim).cuda()
    else:
        fixed_noise_128 = torch.randn(128, args.dim).cuda()
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = model(noisev)
    if model._name == 'mnistG':
        samples = samples.view(batch_size, 28, 28)
    else:
        samples = samples.view(-1, *(datashape[::-1]))
        samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(samples, save_path+'/samples_{}.jpg'.format(iter))


def save_images(X, save_path, use_np=False):
    # [0, 1] -> [0,255]
    plt.ion()
    if not use_np:
        if isinstance(X.flatten()[0], np.floating):
            X = (255.99*X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, int(n_samples/rows)
    if X.ndim == 2:
        s = int(np.sqrt(X.shape[1]))
        X = np.reshape(X, (X.shape[0], s, s))
    if X.ndim == 4:
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        img[j*h:j*h+h, i*w:i*w+w] = x

    plt.imshow(img, cmap='gray')
    plt.draw()
    plt.pause(0.001)

    if use_np:
        np.save(save_path, img)
    else:
        imsave(save_path, img)