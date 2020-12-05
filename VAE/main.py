from __future__ import print_function
import os
import random
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import Encoder, Decoder, VAE
from test import test
from train import train
from utils import loss_function, optiminitializer

parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False)

parser.add_argument('--num-epochs', type=int, default=1500)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--nesterov', action='store_true')

parser.add_argument('--kld-loss-coef', type=float, default=1.0)
parser.add_argument('--rec-loss-coef', type=float, default=1.0)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-model', action='store_true', default=False)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if args.cuda else "cpu")

#Define variables
INPUT_DIM = 28 * 28
HIDDEN_DIM = 400
LATENT_DIM = 20

#Random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# SEt Model and Optimizer
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
model = VAE(encoder, decoder).to(device)

optimizer = optiminitializer(args, model, args.lr)
scheduler = StepLR(optimizer, step_size=1)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#Train, Testloader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def main(args,):
    for epoch in range(1, args.num_epochs + 1):
        scheduler.step()
        train(epoch, model, optimizer, train_loader, device, args.log_interval)
        test(epoch, model, optimizer, test_loader, device, args.batch_size)
    
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

    if args.save_model:
        torch.save(model.state_dict(), "vae.pt")

if __name__ == '__main__':
    print(args.__dict__)
    print()
    main(args,)
