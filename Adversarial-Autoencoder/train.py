import os
import sys
import time
import argparse
import numpy as np
#from scipy.misc import imread

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F

import ops
import plot
import utils
import datagen
import encoders
import generators
import discriminators
from data import mnist
from data import cifar10


def load_args():

    parser = argparse.ArgumentParser(description='aae-wgan')
    parser.add_argument('-d', '--dim', default=100, type=int, help='latent space size')
    parser.add_argument('-l', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-o', '--output_dim', default=4096, type=int)
    parser.add_argument('--dataset', default='celeba')
    parser.add_argument('--use_spectral_norm', default=True)
    args = parser.parse_args()
    return args


def load_models(args):
    if args.dataset in ['mnist', 'fmnist']:
        netG = generators.MNISTgenerator(args).cuda()
        netD = discriminators.MNISTdiscriminator(args).cuda()
        netE = encoders.MNISTencoder(args).cuda()

    if args.dataset in ['cifar', 'cifar_hidden']:
        netG = generators.CIFARgenerator(args).cuda()
        netD = discriminators.CIFARdiscriminator(args).cuda()
        netE = encoders.CIFARencoder(args).cuda()

    if args.dataset == 'celeba':
        netG = generators.CELEBAgenerator(args).cuda()
        netD = discriminators.CELEBAdiscriminator(args).cuda()
        netE = encoders.CELEBAencoder(args).cuda()
	
    print (netG, netD, netE)
    return (netG, netD, netE)


def load_data(args):
    if args.dataset == 'mnist':
        return datagen.load_mnist(args)
    if args.dataset == 'cifar':
        return datagen.load_cifar(args)
    if args.dataset == 'fmnist':
        return datagen.load_fashion_mnist(args)
    if args.dataset == 'cifar_hidden':
        class_list = [0] ## just load class 0
        return datagen.load_cifar_hidden(args, class_list)
    else:
        print ('Dataset not specified correctly')
        print ('choose --dataset <mnist, fmnist, cifar, cifar_hidden>')


def train():
    args = load_args()
    train_gen, test_gen = load_data(args)
    torch.manual_seed(1)
    netG, netD, netE = load_models(args)

    if args.use_spectral_norm:
        optimizerD = optim.Adam(filter(lambda p: p.requires_grad,
            netD.parameters()), lr=2e-4, betas=(0.0,0.9))
    else:
        optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9))
    optimizerE = optim.Adam(netE.parameters(), lr=2e-4, betas=(0.5, 0.9))

    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99) 
    schedulerE = optim.lr_scheduler.ExponentialLR(optimizerE, gamma=0.99)
    
    ae_criterion = nn.MSELoss()
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    iteration = 0 
    for epoch in range(args.epochs):
        for i, (data, targets) in enumerate(train_gen):
            start_time = time.time()
            """ Update AutoEncoder """
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            netE.zero_grad()
            real_data_v = autograd.Variable(data).cuda()
            real_data_v = real_data_v.view(args.batch_size, -1)
            encoding = netE(real_data_v)
            fake = netG(encoding)
            ae_loss = ae_criterion(fake, real_data_v)
            ae_loss.backward(one)
            optimizerE.step()
            optimizerG.step()
            
            """ Update D network """
            for p in netD.parameters():  
                p.requires_grad = True 
            for i in range(5):
                real_data_v = autograd.Variable(data).cuda()
                # train with real data
                netD.zero_grad()
                D_real = netD(real_data_v)
                D_real = D_real.mean()
                D_real.backward(mone)
                # train with fake data
                noise = torch.randn(args.batch_size, args.dim).cuda()
                noisev = autograd.Variable(noise, volatile=True)
                fake = autograd.Variable(netG(noisev).data)
                inputv = fake
                D_fake = netD(inputv)
                D_fake = D_fake.mean()
                D_fake.backward(one)

                # train with gradient penalty 
                gradient_penalty = ops.calc_gradient_penalty(args,
                        netD, real_data_v.data, fake.data)
                gradient_penalty.backward()

                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD.step()

            # Update generator network (GAN)
            noise = torch.randn(args.batch_size, args.dim).cuda()
            noisev = autograd.Variable(noise)
            fake = netG(noisev)
            G = netD(fake)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step() 

            schedulerD.step()
            schedulerG.step()
            schedulerE.step()
            # Write logs and save samples 
            save_dir = './plots/'+args.dataset
            plot.plot(save_dir, '/disc cost', D_cost.cpu().data.numpy())
            plot.plot(save_dir, '/gen cost', G_cost.cpu().data.numpy())
            plot.plot(save_dir, '/w1 distance', Wasserstein_D.cpu().data.numpy())
            plot.plot(save_dir, '/ae cost', ae_loss.data.cpu().numpy())
            
            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 99:
                dev_disc_costs = []
                for i, (images, targets) in enumerate(test_gen):
                    imgs_v = autograd.Variable(images, volatile=True).cuda()
                    D = netD(imgs_v)
                    _dev_disc_cost = -D.mean().cpu().data.numpy()
                    dev_disc_costs.append(_dev_disc_cost)
                plot.plot(save_dir ,'/dev disc cost', np.mean(dev_disc_costs))
                utils.generate_image(iteration, netG, save_dir, args)
                # utils.generate_ae_image(iteration, netE, netG, save_dir, args, real_data_v)

            # Save logs every 100 iters 
            if (iteration < 5) or (iteration % 100 == 99):
                plot.flush()
            plot.tick()
            if iteration % 100 == 0:
                utils.save_model(netG, optimizerG, iteration,
                        'models/{}/G_{}'.format(args.dataset, iteration))
                utils.save_model(netD, optimizerD, iteration, 
                        'models/{}/D_{}'.format(args.dataset, iteration))
            iteration += 1


        
if __name__ == '__main__':
    train()
