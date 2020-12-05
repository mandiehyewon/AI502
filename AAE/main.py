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

from model import Encoder, Decoder, AAE
from test import test
from train import train
from utils import to_np, to_var, loss_function, optiminitializer
from logger import Logger

parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False)

parser.add_argument('--num-epochs', type=int, default=1500)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--genlr', type=float, default=1e-4)
parser.add_argument('--reglr', type=float, default=5e-5)

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
EPS = 1e-15
latentspace_dim = 120
Enc = Encoder(784, 1000, latentspace_dim).cuda()
Dec = Dncoder(784, 1000, latentspace_dim).cuda()
Discrim = Discriminator(500, latentspace_dim).cuda()

criterion = nn.CrossEntropyLoss()  

#encoder/decoder optimizer
optim_Enc = optiminitializer(args, Enc, lr=args.genlr)
optim_Dec = optiminitializer(args, Dec, lr=args.genlr)
#Regularizing optimizer
optim_Enc_gen = optiminitializer(args, Enc, lr=args.reglr)
optim_Discrim = optiminitializer(args, Discrim, lr=args.reglr)

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

# def main(args,):
# 	for epoch in range(1, args.num_epochs + 1):
# 		scheduler.step()
# 		train(args, Enc, Dec, Discrim, optimizer, train_loader, device )
# 		test(epoch, model, optimizer, test_loader, device, args.batch_size)
	
# 		with torch.no_grad():
# 			sample = torch.randn(64, 20).to(device)
# 			sample = model.decode(sample).cpu()
# 			save_image(sample.view(64, 1, 28, 28),
# 					   './results/sample_' + str(epoch) + '.png')

# 	if args.save_model:
# 		torch.save(model.state_dict(), "vae.pt")

# if __name__ == '__main__':
# 	print(args.__dict__)
# 	print()
# 	main(args,)


# data_iter = iter(data_loader)
# iter_per_epoch = len(data_loader)
	
for step in range(args.epoch):

	# Reset the data_iter
	if (step+1) % args.iter_per_epoch == 0:
	# 	data_iter = iter(data_loader)
	for i, (image, label) in enumerate(train_loader):
		start_time = time.time()

		for param in Discrim.parameters():
			param.requires_grad = False

	# # Fetch the images and labels and convert them to variables
	# images, labels = next(data_iter)
	# images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)

	#finding reconstruction loss
	Dec.zero_grad()
	Enc.zero_grad()
	Discrim.zero_grad()

	z_sample = Enc(images)   #encode to z
	X_sample = Dec(z_sample) #decode to X reconstruction
	recon_loss = F.binary_cross_entropy(X_sample,images)

	recon_loss.backward()
	optim_Dec.step()
	optim_Enc.step()

	# Discriminator
	## true prior is random normal (randn)
	## this is constraining the Z-projection to be normal!
	Enc.eval()
	z_real = Variable(torch.randn(images.size()[0], latentspace_dim) * 5.).cuda()
	D_real = Discrim(z_real)

	z_fake = Enc(images)
	D_fake = Discrim(z_fake)

	D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))

	D_loss.backward()
	optim_Dec.step()

	# Generator
	Enc.train()
	z_fake = Enc(images)
	D_fake = Discrim(z_fake)
	
	G_loss = -torch.mean(torch.log(D_fake))

	G_loss.backward()
	optim_Enc_gen.step()   

	
	if (step+1) % 100 == 0:
		# print ('Step [%d/%d], Loss: %.4f, Acc: %.2f' 
		#		%(step+1, total_step, loss.data[0], accuracy.data[0]))

		#============ TensorBoard logging ============#
		# (1) Log the scalar values
		info = {
			'recon_loss': recon_loss.data[0],
			'discriminator_loss': D_loss.data[0],
			'generator_loss': G_loss.data[0]
		}

		for tag, value in info.items():
			logger.scalar_summary(tag, value, step+1)

		# (2) Log values and gradients of the parameters (histogram)
		for net,name in zip([Enc,Dec,Discrim],['Encoder','Decoder','Discrim']): 
			for tag, value in net.named_parameters():
				tag = name+tag.replace('.', '/')
				logger.histo_summary(tag, to_np(value), step+1)
				logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)

		# (3) Log the images
		info = {
			'images': to_np(images.view(-1, 28, 28)[:10])
		}

		for tag, images in info.items():
			logger.image_summary(tag, images, step+1)

#save the Encoder
torch.save(Enc.state_dict(),'Encoder_weights.pt')
