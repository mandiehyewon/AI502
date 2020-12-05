import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
	    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.gaussian = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
		layer = F.dropout(self.linear(x), p=0.25, training = self.training)
		layer = F.relu(layer)
		layer = F.dropout(self.linear2(layer), p=0.25, training = self.training)
		layer = F.relu(layer)
		output = self.gaussian(layer)

        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
	    self.linear1 = nn.Linear(hidden_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        layer = F.dropout(self.linear(x), p=0.25, training = self.training)
        layer = F.relu(layer)
        layer = F.dropout(self.linear1(layer), p=0.25, training = self.training)
        layer = linear2(layer)

        return F.sigmoid(layer)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, hidden_dim ,z_dim):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
	    self.linear1 = nn.Linear(hidden_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        layer = F.dropout(self.linear(layer), p=0.2, training=self.training)
        layer = F.relu(layer)
        layer = F.dropout(self.linear1(layer), p=0.2, training=self.training)
        layer = F.relu(layer)

        return F.sigmoid(self.linear2(layer))
