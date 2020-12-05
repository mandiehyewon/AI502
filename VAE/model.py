import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        return z_mu, z_var

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        predicted = torch.sigmoid(self.out(hidden))

        return predicted

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mu, z_var = self.encoder(x)

        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        predicted = self.decoder(x_sample)

        return predicted, z_mu, z_var

# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)
#         self.fc3 = nn.Linear(20, 400)
#         self.fc4 = nn.Linear(400, 784)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

class VAE_Linear(nn.Module):
    def __init__(self, args,):
        super(VAE_Linear, self).__init__()
        self.dataset_name = args.dataset_name.upper()
        if self.dataset_name == 'MNIST':
            self.rec_criterion = nn.BCELoss(reduction='sum')
            self.hidden_size = 500
        elif self.dataset_name == 'FREY-FACE':
            self.rec_criterion = nn.BCELoss(reduction='sum')
            self.hidden_size = 200

        self.feature_size = args.image_height * args.image_width
        self.latent_size = args.latent_size

        self.encoder_fc = nn.Linear(in_features=self.feature_size, \
                                    out_features=self.hidden_size, \
                                    bias=True, \
                                   )
        self.encoder_relu = nn.ReLU()

        self.mu_fc = nn.Linear(in_features=self.hidden_size, \
                               out_features=self.latent_size, \
                               bias=True, \
                              )
        self.logvar_fc = nn.Linear(in_features=self.hidden_size, \
                                   out_features=self.latent_size, \
                                   bias=True, \
                                  )
        self.z_fc = nn.Linear(in_features=self.latent_size, \
                              out_features=self.hidden_size, \
                              bias=True, \
                             )
        self.z_relu = nn.ReLU()

        self.decoder_fc = nn.Linear(in_features=self.hidden_size, \
                                    out_features=self.feature_size, \
                                    bias=True, \
                                   )

        if args.ckpt is None:
            self._init_weights()

    def _init_weights(self,):
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(params)

    def reparameterize(self, mu, logvar,):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)

        return mu + std * eps

    def encode(self, x,):
        x = self.encoder_fc(x)
        x = self.encoder_relu(x)

        return x

    def decode(self, x,):
        x = self.decoder_fc(x)

        return x

    def forward(self, x,):
        x = self.encode(x,)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        z = self.reparameterize(mu, logvar,)
        feat = self.z_fc(z)
        feat = self.z_relu(feat)

        rec_x = self.decode(feat,)
        rec_x = torch.sigmoid(rec_x)

        return dict(rec_x=rec_x, mu=mu, logvar=logvar,)

    def generate(self, z,):
        feat = self.z_fc(z)
        feat = self.z_relu(feat)

        gen_x = self.decode(feat,)
        gen_x = torch.sigmoid(gen_x)

        return dict(gen_x=gen_x,)

    def compute_rec_loss(self, pred, target,):
        rec_loss = self.rec_criterion(pred, target)

        return rec_loss

    def compute_kld_loss(self, mu, logvar,):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return kld_loss