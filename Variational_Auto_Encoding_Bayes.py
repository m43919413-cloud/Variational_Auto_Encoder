import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.w1 = nn.Linear(input_dim, hidden_dim)

        self.w_mu = nn.Linear(hidden_dim, latent_dim)

        self.w_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.tanh(self.w1(x))

        mu = self.w_mu(h)

        logvar = self.w_logvar(h)

        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.w1 = nn.Linear(latent_dim, hidden_dim)

        self.w2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        y = torch.sigmoid(self.w2(torch.tanh(self.w1(z))))

        return y


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)

    eps = torch.randn_like(std)

    return mu + std * eps


class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)

        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = reparameterize(mu, logvar)

        y = self.decoder(z)

        return y, mu, logvar

    def loss(self, x, y, mu, logvar, beta=1.0, eps=1e-7):
        log_px = torch.sum(
            x * torch.log(y + eps) + (1 - x) * torch.log(1 - y + eps),
            dim=-1,
        )

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        return (-log_px + beta * kl).mean()