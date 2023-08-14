import torch
import torch.nn as nn

# 잠재 변수 차원을 결정, 클수록 많은 정보 보존 + 모델 복잡
LATENT_DIM = 20


class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784,400),
            nn.ReLU(),
            nn.Linear(400, 2*LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon*std
    
    def forward(self, x):
        x = x.view(-1, 784)
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :LATENT_DIM]
        logvar = mu_logvar[:, LATENT_DIM:]

        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar