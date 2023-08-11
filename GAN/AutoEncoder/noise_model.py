import torch.nn as nn


latent_dim = 20

class NoisyAutoEncoder(nn.Module):
    def __init__(self):
        super(NoisyAutoEncoder,self).__init__()

        ### input image = 28*28 = 784
        self.encoder = nn.Sequential(
            nn.Linear(784,400),
            nn.ReLU(),
            nn.Linear(400,200),
            nn.ReLU(),
            nn.Linear(200, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Tanh()
        )

    
    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded