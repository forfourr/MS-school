import torch
import matplotlib.pyplot as plt
from auto_model import AutoEncoder

## load model
load_auto_model = AutoEncoder()
load_auto_model.load_state_dict(torch.load("autoencoder_model.pt",
                                map_location='cpu'))
load_auto_model.eval()


with torch.no_grad():
    test_sample = torch.randn(1,32)
    generated_sample = load_auto_model.decoder(test_sample).view(1,1,28,28)

plt.imshow(generated_sample.squeeze().numpy(), cmap='gray')
plt.show()