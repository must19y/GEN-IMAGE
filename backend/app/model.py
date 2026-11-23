import torch.nn as nn


class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        z_mean, z_log_var = inputs
        eps = torch.randn_like(z_mean)          # same shape & device as z_mean
        std = torch.exp(0.5 * z_log_var)
        return z_mean + std * eps



class CelebAEncoder(nn.Module):

    def __init__(self, latent_dim=200):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)

        self.flatten = nn.Flatten()

        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)

        self.sampling = Sampling()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)

        x = self.flatten(x)

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z


## now the decoder

class CelebADecoder(nn.Module):

    def __init__(self, latent_dim=200):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)

        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(128)

        self.conv_out = nn.ConvTranspose2d(
            128, 3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):

        x = self.fc(z)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = x.view(-1, 128, 2, 2)

        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.deconv2(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        x = self.deconv3(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)

        x = self.deconv4(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)

        x = self.conv_out(x)
        x = torch.sigmoid(x)

        return x



## now coding up the architecture VAE


import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)

        return z_mean, z_log_var, reconstruction


## building the model

def build_model():
    encoder = CelebAEncoder(latent_dim=200)
    decoder = CelebADecoder(latent_dim=200)

    model = VAE(encoder,decoder)

    return model