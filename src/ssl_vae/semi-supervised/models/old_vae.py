"""
M1 code replication from the paper
'Semi-Supervised Learning with Deep Generative Models'
(Kingma 2014) in PyTorch.

This "Latent-feature discriminative model" is eqiuvalent
to a classifier with VAE latent representation as input.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from layers import StochasticGaussian


class Encoder(nn.Module):
    """
    Inference network

    Attempts to infer the probability distribution
    p(z|x) from the data by fitting a variational
    distribution q_φ(z|x). Returns the two parameters
    of the distribution (µ, log σ²).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    """
    def __init__(self, convnet, dims, encoder_activation):
        super(Encoder, self).__init__()
        self.encoder_activation = encoder_activation
        self.convnet = convnet
        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        self.convnet = convnet
        if self.convnet: # cnn
            layers = [nn.Conv2d(3, 32, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            #nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            #nn.ReLU(True),
            #nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            nn.Linear(10,neurons[-1])]
        else: # dense
            layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(layers)
        self.sample = StochasticGaussian(h_dim[-1], z_dim)

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if not self.convnet:
                if i < len(self.hidden) - 1:
                    x = self.encoder_activation(x)#F.relu(x)
        return self.sample(x)


class Decoder(nn.Module):
    """
    Generative network

    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """
    def __init__(self, convnet, dims, decoder_activation):
        super(Decoder, self).__init__()
        self.decoder_activation = decoder_activation
        self.convnet = convnet
        [z_dim, h_dim, x_dim] = dims
        neurons = [z_dim, *h_dim]
        """
        if self.convnet:
            layers = [nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()]#,
            #nn.Linear(neurons[-2],neurons[-1])]
        else:
        """
        layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
    
        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            """
            if self.convnet:
                x = layer(x)
            else:
            """
            x = self.decoder_activation(layer(x)) #F.relu(layer(x))
        x = self.output_activation(self.reconstruction(x))
        return x


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder model consisting
    of an encoder/decoder pair for which a
    variational distribution is fitted to the
    encoder.

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    :return: (x_hat, latent) where latent is represented
        by parameters of the z-distribution along with a
        sample.
    """
    def __init__(self, convnet, dims ,encoder_activation=F.softplus, decoder_activation=F.softplus):
        super(VariationalAutoencoder, self).__init__()
        self.convnet = convnet
        [x_dim, z_dim, h_dim] = dims
        self.encoder = Encoder(self.convnet,[x_dim, h_dim, z_dim],encoder_activation)
        self.decoder = Decoder(self.convnet,[z_dim, list(reversed(h_dim)), x_dim],decoder_activation)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, (z, mu, log_var)

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

    
