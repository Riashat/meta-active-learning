from limitedmnist import LimitedMNIST
from torchvision.datasets import MNIST
from utils import generate_label, onehot
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from dgm import StackedDeepGenerativeModel, DeepGenerativeModel
from vae import VariationalAutoencoder
from loss import VariationalInferenceWithLabels, kl_divergence_normal, discrete_uniform_prior
from trainers import DGMTrainer

batch_size = 32

labels = np.arange(10)
n = len(labels)

# Load in data
mnist_lab = LimitedMNIST('./', train=True, transform=torch.bernoulli, target_transform=onehot(n), digits=labels, fraction=0.0025)
mnist_ulab = LimitedMNIST('./', train=True, transform=torch.bernoulli, target_transform=onehot(n), digits=labels, fraction=0.1)
mnist_val = LimitedMNIST('./', train=False, transform=torch.bernoulli, target_transform=onehot(n), digits=labels)

# Unlabelled data
unlabelled = torch.utils.data.DataLoader(mnist_ulab, batch_size=100, shuffle=True, num_workers=2)

# Validation data
validation = torch.utils.data.DataLoader(mnist_val, batch_size=1000, shuffle=True, num_workers=2)

# Labelled data
labelled = torch.utils.data.DataLoader(mnist_lab, batch_size=100, shuffle=True, num_workers=2)


epsilon = 1e-8
model = DeepGenerativeModel(ratio=len(mnist_ulab)/len(mnist_lab), dims=[28 * 28, n, 50, [500]])

def custom_logger(d):
    x, y = next(iter(validation))
    _, y_logits = torch.max(model.classifier(Variable(x)), 1)
    _, y = torch.max(y, 1)

    acc = torch.sum(y_logits.data == y)/len(y)
    d["Accuracy"] = acc
    
    print(d)

vae = VariationalAutoencoder([28*28, 50, [500]])
stacked_dgm = StackedDeepGenerativeModel([28*28, 10, 50, [500]], len(mnist_ulab)/len(mnist_lab), vae)


def binary_cross_entropy(r, x):
    return torch.sum((x * torch.log(r + epsilon) + (1 - x) * torch.log((1 - r) + epsilon)), dim=-1)


objective = VariationalInferenceWithLabels(binary_cross_entropy,kl_divergence_normal,discrete_uniform_prior)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

trainer = DGMTrainer(stacked_dgm, objective, optimizer, logger=custom_logger, cuda=False)
trainer.train(labelled, unlabelled, n_epochs=150)