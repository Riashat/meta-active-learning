import sys
sys.path.append("semi-supervised")
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from data.limitedmnist import LimitedMNIST
from torchvision.datasets import MNIST
from utils import generate_label, onehot

from models import DeepGenerativeModel
from inference.loss import VariationalInferenceWithLabels, VariationalInference

from torch.autograd import Variable
from trainers import DGMTrainer, VAETrainer

from models import VariationalAutoencoder, StackedDeepGenerativeModel

import numpy as np

batch_size = 32

labels = np.arange(10)
n = len(labels)

# Load in data
mnist_lab = LimitedMNIST('./', train=True, transform=torch.bernoulli, target_transform=onehot(n), digits=labels, fraction=0.0025)
mnist_ulab = LimitedMNIST('./', train=True, transform=torch.bernoulli, target_transform=onehot(n), digits=labels, fraction=0.5)
mnist_val = LimitedMNIST('./', train=False, transform=torch.bernoulli, target_transform=onehot(n), digits=labels)
"""
import matplotlib.pyplot as plt
plt.imshow(mnits_lab[0])
plt.show()
"""
# Unlabelled data
unlabelled = torch.utils.data.DataLoader(mnist_ulab, batch_size=100, shuffle=True, num_workers=2)

# Validation data
validation = torch.utils.data.DataLoader(mnist_val, batch_size=1000, shuffle=True, num_workers=2)

# Labelled data
labelled = torch.utils.data.DataLoader(mnist_lab, batch_size=100, shuffle=True, num_workers=2)


epsilon = 1e-8
model = DeepGenerativeModel(ratio=len(mnist_ulab)/len(mnist_lab), dims=[28 * 28, n, 50, [600]])

def custom_logger(d):
    x, y = next(iter(validation))
    _, y_logits = torch.max(model.classifier(Variable(x)), 1)
    _, y = torch.max(y, 1)

    acc = torch.sum(y_logits.data == y)/len(y)
    d["Accuracy"] = acc
    
    print(d)

vae = VariationalAutoencoder([28*28, 50, [600,600]])

def binary_cross_entropy(r, x):
    return torch.sum((x * torch.log(r + epsilon) + (1 - x) * torch.log((1 - r) + epsilon)), dim=-1)

# Train DGM (M2)
objective = VariationalInferenceWithLabels(binary_cross_entropy)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
trainer = DGMTrainer(model, objective, optimizer, logger=custom_logger, cuda=False)
#trainer.train(labelled, unlabelled, 10)

# Train VAE (M1)
objective = VariationalInference(binary_cross_entropy)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
trainer = VAETrainer(vae, objective, optimizer, args={"iw": 1, "eq": 1, "temperature": 1})
trainer.train(None, unlabelled, 10)

#Train SDGM (M1+M2)
stacked_dgm = StackedDeepGenerativeModel([28*28, 10, 50, [600,600]], len(mnist_ulab)/len(mnist_lab), vae)
objective = VariationalInferenceWithLabels(binary_cross_entropy)
m=[]
m.extend(list(stacked_dgm.encoder.parameters()))
m.extend(list(stacked_dgm.decoder.parameters()))
m.extend(list(stacked_dgm.classifier.parameters()))

optimizer = torch.optim.Adam(m, lr=1e-3)
trainer = DGMTrainer(stacked_dgm, objective, optimizer, logger=custom_logger, cuda=False)
trainer.train(labelled, unlabelled, 200)

exit()

use_visdom = True

if use_visdom:
    import visdom
    vis = visdom.Visdom()

class Visualiser():
    def __init__(self):
        self.loss = vis.line(X=np.array([0]), Y=np.array([0]), opts=dict(title="Training Loss", xlabel="Epoch"))
        self.acc  = vis.line(X=np.array([0]), Y=np.array([0]), opts=dict(title="Accuracy", xlabel="Epoch"))

    def update_loss(self, L, U):
        vis.updateTrace(X=np.array([epoch]), Y=L.data.numpy(), win=self.loss, name="Labelled")
        vis.updateTrace(X=np.array([epoch]), Y=U.data.numpy(), win=self.loss, name="Unlabelled")
        
    def update_accuracy(self, model):
        accuracy = []
        for x, y in validation:
            _, prediction = torch.max(model.classifier(Variable(x)), 1)
            _, y = torch.max(y, 1)

            accuracy += [torch.mean((prediction.data == y).float())]

        vis.updateTrace(X=np.array([epoch]), Y=np.array([np.mean(accuracy)]), win=self.acc)
        
    def update_images(self, model):
        x, y = next(iter(validation))
        input = Variable(x[:5])
        label = Variable(y[:5].type(torch.FloatTensor))
        x_hat, *_ = model(input, label)
        images = x_hat.data.numpy().reshape(-1, 1, 28, 28)

        vis.images(images, opts=dict(width=5*64, height=64, caption="Sample epoch {}".format(epoch)))

trainer = DGMTrainer(model, objective, optimizer)
visual = Visualiser()

for epoch in range(1001):
    L, U = trainer.train(labelled, unlabelled,n_epochs=2)

    if use_visdom:
        # Plot the last L and U of the epoch
        visual.update_loss(L, U)
        visual.update_accuracy(model)

        if epoch % 10 == 0:
            visual.update_images(model)