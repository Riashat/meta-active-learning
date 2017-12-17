import sys
sys.path.append("bayesbench")
sys.path.append("probtorch")
sys.path.append("../")
from torch.autograd import Variable
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader

#from data.limitedmnist import LimitedMNIST
from torchvision.datasets import MNIST
from ss_utils import generate_label, onehot
from functools import wraps

# BayesNet
from bayesbench.benchmarking import regression
from bayesbench.networks.simple import SimpleMLP
from bayesbench.utils.metrics import accuracy
from bayesbench.methods import DeterministicMethod, MCDropoutMethod
from datatools import data_pipeline

# Keras CIFAR / MNIST utility
from keras import backend as K

# Probtorch
import probtorch

def expand_inputs(f):
    @wraps(f)
    def g(*args, num_samples=None, **kwargs):
        if not num_samples is None:
            new_args = []
            new_kwargs = {}
            for arg in args:
                if hasattr(arg, 'expand'):
                    new_args.append(arg.expand(num_samples, *arg.size()))
                else:
                    new_args.append(arg)
            for k in kwargs:
                arg = kwargs[k]
                if hasattr(arg, 'expand'):
                    new_args.append(arg.expand(num_samples, *arg.size()))
                else:
                    new_args.append(arg)
            return f(*new_args, num_samples=num_samples, **new_kwargs)
        else:
            return f(*args, num_samples=None, **kwargs)
    return g


class bnn_dataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            x, y = self.x[idx], self.y[idx]
            return x, y
        else:
            return self.x[idx]


class Encoder(nn.Module):
    def __init__(self, num_pixels, 
                       num_hidden,
                       num_digits,
                       num_style,
                       num_batch):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential( 
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.digit_log_weights = nn.Linear(num_hidden, num_digits)
        self.digit_temp = 0.66
        self.style_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.style_log_std = nn.Linear(num_hidden + num_digits, num_style)
    
    @expand_inputs
    def forward(self, images, labels=None, num_samples=None):
        q = probtorch.Trace()
        hiddens = self.enc_hidden(images) 
        digits = q.concrete(self.digit_log_weights(hiddens),
                            self.digit_temp,
                            value=labels,
                            name='digits')
        hiddens2 = torch.cat([digits, hiddens], -1)
        styles_mean = self.style_mean(hiddens2)
        styles_std = torch.exp(self.style_log_std(hiddens2))
        q.normal(styles_mean,
                 styles_std,
                 name='styles')
        return q

class Decoder(nn.Module):
    def __init__(self, num_pixels, 
                       num_hidden,
                       num_digits,
                       num_style):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.digit_log_weights = Parameter(torch.zeros(num_digits))
        self.digit_temp = 0.66
        self.style_mean = Parameter(torch.zeros(num_style))
        self.style_log_std = Parameter(torch.zeros(num_style))
        self.dec_hidden = nn.Sequential(
                            nn.Linear(num_style + num_digits, num_hidden),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())

    def forward(self, images=None, q=None):
        p = probtorch.Trace()
        digits = p.concrete(self.digit_log_weights, self.digit_temp,
                            value=q['digits'],
                            name='digits')
        styles = p.normal(0.0, 1.0,
                          value=q['styles'],
                          name='styles')
        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        images_mean = self.dec_image(hiddens)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x + 
                                  torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
               images_mean, images, name='images')
        return p

class ssl_vae:
    def __init__(self,
                 classes,
                 batch_size,
                 lr,
                 epochs,
                 dims,
                 beta1,
                 beta2,
                 samples,
                 eps,
                 cuda):
        self.classes = classes
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dims = dims
        self.beta1 = beta1
        self.beta2 = beta2
        self.samples = samples
        self.eps = eps
        self.cuda = cuda
        
        self.enc = Encoder(num_pixels=dims[0],num_hidden=dims[2][0],num_digits=classes,num_style=dims[1],num_batch=batch_size)
        self.dec = Decoder(num_pixels=dims[0],num_hidden=dims[2][0],num_digits=classes,num_style=dims[1])
        self.optimizer =  torch.optim.Adam(list(self.enc.parameters())+list(self.dec.parameters()),
                              lr=self.lr,
                              betas=(self.beta1, self.beta2))

    def __backward_pass(self,data):
        epoch_elbo = 0.0
        self.enc.train()
        self.dec.train()
        N = 0
        for b, (images, labels) in enumerate(data):
            if images.size()[0] == self.batch_size:
                N += self.batch_size
                images = images.view(-1, self.dims[0])
                labels_onehot = torch.zeros(self.batch_size, self.classes)
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                labels_onehot = torch.clamp(labels_onehot, self.eps, 1-self.eps)
                if self.cuda:
                    images = images.cuda()
                    labels_onehot = labels_onehot.cuda()
                images = Variable(images)
                labels_onehot = Variable(labels_onehot)
                self.optimizer.zero_grad()
                if self.labels:
                    q = self.enc(images, labels_onehot, num_samples=self.samples)
                else:
                    q = self.enc(images, num_samples=self.samples)
                p = self.dec(images, q)
                loss = -self.__elbo(q, p)
                loss.backward()
                self.optimizer.step()
                if self.cuda:
                    loss = loss.cpu()
                epoch_elbo -= loss.data.numpy()[0]
        return epoch_elbo / N

    def train(self, X_labeled, Y_labeled, X_unlabeled):
        self.X_labeled = X_labeled
        self.X_unlabeled = X_unlabeled
        self.Y_labeled = Y_labeled
        self.unlabeled = bnn_dataset(self.X_unlabeled,None,self.batch_size)
        self.labeled = bnn_dataset(self.X_labeled,self.Y_labeled,self.batch_size)
        for e in range(self.epochs):
            train_start = time.time()
            train_elbo_labeled = self.__backward_pass(self.labeled)
            train_elbo_unlabeled = self.__backward_pass(self.unlabeled)
            train_elbo = (train_elbo_labeled + train_elbo_unlabeled ) / 2.0
            train_end = time.time()
            print('[Epoch %d] Train: Labeled ELBO %.4e Unlabeled ELBO %.4e (%ds)' % (
                    e, train_elbo_labeled, train_elbo_unlabeled, train_end - train_start))

    def predict(self, X, infer=True):
        self.enc.eval()
        self.dec.eval()
        epoch_elbo = 0.0
        epoch_correct = 0
        N = 0
        y_preds = []
        for b, images in enumerate(X):
            N += 1
            images = images.view(-1, self.dims[0])
            if self.cuda:
                images = images.cuda()
            images = Variable(images)
            q = self.enc(images, num_samples=self.samples)
            p = self.dec(images, q)
            batch_elbo = self.__elbo(q, p)
            if self.cuda:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.data.numpy()[0]
            if infer:
                log_p = p.log_joint(0, 1)
                log_q = q.log_joint(0, 1)
                log_w = log_p - log_q
                w = torch.nn.functional.softmax(log_w, 0)
                y_samples = q['digits'].value
                y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
                _ , y_pred = y_expect.data.max(-1)
                if self.cuda:
                    y_pred = y_pred.cpu()
                y_preds.append(y_pred)
            else:
                _, y_pred = q['digits'].value.data.max(-1)
                if self.cuda:
                    y_pred = y_pred.cpu()
                y_preds.append(y_pred)
        return epoch_elbo / N, y_preds

    def evaluate(self,X,y_true, infer =True):
        epoch_elbo, y_preds = self.predict(X,infer=infer)
        acc = (y_true == y_pred).sum()*1.0 / (len(y_pred) or 1.0)
        return epoch_elbo, acc

    def __elbo(self,q, p, alpha=0.1):
        if self.samples is None:
            return probtorch.objectives.importance.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
        else:
            return probtorch.objectives.importance.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)

class SSClassifier:

    def __init__(self, params):
        self.params = params
        self.n_channels = 1
        self.img_rows = 28 
        self.img_cols = 28 
        self.__setup_SSClassifier()

    def __setup_SSClassifier(self):
        self.ssl_vae =ssl_vae(**self.params)

    def fit(self,X_labeled,Y_labeled,X_unlabeled):
        X_labeled = X_labeled.reshape(X_labeled.shape[0],np.prod(X_labeled.shape[1:]))
        X_unlabeled = X_unlabeled.reshape(X_unlabeled.shape[0],np.prod(X_unlabeled.shape[1:]))
        self.X_labeled = X_labeled
        self.X_unlabeled = X_unlabeled
        self.Y_labeled = Y_labeled
        self.ssl_vae.train(self.X_labeled,self.Y_labeled,self.X_unlabeled)

    def predict(self,X,infer=True):
        X = X.reshape(X.shape[0],np.prod(X.shape[1:]))
        return self.ssl_vae.predict(X,infer=infer)

    def evaluate(self,X,y_true,infer=True):
        X = X.reshape(X.shape[0],np.prod(X.shape[1:]))
        return self.ssl_vae.evaluate(X,y_true, infer=infer)

if __name__ == "__main__":
    training_data, validation_data, pool_data, testing_data = data_pipeline(valid_ratio=0.3, dataset='mnist')
    X_labeled,Y_labeled = validation_data #training_data
    X_unlabeled, Y_unlabeled = pool_data #validation_data

    params = {
        'batch_size':50,
        'lr':3e-5,
        'epochs':1,
        'dims':[784, 50, [600]], # 784: MNIST, 512: CIFAR10
        'samples':8,
        'beta1':0.9,
        'beta2':0.999,
        'eps':1e-9,
        'cuda':torch.cuda.is_available()
    }

    model = SSClassifier(params)
    model.fit(X_labeled, Y_labeled, X_unlabeled)
    