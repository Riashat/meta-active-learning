import sys
#sys.path.append("bayesbench")
sys.path.append("probtorch")
sys.path.append("../")
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader

from torchvision.datasets import MNIST
from functools import wraps
import time
import uuid

# BayesNet
"""
from bayesbench.benchmarking import regression
from bayesbench.networks.simple import SimpleMLP
from bayesbench.utils.metrics import accuracy
from bayesbench.methods import DeterministicMethod, MCDropoutMethod
"""
from keras import backend as K
from datatools import data_pipeline

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
    def __init__(self, x,y,batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        f= lambda x: x // self.batch_size + (0 if x % self.batch_size == 0 else 1)
        return f(len(self.x))

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = start + self.batch_size
        if self.y is not None:
            return self.x[start:end], self.y[start:end]
        else:
            return self.x[start:end], None


class Encoder(nn.Module):
    def __init__(self, num_pixels, 
                       num_hidden,
                       num_digits,
                       num_style,
                       num_batch,
                       cuda,
                       cnn,
                       input_dimensions):
        super(self.__class__, self).__init__()
        self.cnn = cnn
        if self.cnn:
            self.enc_cnn_1 = nn.Conv2d(input_dimensions[1], 10, kernel_size=5)
            self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
            self.enc_linear_1 = nn.Linear(320, 50)
            self.enc_linear_2 = nn.Linear(50, num_hidden)
        else:
            self.enc_hidden = nn.Sequential( 
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.digit_log_weights = nn.Linear(num_hidden, num_digits)
        self.digit_temp = 0.66
        self.style_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.style_log_std = nn.Linear(num_hidden + num_digits, num_style)
        if cuda:
            if self.cnn:
                self.enc_hidden.cuda()
            self.digit_log_weights.cuda()
            self.style_mean.cuda()
            self.style_log_std.cuda()
    
    @expand_inputs
    def forward(self, images, labels=None, num_samples=None):
        q = probtorch.Trace()
        if self.cnn:
            hiddens = []
            def hidden_pass(inp):
                hiddens = self.enc_cnn_1(inp)
                hiddens = F.selu(F.max_pool2d(hiddens, 2))
                hiddens = self.enc_cnn_2(hiddens)
                hiddens = F.selu(F.max_pool2d(hiddens, 2))
                hiddens = hiddens.view([inp.size(0), -1])
                hiddens = F.selu(self.enc_linear_1(hiddens))
                hiddens = self.enc_linear_2(hiddens)
                hiddens = F.relu(hiddens)
                return hiddens
            for i in range(images.size(0)):
                h = hidden_pass(images[i])
                h = h.unsqueeze(0)
                hiddens.append(h)
            hiddens = torch.cat(hiddens, 0)
        else:
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
                       num_style,
                       eps,
                       cuda,
                       cnn,
                       input_dimensions):
        super(self.__class__, self).__init__()
        self.cnn = cnn
        self.input_dimensions = input_dimensions
        self.num_digits = num_digits
        self.digit_log_weights = Parameter(torch.zeros(num_digits)) if not cuda else Parameter(torch.zeros(num_digits).cuda())
        self.digit_temp = 0.66
        self.eps = eps
        self.style_mean = Parameter(torch.zeros(num_style)) if not cuda else Parameter(torch.zeros(num_style).cuda())
        self.style_log_std = Parameter(torch.zeros(num_style)) if not cuda else Parameter(torch.zeros(num_style).cuda())
        self.dec_hidden = nn.Sequential(
                            nn.Linear(num_style + num_digits, num_hidden),
                            nn.ReLU())
        if cnn:
            self.dec_linear_1 = nn.Linear(num_hidden, 160)
            self.dec_linear_2 = nn.Linear(160, num_pixels)
        else:
            self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())
        if cuda:
            self.dec_hidden.cuda()
            self.dec_image.cuda()

    def forward(self, images=None, q=None):
        p = probtorch.Trace()
        digits = p.concrete(self.digit_log_weights, self.digit_temp,
                            value=q['digits'],
                            name='digits')
        styles = p.normal(0.0, 1.0,
                          value=q['styles'],
                          name='styles')
        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        if self.cnn:
            def hidden_pass(inp):
                out = F.selu(self.dec_linear_1(inp))
                out = F.sigmoid(self.dec_linear_2(out))
                out = out.view([inp.size(0), self.input_dimensions[1], self.input_dimensions[2],self.input_dimensions[3]])
                return out
            images_mean = []
            for i in range(hiddens.size(0)):
                h = hidden_pass(hiddens[i])
                h = h.unsqueeze(0)
                images_mean.append(h)
            images_mean = torch.cat(images_mean, 0)
        else:    
            images_mean = self.dec_image(hiddens)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + self.eps) * x + 
                                  torch.log(1 - x_hat + self.eps) * (1-x)).sum(-1),
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
                 cuda,
                 logger,
                 cnn,
                 input_dimensions):
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
        self.logger = logger
        self.cnn = cnn
        self.input_dimensions = input_dimensions
        self.uuid = str(uuid.uuid4())
        self.enc = Encoder(num_pixels=dims[0],num_hidden=dims[2][0],num_digits=classes,num_style=dims[1],num_batch=batch_size,cuda=self.cuda,cnn=self.cnn,input_dimensions=self.input_dimensions)
        self.dec = Decoder(num_pixels=dims[0],num_hidden=dims[2][0],num_digits=classes,num_style=dims[1],eps=self.eps,cuda=self.cuda,cnn=self.cnn,input_dimensions=self.input_dimensions)
        self.optimizer =  torch.optim.Adam(list(self.enc.parameters())+list(self.dec.parameters()),
                              lr=self.lr,
                              betas=(self.beta1, self.beta2))

    def __backward_pass(self,data):
        epoch_elbo = 0.0
        self.enc.train()
        self.dec.train()
        N = 0
        for b in range(len(data)):
            (images, labels_onehot) = data[b]
            if images.size()[0] == self.batch_size:
                N += self.batch_size
                #images = images.view(-1, self.dims[0])
                if self.cuda:
                    images = images.cuda()
                    if labels_onehot is not None:
                        labels_onehot = labels_onehot.cuda()
                images = Variable(images)
                self.optimizer.zero_grad()
                if labels_onehot is not None:
                    labels_onehot = torch.clamp(labels_onehot,self.eps,1-self.eps)
                    labels_onehot = Variable(labels_onehot)
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
        history = []
        for e in range(self.epochs):
            train_start = time.time()
            train_elbo_labeled = self.__backward_pass(self.labeled)
            train_elbo_unlabeled = self.__backward_pass(self.unlabeled)
            _, train_acc = self.evaluate(self.X_labeled,Y_labeled)
            train_elbo = (train_elbo_labeled + train_elbo_unlabeled ) / 2.0
            train_end = time.time()
            print('[Epoch %d] Train: Labeled ELBO %.4f Unlabeled ELBO %.4f Accuracy %.4f (%ds) UUID %s' % (
                    e, train_elbo_labeled, train_elbo_unlabeled, train_acc,train_end - train_start, self.uuid[:4]))
            if self.logger is not None:
                self.logger.write('[Epoch %d] Train Labeled ELBO {%.4f} Train Unlabeled ELBO {%.4f} Train Accuracy {%.4f} UUID %s\n' % (
                    e, train_elbo_labeled, train_elbo_unlabeled, train_acc, self.uuid[:4]))
                self.logger.flush()
            history.append((train_elbo_labeled,train_elbo_unlabeled,train_acc))
        return np.array(history)

    def predict(self, X, batch_size=50, infer=True, verbose=0):
        self.enc.eval()
        self.dec.eval()
        epoch_elbo = 0.0
        epoch_correct = 0
        N = 0
        y_preds = []
        y_preds_one_hot = []
        X = bnn_dataset(X,None,self.batch_size)
        for b in range(len(X)):
            (images, labels_onehot) = X[b]
            N += self.batch_size
            #images = images.view(-1, self.dims[0])
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
                if verbose==1:
                    y_preds_one_hot.append(y_expect)
                _ , y_pred = y_expect.data.max(-1)
                if self.cuda:
                    y_pred = y_pred.cpu()
                y_preds.append(y_pred)
            else:
                if verbose==1:
                    y_preds_one_hot.append(q['digits'].value.data)
                _, y_pred = q['digits'].value.data.max(-1)
                if self.cuda:
                    y_pred = y_pred.cpu()
                y_preds.append(y_pred)
        if verbose == 1:
            y_preds_one_hot = torch.cat(y_preds_one_hot,0)
            return np.array(y_preds_one_hot)
        y_preds = torch.cat(y_preds,0)
        return epoch_elbo / N, y_preds #torch.LongTensor(np.array(y_preds)) if not self.cuda else torch.cuda.LongTensor(np.array(y_preds))

    def evaluate(self,X,y_true, infer =True):
        _, y_true = torch.max(y_true, 1)
        epoch_elbo, y_preds = self.predict(X,infer=infer)
        acc = (y_true.eq(y_preds.view_as(y_true))).sum()*1.0 / (len(y_preds) or 1.0)
        print('Validation: ELBO %.4f Accuracy %.4f UUID %s\n' % (epoch_elbo,acc,self.uuid[:4]))
        if self.logger is not None:
            self.logger.write('Validation ELBO {%.4f} Validation Accuracy {%.4f} UUID %s\n' % (epoch_elbo,acc,self.uuid[:4]))
            self.logger.flush()
        return epoch_elbo, acc

    def __elbo(self,q, p, alpha=0.1):
        if self.samples is None:
            return probtorch.objectives.importance.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
        else:
            return probtorch.objectives.importance.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)

class SSClassifier:

    def __init__(self, params, n_channels, img_rows, img_cols):
        self.params = params
        self.n_channels = n_channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.__setup_SSClassifier()

    def __setup_SSClassifier(self):
        self.ssl_vae =ssl_vae(**self.params, input_dimensions=(self.params['batch_size'],self.n_channels,self.img_rows,self.img_cols))

    def fit(self,X_labeled,Y_labeled,X_unlabeled):
        X_labeled = self.transform(X_labeled)
        X_unlabeled = self.transform(X_unlabeled)
        Y_labeled = self.transform(Y_labeled,features=False)
        self.X_labeled = X_labeled
        self.X_unlabeled = X_unlabeled
        self.Y_labeled = Y_labeled
        history = self.ssl_vae.train(self.X_labeled,self.Y_labeled,self.X_unlabeled)
        return history

    def predict(self,X,batch_size=50,infer=True,verbose=0):
        X = self.transform(X)
        return self.ssl_vae.predict(X,batch_size=batch_size,infer=infer,verbose=verbose)

    def evaluate(self,X,y_true,infer=True):
        X = self.transform(X)
        y_true = self.transform(y_true, features=False)
        return self.ssl_vae.evaluate(X,y_true, infer=infer)

    def transform(self,X,features=True):
        if features:
            if self.params['cnn']:
                if K.image_data_format() is not 'channels_first':
                    X = X.reshape(X.shape[0],self.n_channels,self.img_rows,self.img_cols)
            else:
                X = np.reshape(X,(X.shape[0],np.prod(X.shape[1:])))
        return torch.FloatTensor(X) if not self.params['cuda'] else torch.cuda.FloatTensor(X.tolist())

if __name__ == "__main__":
    training_data, validation_data, pool_data, testing_data = data_pipeline(valid_ratio=0.3, dataset='mnist')
    X_labeled,Y_labeled = validation_data #training_data
    X_unlabeled, Y_unlabeled = pool_data #validation_data
    train_size = 500
    pool_size = 10000
    idx = np.random.randint(0,len(X_labeled),train_size)
    X_labeled = X_labeled[idx]
    Y_labeled = Y_labeled[idx]
    idx = np.random.randint(0,len(X_unlabeled),pool_size)
    X_unlabeled = X_unlabeled[idx]
    Y_unlabeled = Y_unlabeled[idx]
    logger = open("log.log","w")
    params = {
        'classes':10, # MNIST and CIFAR10
        'batch_size':50,
        'lr':3e-3,
        'epochs':1,
        'dims':[784, 50, [600]], # 784: MNIST, 512: CIFAR10
        'samples':8,
        'beta1':0.9,
        'beta2':0.999,
        'eps':1e-9,
        'cuda':torch.cuda.is_available(),
        'logger':logger
    }
    model = SSClassifier(params)
    model.fit(X_labeled, Y_labeled, X_unlabeled)
    elbo ,acc = model.evaluate(X_unlabeled, Y_unlabeled)
    logger.close()