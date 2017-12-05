import sys
sys.path.append("semi-supervised")
sys.path.append("bayesbench")
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader

from data.limitedmnist import LimitedMNIST
from torchvision.datasets import MNIST
from utils import generate_label, onehot

# BayesNet
from bayesbench.benchmarking import regression
from bayesbench.networks.simple import SimpleMLP
from bayesbench.utils.metrics import accuracy
from bayesbench.methods import DeterministicMethod, MCDropoutMethod

def binary_cross_entropy(r, x):
            return F.binary_cross_entropy(r, x, size_average=False)
class bnn_dataset(Dataset):
    def __init__(self, dl, m1):
        self.dataset = dl.dataset
        self.ds = self.dataset
        self.m1 = m1

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds.mnist.data[idx], self.ds.mnist.target[idx]
        x = torch.FloatTensor((x/255).astype(np.float32))
        y = int(np.asscalar(y))
        x = torch.autograd.Variable(x)
        _, (z, _, _) = self.m1(x)
        z = z.data
        z = z.view(1,len(z))
        y = generate_label(1, y, nlabels=10)
        y = y.type(torch.FloatTensor)
        return z,y
class ssl_vae:
    
    def __init__(self,labeled,unlabeled,#,X_labeled,Y_labeled,X_unlabeled,
                 batch_size=64,
                 num_workers=1,
                 lr_m1=3e-5,
                 lr_m2=1e-2,
                 epochs_m1=5,
                 epochs_m2=5,
                 dims=[784, 32, [600,600]],
                 verbose=False,
                 log=True,
                 dropout=0):
        
        """
        Args:
            X_labeled (array): for now accepts a flat tensor [observations x k] [observations x height x weight x channels]
            Y_labeled (array): [observations x 1]
            X_unlabeled (array): [observations x height x weight x channels]
            batch_size (int): size of batch sample
            num_workers (int): number of workers on each dataset
            lr_m1,lr_m2 (int,int): learning rates for model1 and model2
            epochs_m1,epochs_m2 (int,int): number of epochs to train model1 and model2 for
            dims (list): has the format [x_dim, z_dim, [h_dim]] , creates the VAE
            verbose (boolean): prints the training process of the VAE
        """
        #flatten_bernoulli = transforms.Lambda(lambda img: transforms.ToTensor()(img).view(-1).bernoulli())
       
        self.lr_m1 = lr_m1
        self.lr_m2 = lr_m2
        self.epochs_m1 = epochs_m1
        self.epochs_m2 = epochs_m2
        self.dims = dims
        self.verbose = verbose
        self.log = log
        self.dropout = dropout
        if self.log:
            self.logger = open("log","w")

        #labeled=ssl_vae_dataset(X_labeled,Y_labeled)
        #unlabeled=ssl_vae_dataset(X_labeled)
        
        #self.unlabeled = DataLoader(unlabeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #self.labeled = DataLoader(labeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.unlabeled = unlabeled
        self.labeled = labeled

        self.network_params = {
        'input_dims': self.dims[1],
        'output_dims': 10,
        'h_units':self.dims[2][0],
        'dropout':self.dropout,
        'n_layers': 2,
        'nonlinearity':lambda x:torch.nn.functional.relu(x,inplace=True),
        'out_non_linearity':nn.Softmax()
        }
        
    def train(self):
        self.__create_m1() # create m1
        self.__train_m1()  # train m1
        self.__create_m2() # create m2
        self.__train_m2()  # train m2
    
    def __create_m1(self):
        
        from models.old_vae import VariationalAutoencoder
        from inference.old_loss import VariationalInference, kl_divergence_normal
        
        self.model = VariationalAutoencoder(False, self.dims)
            
        self.objective = VariationalInference(binary_cross_entropy, kl_divergence_normal)
        self.optimizer_m1 = torch.optim.Adam(self.model.parameters(), lr=self.lr_m1)
        
    def __train_m1(self):
        
        for epoch in range(self.epochs_m1):
            for u,_ in self.unlabeled:
                #u = u['X']
                u = Variable(u)
                
                reconstruction, (_, z_mu, z_log_var) = self.model(u)
                # Equation 3
                self.L = self.objective(reconstruction, u, z_mu, z_log_var)
        
                self.L.backward()
                self.optimizer_m1.step()
                self.optimizer_m1.zero_grad()
        
            if self.verbose and epoch % 10== 0:
                l = self.L.data[0]
                print("Epoch: {0:} loss: {1:.3f}".format(epoch, l))
            if self.log and epoch % 10== 0:
                l = self.L.data[0]
                self.logger.write("Epoch: {0:} loss: {1:.3f}\n".format(epoch, l))
        for epoch in range(self.epochs_m1):
            for u,_ in self.labeled:
                u = Variable(u)
                
                reconstruction, (_, z_mu, z_log_var) = self.model(u)
                # Equation 3
                self.L = self.objective(reconstruction, u, z_mu, z_log_var)
        
                self.L.backward()
                self.optimizer_m1.step()
                self.optimizer_m1.zero_grad()
        
            if self.verbose and epoch % 10== 0:
                l = self.L.data[0]
                print("Epoch: {0:} loss: {1:.3f}".format(epoch, l))
            if self.log and epoch % 10== 0:
                l = self.L.data[0]
                self.logger.write("Epoch: {0:} loss: {1:.3f}\n".format(epoch, l))

         
    def __create_m2(self):
        
        if not self.dropout:
            self.classifier_m2 = nn.Sequential(
            nn.Linear(self.dims[1], self.dims[2][0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.dims[2][0], 10),
            nn.Softmax())
            self.optimizer_m2 = torch.optim.Adam(self.classifier_m2.parameters(), lr=self.lr_m2)
        else:
            self.classifier_m2 = MCDropoutMethod(
                    SimpleMLP(**self.network_params),
                    F.binary_cross_entropy,
                    N=len(self.labeled),
                    dropout=0.5,
                    tau=0.1,
                    #metrics=accuracy,#lambda true,pred:accuracy(true,generate_label(1, int(pred), nlabels=10)),
                    optimizer_params={'lr':self.lr_m2})  
       
    def __train_m2(self):
        if not self.dropout:
            for epoch in range(self.epochs_m2):
                for x,y in self.labeled:

                    x = Variable(x)
                    y = Variable(y)
            
                    _, (z, _, _) = self.model(x)
                    logits = self.classifier_m2(z)
                    y = y.type(torch.FloatTensor)
                    self.loss = F.binary_cross_entropy(logits, y)
            
                    self.loss.backward()
                    self.optimizer_m2.step()
                    self.optimizer_m2.zero_grad()
                    
                if self.verbose and epoch % 10== 0:
                    l = self.loss.data[0]
                    print("Epoch: {0:} loss: {1:.3f}".format(epoch, l))
                if self.log and epoch % 10== 0:
                    l = self.loss.data[0]
                    self.logger.write("Epoch: {0:} loss: {1:.3f}\n".format(epoch, l))
        else:
            bnn_train = bnn_dataset(self.labeled, self.model)
            self.classifier_m2.train(bnn_train,
                val_dataloader=None,
                epochs=self.epochs_m2,
                log_every_x_batches=-1,
                verbose=self.verbose,
                approx_train_metrics=False)


    def predict(self,X):
        if not self.dropout:
            if type(X) is DataLoader:
                x, y = next(iter(X))
            else:
                x = X
            _, (z, _, _) = self.model(Variable(x))
            _, y_logits = torch.max(self.classifier_m2(z), 1)
            _, y = torch.max(y,1)
            if type(X) is DataLoader:
                y_logits = y_logits.data
                acc = sum(y==y_logits)/len(y)
                print(acc)
                return y,y_logits
            else:
                return y_logits
        if type(X) is DataLoader:
            acc = 0
            for i in range(len(X)):
                x, y = X.dataset.mnist.data[i], X.dataset.mnist.target[i]
                x = torch.FloatTensor((x/255).astype(np.float32))
                y = int(np.asscalar(y))
                x = torch.autograd.Variable(x)
                _, (z, _, _) = self.model(x)
                z = z.data
                z = z.view(1,len(z))
                y = generate_label(1, y, nlabels=10)
                y = y.type(torch.FloatTensor)
                y_logits = self.classifier_m2.predict(z)
                _, y = torch.max(y,1)
                #_, y_logits = torch.max(x,1)
                acc += int(y==y_logits.data)
            acc = acc /len(X)
            return acc
class ssl_vae_dataset(Dataset):

    def __init__(self, X, y=None, transform=None):
        """
        Args:
            X (matrix): input data
            y (vector): labels for data. If None, assume unlabeled
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = X
        self.size = len(self.data_frame)
        self.labels = y
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        if self.labels is not None: # labeled
            sample = {'X': self.data_frame[idx] , 'y': self.labels[idx]}
        else: # unlabeled
            sample = {'X': self.data_frame[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

labels = np.arange(10)
n = len(labels)

# Load in data
mnist_lab = LimitedMNIST('./', train=True, transform=torch.bernoulli, target_transform=onehot(n), digits=labels, fraction=0.1)
mnist_ulab = LimitedMNIST('./', train=True, transform=torch.bernoulli, target_transform=onehot(n), digits=labels, fraction=0.8)
mnist_val = LimitedMNIST('./', train=False, transform=torch.bernoulli, target_transform=onehot(n), digits=labels)

# Unlabelled data
unlabelled = torch.utils.data.DataLoader(mnist_ulab, batch_size=100, shuffle=True, num_workers=2)

# Validation data
validation = torch.utils.data.DataLoader(mnist_val, batch_size=1000, shuffle=False, num_workers=1)

# Labelled data
labelled = torch.utils.data.DataLoader(mnist_lab, batch_size=100, shuffle=True, num_workers=2)

dummy_ssl_vae =ssl_vae(labelled,unlabelled,
                 batch_size=64,
                 num_workers=2,
                 lr_m1=3e-5,
                 lr_m2=1e-2,
                 epochs_m1=200,
                 epochs_m2=200,
                 dims=[784, 50, [600]],
                 verbose=True,
                 log=True,
                 dropout=0.05)
dummy_ssl_vae.train()
print("Training done")
trials = 5
accs = []
for n in range(trials):
    acc = dummy_ssl_vae.predict(validation)
    accs.append(acc)
mean = lambda x: sum(x)/len(x)
print("Accuracy")
print("Mean: %.5f Variance: %.5f" % (mean(accs),
    sum(list(map(lambda x:(x-mean(accs))**2,accs)))/(len(accs)-1)
    ))
if dummy_ssl_vae.log:
    dummy_ssl_vae.logger.write("Mean: %.5f Variance: %.5f" % (mean(accs),
    sum(list(map(lambda x:(x-mean(accs))**2,accs)))/(len(accs)-1)
    ))
dummy_ssl_vae.logger.close()
"""
import pickle
val=pickle.load(open("bnn_vae.pickle","rb"))
print(torch.sum(val[0]))
"""