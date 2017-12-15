import sys
sys.path.append("semi-supervised")
sys.path.append("bayesbench")
sys.path.append("../")
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader

#from data.limitedmnist import LimitedMNIST
from torchvision.datasets import MNIST
from ss_utils import generate_label, onehot

# BayesNet
from bayesbench.benchmarking import regression
from bayesbench.networks.simple import SimpleMLP
from bayesbench.utils.metrics import accuracy
from bayesbench.methods import DeterministicMethod, MCDropoutMethod
from datatools import data_pipeline

# Keras CIFAR / MNIST utility
from keras import backend as K

def binary_cross_entropy(r, x):
    return F.binary_cross_entropy(r, x, size_average=False)

class bnn_dataset(Dataset):
    def __init__(self, x,y, m1=None):
        self.x = x
        self.y = y
        self.m1 = m1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.m1 is not None:
            x, y = self.x[idx], self.y[idx]
            if type(x) is not torch.autograd.Variable:
                x = torch.FloatTensor(x)
                x = torch.autograd.Variable(x)
            _, (z, _, _) = self.m1(x)
            z = z.data
            z = z.view(1,len(z))
            y = torch.FloatTensor(y)
            y = y.view(1,10)
            _, y = torch.max(y,1)
            return z,y
        else:
            return self.x[idx]

class ssl_vae:
    def __init__(self,
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
            self.logger = open("log.log","w")

        self.network_params = {
        'input_dims': self.dims[1],
        'output_dims': 10,
        'h_units':self.dims[2][0],
        'dropout':self.dropout,
        'n_layers': 2,
        'nonlinearity':lambda x:torch.nn.functional.relu(x,inplace=True),
        'out_non_linearity':nn.Softmax()
        }
        
    def train(self,X_labeled,Y_labeled,X_unlabeled):
        self.X_labeled = X_labeled
        self.X_unlabeled = X_unlabeled
        self.Y_labeled = Y_labeled
        self.unlabeled = bnn_dataset(self.X_unlabeled,None,None)
        self.labeled = bnn_dataset(self.X_labeled,self.Y_labeled,None)
        self.__create_m1() # create m1
        history_m1 = self.__train_m1()  # train m1
        self.__create_m2() # create m2
        history_m2 = self.__train_m2()  # train m2

        return history_m1, history_m2

    def __create_m1(self):
        
        from models.old_vae import VariationalAutoencoder
        from inference.old_loss import VariationalInference, kl_divergence_normal
        
        self.model = VariationalAutoencoder(self.dims)
            
        self.objective = VariationalInference(binary_cross_entropy, kl_divergence_normal)
        self.optimizer_m1 = torch.optim.Adam(self.model.parameters(), lr=self.lr_m1)
        
    def __train_m1(self):
        losses = []
        for epoch in range(self.epochs_m1):
            l = 0
            for u in self.unlabeled:
                #u = Variable(u)
                
                reconstruction, (_, z_mu, z_log_var) = self.model(u)
                # Equation 3
                self.L = self.objective(reconstruction, u, z_mu, z_log_var)
        
                self.L.backward()
                self.optimizer_m1.step()
                self.optimizer_m1.zero_grad()

                l = self.L.data[0]

            if self.verbose and epoch % 1== 0:
                print("Epoch: {0:} loss: {1:.3f}".format(epoch, l))
            if self.log and epoch % 10== 0:
                self.logger.write("Epoch: {0:} loss: {1:.3f}\n".format(epoch, l))

            for u in self.labeled:
                #u = Variable(u)
                
                reconstruction, (_, z_mu, z_log_var) = self.model(u)
                # Equation 3
                self.L = self.objective(reconstruction, u, z_mu, z_log_var)
        
                self.L.backward()
                self.optimizer_m1.step()
                self.optimizer_m1.zero_grad()
        
                l = self.L.data[0]

            if self.verbose and epoch % 1== 0:
                print("Epoch: {0:} loss: {1:.3f}".format(epoch, l))
            if self.log and epoch % 10== 0:
                self.logger.write("Epoch: {0:} loss: {1:.3f}\n".format(epoch, l))

            losses.append(l)
        
        return {'loss':losses,'acc':None} # No accuracy for autoencoders


         
    def __create_m2(self):

        self.classifier_m2 = MCDropoutMethod(
                SimpleMLP(**self.network_params),
                F.cross_entropy,
                N=len(self.labeled),
                dropout=self.dropout,
                tau=0.1,
                metrics=accuracy,
                optimizer_params={'lr':self.lr_m2})  
        self.bnn_train = bnn_dataset(self.X_labeled,self.Y_labeled,self.model)
       
    def __train_m2(self):

        history = self.classifier_m2.train(self.bnn_train,
            val_dataloader=None,
            epochs=self.epochs_m2,
            log_every_x_batches=-1,
            verbose=False,
            approx_train_metrics=True)

        return history

    def predict(self,X,convert_logits):
        ys = []
        for i in range(len(X)):
            x = X[i]
            if type(x) is not torch.autograd.Variable:
                x = torch.FloatTensor(x)
                x = torch.autograd.Variable(x)
            _, (z, _, _) = self.model(x)
            z = z.data
            z = z.view(1,len(z))
            y_logits = self.classifier_m2.predict(z,convert_logits=convert_logits)
            #_, y_logits = torch.max(x,1)
            ys.append(y_logits.data.numpy())
        return np.array(ys)[:,0,:]

class SSClassifier:

    def __init__(self, params, deep_extractor="resnet18", convnet=True):
        """
        X_labeled: n_examples x 3 x height x width
        y_labeld: n_examples x n_classes
        X_unlabeled: n_examples x 3 x height x width
        deep_extractor: alexnet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resent50, resnet101, resnet152, squeezenet1_0, squeezenet1_1, densenet121, 
            densenet169, densenet161, densenet201 or inception_v3
        """
        self.convnet = convnet
        self.params = params
        self.n_channels = 3
        self.img_rows = 32 # CIFAR
        self.img_cols = 32 # CIFAR
        self.deep_extractor = deep_extractor

        self.__setup_SSClassifier()


    def __setup_extractor(self,model_name):
        from torchvision import models
        extractor = models.__dict__[model_name](pretrained=True)
        for param in extractor.parameters():
            param.requires_grad = False

        extractor = nn.Sequential(*list(extractor.children())[:-1])

        return extractor

    def __setup_SSClassifier(self):
        self.ssl_vae =ssl_vae(**self.params)

    def __extract_features(self,X):
        from skimage.transform import resize

        img_rows = 224
        img_cols = 224

        processed_X = []
        for i in range(len(X)):
            tmp = []
            for ch in range(len(X[i])):
                im = X[i][ch]
                im = resize(im,(img_rows,img_cols),mode = "reflect")
                tmp.append(im)
            processed_X.append(tmp)

        X = np.array(processed_X)
        X = torch.autograd.Variable(torch.FloatTensor(X))

        features = self.extractor(X)
        return features.view(features.shape[0],-1)

    def fit(self,X_labeled,Y_labeled,X_unlabeled):

        if self.convnet and K.image_data_format() is not 'channels_first':
            X_labeled = X_labeled.reshape(X_labeled.shape[0],self.n_channels,self.img_rows,self.img_cols)
            X_unlabeled = X_unlabeled.reshape(X_unlabeled.shape[0],self.n_channels,self.img_rows,self.img_cols)
        self.n_classes = Y_labeled.shape[1]

        if self.convnet:
            self.extractor = self.__setup_extractor(self.deep_extractor)
            self.X_labeled = self.__extract_features(X_labeled)
            self.X_unlabeled = self.__extract_features(X_unlabeled)
        else:
            X_labeled = X_labeled.reshape(X_labeled.shape[0],np.prod(X_labeled.shape[1:]))
            X_unlabeled = X_unlabeled.reshape(X_unlabeled.shape[0],np.prod(X_unlabeled.shape[1:]))
            self.X_labeled = torch.autograd.Variable(torch.FloatTensor(X_labeled))
            self.X_unlabeled = torch.autograd.Variable(torch.FloatTensor(X_unlabeled))
        self.Y_labeled = Y_labeled
        history = self.ssl_vae.train(self.X_labeled,self.Y_labeled,self.X_unlabeled)

        return history

    def predict(self,X):
        if self.convnet and K.image_data_format() is not 'channels_first':
            X = X.reshape(X.shape[0],self.n_channels,self.img_rows,self.img_cols)
        if self.convnet:
            X = self.__extract_features(X)
        else:
            X = X.reshape(X.shape[0],np.prod(X.shape[1:]))
            X = torch.autograd.Variable(torch.FloatTensor(X))
        return self.ssl_vae.predict(X,convert_logits=False)

    def evaluate(self,X,y_true):
        y_pred = self.predict(X)
        y_true = np.argmax(y_true,axis=1)
        y_pred = Variable(torch.FloatTensor(y_pred))
        y_true = Variable(torch.LongTensor(y_true))
        loss = F.cross_entropy(y_pred,y_true)
        _, logits = torch.max(y_pred,1)
        y_true = y_true.type(torch.LongTensor)
        acc = logits.eq(y_true.view_as(logits)).cpu().float().mean()
        return loss, acc

if __name__ == "__main__":
    training_data, validation_data, pool_data, testing_data = data_pipeline(valid_ratio=0.3, dataset='mnist')
    X_labeled,Y_labeled = validation_data #training_data
    X_unlabeled, Y_unlabeled = pool_data #validation_data
    params = {
        'batch_size':50,
        'num_workers':1,
        'lr_m1':3e-5,
        'lr_m2':1e-2,
        'epochs_m1':1,
        'epochs_m2':1,
        'dims':[784, 50, [600]], # 784: MNIST, 512: CIFAR10
        'verbose':True,
        'log':True,
        'dropout':0.1
    }
    model = SSClassifier(params, "resnet18", convnet = False)
    history = model.fit(X_labeled, Y_labeled, X_unlabeled)
    idx = np.random.randint(0,len(X_unlabeled),10)
    X_unlabeled = X_unlabeled[idx]
    trials = 5
    accs = []
    y_true = np.argmax(Y_unlabeled[idx],axis=1)
    from sklearn.metrics import accuracy_score
    for n in range(trials):
        y_pred = model.predict(X_unlabeled)
        accs.append(accuracy_score(y_true,y_pred))
    mean = lambda x: sum(x)/len(x)
    print("Accuracy")
    print("Mean: %.5f Variance: %.5f" % (mean(accs),
        sum(list(map(lambda x:(x-mean(accs))**2,accs)))/(len(accs)-1)
        ))
    if model.ssl_vae.log:
        model.ssl_vae.logger.write("Mean: %.5f Variance: %.5f" % (mean(accs),
        sum(list(map(lambda x:(x-mean(accs))**2,accs)))/(len(accs)-1)
        ))
    model.ssl_vae.logger.close()

"""
import pickle
val=pickle.load(open("bnn_vae.pickle","rb"))
print(torch.sum(val[0]))
"""