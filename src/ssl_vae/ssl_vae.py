from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader

class ssl_vae:
    
    def __init__(self,X_labeled,Y_labeled,X_unlabeled,
                 batch_size=10,
                 num_workers=1,
                 lr_m1=3e-5,
                 lr_m2=1e-2,
                 epochs_m1=50,
                 epochs_m2=50,
                 dims=[3, 32, [128]],
                 verbose=False):
        
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
        
        labeled=ssl_vae_dataset(X_labeled,Y_labeled)
        unlabeled=ssl_vae_dataset(X_labeled)
        
        self.unlabeled = DataLoader(unlabeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.labeled = DataLoader(labeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        
    def train(self):
        self.__create_m1() # create m1
        self.__train_m1()  # train m1
        self.__create_m2() # create m2
        self.__train_m2()  # train m2
    
    def __create_m1(self):
        
        from vae import VariationalAutoencoder
        from loss import VariationalInference, kl_divergence_normal
        
        self.model = VariationalAutoencoder(self.dims)
        
        
        def binary_cross_entropy(r, x):
            return F.binary_cross_entropy(r, x, size_average=False)
            
        self.objective = VariationalInference(binary_cross_entropy, kl_divergence_normal)
        self.optimizer_m1 = torch.optim.Adam(self.model.parameters(), lr=self.lr_m1)
        
    def __train_m1(self):
        
        for epoch in range(self.epochs_m1):
            for u in self.unlabeled:
                u = u['X']
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

         
    def __create_m2(self):
        
        import torch.nn as nn
        
        self.classifier_m2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10),
            nn.Softmax())
        
        self.optimizer_m2 = torch.optim.Adam(self.classifier_m2.parameters(), lr=self.lr_m2)
       
    def __train_m2(self):
        
        for epoch in range(self.epochs_m2):
            for l in self.labeled:

                x = l['X']
                y = l['y']
                x = Variable(x)
                y = Variable(y)
        
                _, (z, _, _) = self.model(x)
                logits = self.classifier_m2(z)
                self.loss = F.cross_entropy(logits, y)
        
                self.loss.backward()
                self.optimizer_m2.step()
                self.optimizer_m2.zero_grad()
                
            if self.verbose and epoch % 10== 0:
                l = self.loss.data[0]
                print("Epoch: {0:} loss: {1:.3f}".format(epoch, l))

    def predict(self,X):
        _, (z, _, _) = self.model(Variable(X))
        logits = self.classifier_m2(z)
        _, prediction = torch.max(logits, 1)
        return prediction.data
      
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

def test_data():    
    dummy_X_labeled = torch.FloatTensor(np.vstack([np.random.normal(4,5,size=[100,3]),
                                              np.random.normal(0,5,size=[100,3])]))

    z=np.hstack([np.repeat(0,100),np.repeat(1,100)])

    dummy_Y_labeled = torch.from_numpy(z)

    dummy_X_unlabeled = torch.FloatTensor(np.vstack([np.random.normal(4,5,size=[25,3]),
                                              np.random.normal(0,5,size=[25,3])]))

    dummy_ssl_vae =ssl_vae(dummy_X_labeled,dummy_Y_labeled,dummy_X_unlabeled,verbose=True)
    dummy_ssl_vae.train()

    z=dummy_ssl_vae.predict(dummy_X_unlabeled).numpy()
    print("Accuracy: "+str(np.mean((z==np.concatenate([np.repeat(0,25),np.repeat(1,25)])).astype(float))))