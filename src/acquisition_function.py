import math
import numpy as np
np.random.seed(1) # TODO
import os
import random
from scipy.stats import mode


def bald():
  pass

def maxentropy():
  pass

def varratio():
  pass

def segnet():
  pass




def acquisition_functions(acq_function, X_Pool_Dropout, num_classes, model, batch_size=128, dropout_iterations=10):
  """
  Returns the uncertain pool points according to different acquisition functions
  :param acq_function: the name of the acquisition function to use
  :param X_Pool_Dropout: the Pool set to use
  :param num_classes: the number of classes in the output of the model
  :param model: the model to use
  :param batch_size: the size of the batches to use when predicting
  :param dropout_iterations: the number of dropout iterations to use

  :return: the uncertainty of each point in the pool
  """
  if acq_function == 'bald':

    print ("BALD Acquisition Function")

    score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], num_classes))
    All_Entropy_Dropout = np.zeros(shape=X_Pool_Dropout.shape[0])

    for d in range(dropout_iterations):
        print ('Dropout Iteration', d)

        dropout_score = model.predict(X_Pool_Dropout, batch_size=batch_size, verbose=1)
        #computing Entropy_Average_Pi
        score_All = score_All + dropout_score
        #computing Average_Entropy
        dropout_score_log = np.log2(dropout_score)
        Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
        Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)
        All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout 


    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    G_X = Entropy_Average_Pi
    Average_Entropy = np.divide(All_Entropy_Dropout, dropout_iterations)
    uncertain_pool_points = Entropy_Average_Pi - Average_Entropy


  elif acq_function == 'maxentropy':

    print ("MaxEntropy Acquisition Function")

    score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], num_classes))
    for d in range(dropout_iterations):
      dropout_score = model.predict(X_Pool_Dropout,batch_size=batch_size, verbose=1)
      score_All = score_All + dropout_score

    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    uncertain_pool_points = Entropy_Average_Pi


  elif acq_function == "varratio":

    print ("Variation Ratio Acquisition Function")

    All_Dropout_Classes = np.zeros(shape=(X_Pool_Dropout.shape[0],1))

    for d in range(dropout_iterations):
      dropout_classes = model.predict_classes(X_Pool_Dropout,batch_size=batch_size, verbose=1)
      dropout_classes = np.array([dropout_classes]).T
      All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

    uncertain_pool_points = np.zeros(shape=(X_Pool_Dropout.shape[0]))
    for t in range(X_Pool_Dropout.shape[0]):
      L = np.array([0])
      for d_iter in range(dropout_iterations):
        L = np.append(L, All_Dropout_Classes[t, d_iter+1])            
      Predicted_Class, Mode = mode(L[1:])
      v = np.array(  [1 - Mode/float(dropout_iterations)])
      uncertain_pool_points[t] = v


  elif acq_function == "segnet":

    print ("Bayes Segnet Acquisition Function")

    All_Dropout_Scores = np.zeros(shape=(X_Pool_Dropout.shape[0], num_classes))
    for d in range(dropout_iterations):
      dropout_score = model.predict(X_Pool_Dropout,batch_size=batch_size, verbose=1)
      All_Dropout_Scores = np.append(All_Dropout_Scores, dropout_score, axis=1)

    All_Std = np.zeros(shape=(X_Pool_Dropout.shape[0],num_classes))
    uncertain_pool_points = np.zeros(shape=(X_Pool_Dropout.shape[0],1)) 

    for t in range(X_Pool_Dropout.shape[0]):
      for r in range(num_classes):
        L = np.array([0])
        L = np.append(L, All_Dropout_Scores[t, r+10])
        
        L_std = np.std(L[1:])
        All_Std[t,r] = L_std
        E = All_Std[t,:]
        uncertain_pool_points[t,0] = sum(E)

  else:
    raise Exception('Need a valid acquisition function')


  return uncertain_pool_points

