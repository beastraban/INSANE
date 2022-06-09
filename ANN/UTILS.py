# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:05:35 2021

@author: beast
"""

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from joblib import load
import sys
import os
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

import signal


class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
    self.kill_now = True
    
class GracefullKillerOut(GracefulKiller):
    def __init__(self,orig_stdout):
        super().__init__()
        self.orig_stdout=orig_stdout
    
    def exit_gracefully(self, *args):
        sys.stdout=self.orig_stdout
        super().exit_gracefully(args)
        

class FIFO(list):
    
    def __init__(self):
        list.__init__(self)
        return None
    
    def put(self,X):
        self.insert(len(self)+1,X)
        return None
    
    def pop(self):
        return(super().pop(0))
    
class ObservableTransformer(FunctionTransformer):
    
    def __init__(self,RANGES=[[0,1]]):
        FunctionTransformer.__init__(self)
        self.ranges=FIFO()
        self.RANGES=RANGES
        return None
    
    def fit_transform(self,X,*RANGES):
        if RANGES:
            self.RANGES=RANGES
        else:
            RANGES=self.RANGES
        X_new=X.copy()
        
        for i in range(len(X[0,:])):
            maxNS=max(X[:,i])
            minNS=min(X[:,i])
            if i>=len(RANGES):
                RANGE=RANGES[-1]
            else:
                RANGE=RANGES[i]
            
            NS_new_std=((X[:,i]-minNS)/(maxNS-minNS))
            NS_new=NS_new_std*(RANGE[1]-RANGE[0])+RANGE[0]
            #Store original ranges for inverse transform
            self.ranges.put([minNS,maxNS])
            X_new[:,i]=NS_new
              
        return(X_new)
    
    def shape(self,X):
        #print(type(X))
        a=np.array([1])
        if X is None:
            return(0)
        else:
            if type(X) ==  type(a):
                return(np.shape(X))
            else:
                x=np.array(X)
                return(np.shape(X))
    
    def inverse_transform(self,X):
        X_new=X.copy()
        q=self.ranges.copy()
        RANGES=self.RANGES
        s=self.shape(X_new)
        l=s[1]
        s=s[0]
        if s==1:
            index=[0]
        else:
            index=range(l)
        for i in index:
            #print(i)
            if s==1:
                #print(X_new)
                x_new=X_new[0]
                #print(x_new)
                x=X[0]
                #print(x)
                for j in range(l):
                    #print(j)
                    if j>=len(RANGES):
                        RANGE=RANGES[-1]
                    else:
                        RANGE=RANGES[j]
                    maxNS=RANGE[1]
                    minNS=RANGE[0]
                    ogRANGE=q.pop(0)
                    NS_new_std=((x[j]-minNS)/(maxNS-minNS))
                    NS_new=NS_new_std*(ogRANGE[1]-ogRANGE[0])+ogRANGE[0]
                    x_new[j]=NS_new
                    X_new=[x_new]
            else:  
                if i>=len(RANGES):
                    RANGE=RANGES[-1]
                else:
                    RANGE=RANGES[i]
                maxNS=RANGE[1]
                minNS=RANGE[0]
                ogRANGE=q.pop(0)
                NS_new_std=((X[:,i]-minNS)/(maxNS-minNS))
                NS_new=NS_new_std*(ogRANGE[1]-ogRANGE[0])+ogRANGE[0]
                X_new[:,i]=NS_new
        return(X_new) 

    def transform(self,X):
        X_new=X.copy()
        q=self.ranges.copy()
        RANGES=self.RANGES
        for i in range(len(X[0,:])):
            if i>=len(RANGES):
                RANGE=RANGES[-1]
            else:
                RANGE=RANGES[i]
            TrRange=q.pop(0)
            maxNS=TrRange[1]
            minNS=TrRange[0]
            
            NS_new_std=((X[:,i]-minNS)/(maxNS-minNS))
            NS_new=NS_new_std*(RANGE[1]-RANGE[0])+RANGE[0]
            X_new[:,i]=NS_new
        return(X_new) 
    
class Validator:
        
    def __init__(self,DIR="./"):
        self.DIR=str(DIR)
        
        return None
    
    
    def validate(self,h5_filename,Xencoder,Yencoder):
            """
            Parameters
            ----------
            h5_filename : TYPE
                DESCRIPTION.
            Xencoder : TYPE
                DESCRIPTION.
            Yencoder : TYPE
                DESCRIPTION.
            data_set : TYPE
                DESCRIPTION.
    
            Returns
            -------
            None.
    
            """  
            DIR=self.DIR

            if os.path.exists(DIR+h5_filename):
            #Make the estimator
                
                estimator=KerasRegressor(build_fn=self.model_build_none)
                estimator.model=load_model(DIR+h5_filename)
                modelAns="Model OK"
            else:
                modelAns="model does not exist at specified location"
            # get the X and Y preprocessors
            if os.path.exists(DIR+Xencoder):
                sc=load(DIR+Xencoder)
                XencoderAns="X data encoder OK"
            else:
                XencoderAns="X data encoder does not exist specified location"
            if os.path.exists(DIR+Yencoder):
                sc1=load(DIR+Yencoder)
                YencoderAns="Y data encoder OK"
            else:
                YencoderAns="Y data encoder does not exist specified location"
            
            
            return [modelAns,XencoderAns,YencoderAns]
            
    def model_build_none(self):
        ann = tf.keras.models.Sequential()
        return ann
            
class Builder:
    def __init__(self):
        return None
    
    def baseline_model(self,layers,nodes):
    
        def bm():
        
            ann = tf.keras.models.Sequential()
            ann.add(tf.keras.layers.Dense(nodes, input_dim=3, kernel_initializer='normal', activation='relu'))
            #ann.add(tf.keras.layers.Dense(nodes, kernel_initializer='normal', activation='relu'))
            for i in range(layers): 
                ann.add(LeakyReLU(alpha=0.01))
            #ann.add(LeakyReLU(alpha=0.01))
            #ann.add(LeakyReLU(alpha=0.01))
            ann.add(tf.keras.layers.Dense(2, kernel_initializer='normal',activation='linear'))
            #ann.compile(optimizer = 'adam', loss='mse', metrics = ['mse'])
            ann.compile(optimizer = 'adam', loss='mse' ,metrics = ['mse'])#, run_eagerly=True)
            return ann
        return bm
        
class Saver_Loader:
    def __init__(self):
        return None
    
    def loadEstimator(self,DIR="./"):
        i=1
        j=1
        builder=Builder()
        #DIR=str("./InsaANNe_LtoObs/")
        filepath=DIR+'insANN.h5'
        #pickle_file=open(DIR+"ij.pkl",'rb')
        #[i,j]=pickle.load(pickle_file)
        estimator = KerasRegressor(build_fn=builder.baseline_model(i,j), epochs=300, batch_size=32, verbose=0,validation_split=0.2)
        model=tf.keras.models.load_model(filepath,custom_objects=None,compile=True)
        estimator.model=model
        estimator.model.load_weights(DIR+'weights.h5')
        sc=load(DIR+'insScalerX.bin')
        sc1=load(DIR+'insScalerY.bin')
        sys.stdout.write("\033[1;31m")
        print("remember to pre-process!")
        sys.stdout.write('\033[0m')
        return estimator, sc , sc1
    
    def saveModel(self,model,DIR='./'):
        filepath=DIR+'insANN.h5'
        #filepath1=DIR+'insANN1.h5'
        #tf.keras.models.save_model(model,filepath,overwrite=True)
        model.save(filepath,overwrite=True)
        model.save_weights(DIR+'weights.h5',overwrite=True)
        return None