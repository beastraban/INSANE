# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:00:32 2020

@author: beast
"""
#runfile('/home/spawn/DATA/Dropbox/PEGASUS/INSANNE/UTILS.py', wdir='/home/spawn/DATA/Dropbox/PEGASUS/INSANNE')
#import silence_tensorflow.auto
import logging, os 
import numpy as np
import csv
import pandas as pd
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from numpy import genfromtxt
import platform
import multiprocessing as mp





import signal


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from matplotlib import pyplot as plt
import time as Time
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import sys
from UTILS import *

from scipy.io import loadmat
import InsANNE_MCMC_Utils as MCMC
from InsANNE_MCMC_Utils import posterior_dist



show=(str(sys.argv[1])=='show')
print(sys.argv[1])


logging.disable(logging.WARNING)

mode='MCMC'
DIR=str("./InsaANNe_LtoObs/")
BurnIn=100000
threshold=0.005
NEW=False            #Start a new chain or continue from last?
iterations=500
ChainNum=4

G = lambda _mu, _x,_sigma : (1/np.sqrt(2*np.pi*_sigma**2))*np.exp(-0.5*((_mu-_x)/_sigma)**2) 

_G=np.vectorize(G)


orig_stdout = sys.stdout

OUT=False

 

def checkExit():
    if OUT:
        unmute()
        sys.exit(1)


def handler(signum, frame):
    global OUT
    
    msg = "Ctrl-c was pressed."#" Do you really want to exit? y/n "
    print(msg, end="", flush=True)

    if mode!='MCMC':
        unmute()
        sys.exit(1)
    else:
        unmute()
        print(msg, end="", flush=True)
        mute()
        OUT=True

 
signal.signal(signal.SIGINT, handler)



def unmute():
    sys.stdout = orig_stdout

def mute():
    sys.stdout = open(os.devnull, 'w') 

def SmoothGauss(ARR:np.array,sigma:float,iterations:int=0):
    if iterations<1:
        arr=np.zeros(shape=ARR.shape,dtype=float)
        SIG=np.round(sigma)
        for i in range(len(ARR)):
            nEnd=min(i+SIG,len(ARR)-1)
            nStart=max(0,i-SIG)
            if nEnd+SIG>len(ARR)-1:
                elNum=int(len(ARR)-nStart) 
                _x=np.linspace(nStart,len(ARR)-1,elNum)
            else:
                elNum=int(nEnd-nStart) 
                _x=np.linspace(nStart,nEnd,elNum+1)
    
            x=ARR[_x.astype(int)]
    
            arr[i]=np.sum(np.multiply(_G(i,_x,SIG),x))
        
        return arr
    else:
        return SmoothGauss(ARR,sigma,iterations-1)

def SmoothNorm(ARR,sigma,iterations=0):
    arr=SmoothGauss(ARR,sigma,iterations)
    arr=arr/max(arr)
    return(arr)


def CombineChains(NofChains:int=0):
    if NofChains==0:
        CHAIN,ACC_RATE,LNPROB=getChain(0)
        return(CHAIN,ACC_RATE,LNPROB)
    else:
        dfs=[]
        for k in range(NofChains):
            dfs.append(getDF(k+1))
        my_data = pd.concat(dfs).sort_index().reset_index(drop=True)

    
    LENGTH=my_data.shape[0]
    my_data=my_data.values
        
    return(my_data[:,0:3], my_data[:,3:4],my_data[:,4:5],int(LENGTH))
                

def MakeChain(x0,Number:int=0,NEW=False,ACCRATE=0,counter=1,Q=None):
    if (os.path.isfile('chains_{0}.csv'.format(Number))and(not NEW)):
        if counter==1:
            x0,ACC_RATE,LNPROB=getChain(Number)
            LENGTH=len(x0[:,0])
            x0=x0[-1]
            ACC_RATE=ACC_RATE[-1].item()
            LNPROB=LNPROB[-1].item()
            
        else:
            x0,ACC_RATE,LNPROB,LENGTH=getLast(Number)

    First=False
    if counter==1 and NEW:
        First=True
        acc_NUM=0
        acc_RATE=0
        LENGTH=0
        [CHAIN, ACC_RATE, LNPROB]=MCMC.mh_sampler(x0, lnprob_fn=POST, prop_fn=G3P,acc_RATE=ACCRATE,acc_NUM=acc_NUM, prop_fn_kwargs={}, iterations=iterations)

    else:
        print(x0)
        acc_NUM=np.round(ACC_RATE*LENGTH)
        [CHAIN, ACC_RATE, LNPROB]=MCMC.mh_sampler(x0, lnprob_fn=POST, prop_fn=G3P,acc_RATE=ACCRATE,acc_NUM=acc_NUM, prop_fn_kwargs={}, iterations=iterations)

    ACCRATE=ACC_RATE[-1]
    saveChain(CHAIN,ACC_RATE,LNPROB,LENGTH,Number,First=First)
    GL_stat=[]
    for jj in range(len(CHAIN[-1])):
       GL_stat.append(np.mean(CHAIN[:,jj]))
       GL_stat.append(np.var(CHAIN[:,jj]))
    Q.put(GL_stat)
    Q.put(len(CHAIN[:,0]))
        


def getChain(Number:int=0):
    if Number==0:
        my_data = pd.read_csv('chains.csv')
    else:
        my_data = pd.read_csv('chains_{0}.csv'.format(Number))
    my_data=my_data.values
        
    return(my_data[:,0:3], my_data[:,3:4],my_data[:,4:5])

def getDF(Number:int=0):
    if Number==0:
        my_data = pd.read_csv('chains.csv')
    else:
        my_data = pd.read_csv('chains_{0}.csv'.format(Number))
    
    return(my_data)

def getLast(Number:int=0):
    if Number==0:
        lengthName='chainsL.csv'
        my_data = pd.read_csv('chains.csv')
    else:
        lengthName='chainsL_{0}.csv'.format(Number)
        my_data = pd.read_csv('chains_{0}.csv'.format(Number))
    my_data=my_data.values
    
    X0=my_data[-1,:]
    LENGTH=genfromtxt(lengthName,dtype=(int),skip_header=1).item()

    return(X0[0:3], X0[3:4].item(),X0[4:5].item(),LENGTH)




def saveChain(chain,acceptance,lnprob,LENGTH,Number:int=0,First=False,ALL=False):
    #make sure to get only the appended chain
    fieldnames=['l2','l3','l4','acceptance','lnprob']
    if Number==0:
        filename='chains.csv'
        lengthName='chainsL.csv'
    else:
        filename='chains_{0}.csv'.format(Number)
        lengthName='chainsL_{0}.csv'.format(Number)
    if not ALL and not First:
        write='a'
        header=False
    else:
        write='w'
        header=["l2", "l3", "l4","acceptance","lnprob"]
    
    
    _chain=np.zeros((chain.shape[0],chain.shape[1]+2))
    _chain[:,0:3]=chain
    _chain[:,3]=acceptance
    _chain[:,4]=lnprob

    #use dataframes
    df=pd.DataFrame(_chain,columns =["l2", "l3", "l4","acceptance","lnprob"])
      
    
    df.to_csv(filename,mode=write,header=header,index=False)

    with open(lengthName,'w', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['length'])
        writer.writeheader() 
        
        if not ALL:
            writer.writerow({'length':(LENGTH+len(chain[:,0]))})
        else:
            writer.writerow({'length':(len(chain[:,0]))})
    if Number==0:
        checkExit()

    return None



def animate1():
    #plt.figure(0)
    plt.cla()
    if os.path.isfile('chains.csv'):
        CHAIN,ACC,LNPR=getChain(0)
        x,bin,p=plt.hist(CHAIN[:,1],bins=1000,)
        axes=plt.gca()
        for item in p:
            
            item.set_height(item.get_height()/max(x))
        axes.set_ylim(top=1.1)
        plt.show()
        plt.pause(0.05)

    
    

def Histogram(CHAIN,bins):
    for i in range(len(CHAIN[0][:])):
        plt.figure(i)
        plt.pause(0.05)
        
        x,bin,p=plt.hist(CHAIN[:,i],bins=1000,)
        axes=plt.gca()
        for item in p:
            
            item.set_height(item.get_height()/max(x))
        axes.set_ylim(top=1.1)
        plt.pause(0.05)
        plt.tight_layout()
        plt.show()  
        return None

def myLossFn(y_pred,y_act,name='weightedLoss'):
        
        differ=tf.math.subtract(y_pred,y_act)
        weights=tf.constant([[1,1,1]],dtype=float)
        wD=tf.math.multiply(weights,differ)
        sqr=tf.math.square(wD)
        LEN=len(differ.numpy())
        loss=sqr/LEN
        return(loss)

builder=Builder()    
HELPER=Saver_Loader()


print(tf.__version__)


if mode=='train':
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    
    
    dataset = pd.read_csv('./INSANNE_4_7_around0965.csv')
    X = dataset.iloc[:, 0:3].values
    y = dataset.iloc[:, 3:5].values
    
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    sc=StandardScaler()

    
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    
    

    sc1=StandardScaler()
    
    y_train = sc1.fit_transform(y_train)

    
    
    dump(sc,DIR+'insScalerX.bin',compress=True)
    dump(sc1,DIR+'insScalerY.bin',compress=True)
    

    
    MSEs=[]
    mse=1
    MSEIN=1000000000
    MSE=1000000000
    maxError=1
    maxErrorAlpha=1
    #maxErrorBeta=1
    i=0
    j=3
    meanAsymNs=100
    meanAsymNr=100
    #meanAsymNrr=100
    MinMeanAsymNs=10
    counter=0
    found=False
    sys.stdout.write("\033[1;36m")
    while ((meanAsymNs>0.15)or(meanAsymNr>20)or(meanAsymNrr>20))and(counter<150):
        counter=counter+1
        sys.stdout.write("\033[1;36m")
        if i>0:
            del history
            del estimator
        estimator = KerasRegressor(build_fn=builder.baseline_model(i,j), epochs=500, batch_size=12, verbose=1,validation_split=0.2)
        history=estimator.fit(X_train, y_train,validation_split=0.2,callbacks=[callback])
        #mse=history.history['mse']
        #mse=mse[-1]
        #MSEs.append(mse)
        i=i+1
        if i>12:
            i=0
            j=j+3
        y_pred = sc1.inverse_transform(estimator.predict(X_test))
        test=y_pred-y_test
        denominator=y_pred+y_test
        test0=test[:,0]
        test1=test[:,1]

        maxError=max(abs(test0))
        maxErrorAlpha=max(abs(test1))

        MSEIN=np.sqrt(np.sum((test0)**2 +test1**2 ))#+test2**2))
        maxAsymNS=max(200*test0/denominator[:,0])
        maxAsymAlpha=max(200*test1/denominator[:,1])

        meanAsymNsIn=np.mean(np.abs(200*test0/denominator[:,0]))
        meanAsymNr=np.mean(np.abs(200*test1/denominator[:,1]))

        print("{0}:{1}".format(meanAsymNsIn,MinMeanAsymNs))
        if (meanAsymNsIn<MinMeanAsymNs)and(MSEIN<MSE):
            MinMeanAsymNs=meanAsymNsIn
            MSE=MSEIN
            print("Minimal asymmetry to date is: {0}".format(MinMeanAsymNs))
            HELPER.saveModel(estimator.model,DIR)
            print("Saved model to disk") 
        meanAsymNs=meanAsymNsIn
        sys.stdout.write("\033[1;31m")
        print("mse: %.4f | maxError: %.4f%% | max NS Asym.: %.4f%%" %(mse,maxError*100,maxAsymNS))
        print("maxError in alpha: %.4f%% | max alpha Asym.: %.4f%%" %(maxErrorAlpha*100,maxAsymAlpha))
        print("number of hidden layers is {0}".format(i+3))
        print("number of nodes is {0}".format(j))
        print("mean asymmetry in Ns is {0}%%".format(meanAsymNs))
        print("calculated MSE is: ",MSEIN)
        Time.sleep(5)
        print('DONE WITH THIS ONE')
    
    
    """
    1.delete everything
    """
    
    
    del estimator, sc1, sc
    del y_test, y_pred, X_test, X_train
    
    """
    2.get the data, and split it
    """
    
    dataset = pd.read_csv('./INSANNE_4_7_around0965.csv')
    X = dataset.iloc[:, 0:3].values
    y = dataset.iloc[:, 3:5].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    
    """
    3. load the model and transformations
    """
    
    estimator,sc,sc1=HELPER.loadEstimator(DIR)
    sys.stdout.write('\033[32m')
    print('Found a suitable arch')
    found=True
    
    """
    4. use the loaded transformations to transform X_test, X_train, y_test, y_train.
    """
    
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    
    y_train = sc1.transform(y_train)
    
    Y_TEST= sc1.transform(y_test)
    
    
    Time.sleep(5)
    sys.stdout.write('\033[33m')
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X_test, Y_TEST, cv=kfold, scoring='neg_mean_squared_error')
    print("Baseline: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    sys.stdout.write('\033[0m')
    
    """check the model from loading"""
    
    y_pred = sc1.inverse_transform(estimator.predict(X_test))
    test=y_pred-y_test
    denominator=y_pred+y_test
    test0=test[:,0]
    test1=test[:,1]
    test2=test[:,2]
    maxError=max(abs(test0))
    maxErrorAlpha=max(abs(test1))

    maxAsymNS=max(200*test0/denominator[:,0])
    maxAsymAlpha=max(200*test1/denominator[:,1])
    meanAsymAlpha=np.mean(200*test1/denominator[:,1])

    meanAsymNs=np.mean(np.abs(200*test0/denominator[:,0]))
    MSE=np.mean(np.square(test))
    MSEerror=np.std(np.square(test)) 
    if meanAsymNs<0.15:
        sys.stdout.write('\033[48;5;46m')
    
    print("overall MSE: %.4f (%.4f)" % (np.mean(np.square(test)), np.std(np.square(test)) ))
    print('standard deviation in n_s is {0}'.format(np.std(test0)))
    print('standard deviation in alpha is {0}'.format(np.std(test1)))
    #print('standard deviation in beta is {0}'.format(np.std(test2)))
    #print(sc1.inverse_transform(estimator.predict(sc.transform([[-0.046325624,0.37679353,-1.1210711]]))))
    #print(sc2.inverse_transform([estimator.predict(sc.transform([[-0.046325624,0.37679353,-1.1210711]]))]))
    print("mse: %.4f | maxError: %.4f%% | max NS Asym.: %.4f%%" %(MSE,maxError*100,maxAsymNS))
    print("MeanAsym in alpha: %.4f%% | max alpha Asym.: %.4f%%" %(meanAsymAlpha,maxAsymAlpha))
    print("MeanAsym in beta: %.4f%% | max beta Asym.: %.4f%%" %(meanAsymBeta,maxAsymBeta))
      
    print('mean asymmetry in Ns is %.4f%%'%(meanAsymNs))
    print('ann fully trained')


if (mode=='MCMC'):# or (found):
    
    ## First load the model:
    tf.autograph.set_verbosity(0)

            
    dataset = pd.read_csv('./INSANNE_4_7_around0965.csv')
    X = dataset.iloc[:, 0:3].values
    y = dataset.iloc[:, 3:5].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    
    """
    3. load the model and transformations
    """
    
    estimator,sc,sc1=HELPER.loadEstimator(DIR)
    sys.stdout.write('\033[32m')
    print('Found a suitable arch')
    
    """
    4. use the loaded transformations to transform X_test, X_train, y_test, y_train.
    """
    
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    
    y_train = sc1.transform(y_train)
    
    Y_TEST= sc1.transform(y_test)
    
    
    #Make sure we have the correct model:
    y_pred = sc1.inverse_transform(estimator.predict(X_test))
    test=y_pred-y_test
    denominator=y_pred+y_test
    test0=test[:,0]
    test1=test[:,1]
    
    maxError=max(abs(test0))
    maxErrorAlpha=max(abs(test1))
  
    maxAsymNS=max(200*test0/denominator[:,0])
    maxAsymAlpha=max(200*test1/denominator[:,1])
    meanAsymAlpha=np.mean(200*test1/denominator[:,1])
   
    meanAsymNs=np.mean(np.abs(200*test0/denominator[:,0]))
    MSE=np.mean(np.square(test))
    MSEerror=np.std(np.square(test)) 
    if meanAsymNs<0.15:
        sys.stdout.write('\033[48;5;46m')
        print('Woohooo')
    
    #Load the posterior distributions:
        
    matC=loadmat('./Likelihoods/DP')


    ns=matC['xns'][0]
    nsL=matC['Pns'][0]
    
    nr=matC['xnr'][0]
    nrL=matC['Pnr'][0]
    

    POST=posterior_dist()
    

    
    NS=np.vstack((ns,nsL))
    NR=np.vstack((nr,nrL))
    #NRR=np.vstack((nrr,nrrL))
    
    GR=1

    POST.load(NS,nrun=NR)#,nrunrun=NRR)
    #Initial conditions
    x0=[0.1055,-0.1176,-0.63440]

    # Define proposal
    sigma=None
    G3P=MCMC.gaussian_3_proposal()
    G3P.define_Sigma(x0)
    counter=1
    acc_NUM=0
    ACCRATE=0
    
    plat=platform.system()
    if ChainNum==0:# or plat.lower() !='linux':
    
        if (os.path.isfile('chains.csv')and(not NEW)):
            try:    
                x0,ACC_RATE,LNPROB,LENGTH=getLast(0)
            except:
                x0,ACC_RATE,LNPROB=getChain(0)
                LENGTH=len(x0[:,0])
            acc_NUM=np.round(len(ACC_RATE)*LENGTH)
            counter=2
        ShowCounter=0
        while acc_NUM<100000*iterations:   
            
            GR=1
            [chain, accept_rate, lnprob]=MCMC.mh_sampler(x0, lnprob_fn=POST, prop_fn=G3P,acc_RATE=ACCRATE,acc_NUM=acc_NUM, prop_fn_kwargs={}, iterations=iterations)
            
            acc_NUM=np.round(len(ACC_RATE)*ACC_RATE[-1])
            ACCRATE=ACC_RATE[-1]
            for j in range(len(x0)):
                GR=MCMC.GR_stats(CHAIN[:,j],BurnIn)
            GR=GR-1
            try: 
                LENGTH = int(genfromtxt('chainsL_{0}.csv'.format(Number),dtype=(int),skip_header=1).item())
            except:
                LENGTH=0
            #G3P.move_sigma(ACC_RATE[-1])
            saveChain(chain,ACCRATE,lnprob,LENGTH,0)
            #if ShowCounter>50:
                #_saveChain(CHAIN,ACC_RATE,LNPROB)
            print("Acceptance rate is {0},  GR stat is {1}".format(ACCRATE,GR))
            print('length of chain is {}'.format(LENGTH+len(chain[:,0])))
            counter=counter+1
            x0=chain[-1,:]
            print(x0)
            #plt.tight_layout()
            #plt.draw()
            #plt.show()
            #print("---Plot graph finish---")
            #plt.ioff()
            #plt.show()
            #Time.sleep(0.5)
            #Histogram(CHAIN,1000)
            if ((show)and(ShowCounter>50)):    
                ShowCounter=0
                for i in range(len(CHAIN[0][:])):
                    plt.figure(i)
                    plt.cla()
                    plt.clf()
                    if len(CHAIN[:,i])<BurnIn:
                        x,bin,p=plt.hist(CHAIN[:,i],bins=100,)
                    else:
                        x,bin,p=plt.hist(CHAIN[BurnIn:,i],bins=100,)
                    a=SmoothNorm(x,5)
                    plt.plot(bin[:-1],a)
                    axes=plt.gca()
                    for item in p:
                        item.set_height(item.get_height()/max(x))
                    axes.set_ylim(top=1.1)
                    plt.pause(0.05)
                    plt.tight_layout()
                    plt.show() 
            ShowCounter+=1
            print(ShowCounter)
    else:
        
        
        if __name__ == '__main__':
            if plat.lower()!='linux':
                if mp.get_start_method(allow_none=False) is None:
                    mp.set_start_method('spawn')
                mp.freeze_support()

            manager = mp.Manager()
            
            print("starting parallel MCMC chains")
            GR=5
            counter=1
            GR_array=[]

            while abs(GR-1)>threshold:
                GR_table=np.zeros(shape=(ChainNum,2*len(x0)))
                QS=[]
                processes=[]
                lock=mp.Lock()
                ACCRATE=1
                pool = mp.Pool(ChainNum)
                for k in range(1,ChainNum+1):
                    mute() 
                    
                    QS.append(manager.Queue())
                    #p=mp.Process(target=MakeChain(x0*(0.9+0.2*np.random.rand(len(x0))),Number=k,NEW=NEW,ACCRATE=ACCRATE,counter=counter,Q=QS[k-1]))
                    #processes.append(p)
                    #p.start()
                    pool.apply_async(MakeChain(x0*(0.9+0.2*np.random.rand(len(x0))),Number=k,NEW=NEW,ACCRATE=ACCRATE,counter=counter,Q=QS[k-1]))
                    
                #for p in processes:   
                    #print("joining process {0}".format(p))
                #    p.join()
                pool.close()
                pool.join()
                if NEW:
                    NEW=False
                checkExit()
                row=0
                for k in range(ChainNum):
                    GR_stats=(QS[k].get())
                    L=QS[k].get()
                    for jj in range(len(GR_stats)):
                        GR_table[row,jj]=GR_stats[jj]
                    row=row+1
                    
                W=np.zeros(shape=(len(x0)))
                B=np.zeros(shape=(len(x0)))
                Rstat=np.zeros(shape=(len(x0)))
                
                for jj in range(len(x0)):
                    B[jj]=np.var(GR_table[:,2*jj])
                    W[jj]=np.mean(GR_table[:,2*jj+1])
                    Rstat[jj]=(((L-1)/L)*W[jj] +(1/L)*B[jj])/W[jj]                
                    
                unmute()
                GR_array.append(max(Rstat))
                sys.stdout.write("\033[1;31m")  
                print('Gelman-Rubin stats are: {}'.format(Rstat))
                #print(GR_array)
                if counter*iterations<BurnIn:
                    
                    GR=4
                else:
                    GR=max(GR_array)
                if len(GR_array)>100:
                    del GR_array[0]
                print('BurnIn is: {0}, we are at {1}'.format(BurnIn,counter*iterations))
                
                print('Number of iterations in this run is {0}\r\n'.format(counter))
                print()
                print()
                if counter%10==1:
                    print('saving combined file')
                    _CHAIN,_ACC_RATE,_LNPROB,LENGTH=CombineChains(ChainNum)
                    saveChain(_CHAIN,_ACC_RATE.squeeze(),_LNPROB.squeeze(),LENGTH,0,ALL=True)
                sys.stdout.write("\033[0;0m\r\n")
                counter=counter+1
            print('MCMC done!')
