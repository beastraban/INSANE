# -*- coding: utf-8 -*-
"""
Created on Sun May  9 09:34:20 2021

@author: Ira Wolfson

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from UTILS import *
from UTILS import Saver_Loader


def mh_sampler(x0, lnprob_fn, prop_fn,acc_RATE,acc_NUM, prop_fn_kwargs={}, iterations=100000):
    """Simple metropolis hastings sampler.

    :param x0: Initial array of parameters.
    :param lnprob_fn: Function to compute log-posterior.
    :param prop_fn: Function to perform jumps.
    :param prop_fn_kwargs: Keyword arguments for proposal function
    :param iterations: Number of iterations to run sampler. Default=100000

    :returns:
        (chain, acceptance, lnprob) tuple of parameter chain , acceptance rate
        and log-posterior chain.
    """
    if acc_NUM>0:
        r=0
    else:
        r=1
    # number of dimensions
    ndim = len(x0)

    # initialize chain, acceptance rate and lnprob
    chain = np.zeros((iterations, ndim))
    lnprob = np.zeros(iterations)
    accept_rate = np.zeros(iterations)

    # first sample
    chain[0] = x0
    lnprob0 = lnprob_fn(x0)
    lnprob[0] = lnprob0

    # start loop
    naccept = np.round(acc_NUM)
    for ii in range(r, iterations):

        # propose
        x_star, factor = prop_fn(x0, **prop_fn_kwargs)

        # draw random uniform number
        u = np.random.uniform(0, 1)

        # compute hastings ratio
        lnprob_star = lnprob_fn(x_star)
        H = np.exp(lnprob_star - lnprob0) * factor

        # accept/reject step (update acceptance counter)
        if u < H:
            x0 = x_star
            lnprob0 = lnprob_star
            naccept += 1

        # update chain
        chain[ii] = x0
        lnprob[ii] = lnprob0            
        accept_rate[ii] = naccept / (ii+acc_NUM)

    return chain, accept_rate, lnprob



class posterior_dist:
    
    def __init__ (self,ndim=3,seed=12343,DIR='./InsaANNe_LtoObs/'):
        
        HELPER=Saver_Loader()

        self.estimator,self.sc,self.sc1=HELPER.loadEstimator(DIR)
        
        
        
        
        np.random.seed(seed)
        self.nsLike=np.array([])
        self.Xns=np.array([])
        
        self.nrunLike=np.array([])
        self.Xnrun=np.array([])
        
        self.nrunrunLike=np.array([])
        self.Xnrunrun=np.array([])
        return None
        
    def load_ns(self,ns):
        self.nsLike=np.array(ns[1])
        self.Xns=np.array(ns[0])
        
        self.Ns=sp.interpolate.interp1d(self.Xns,self.nsLike)
        
        return None 
        
    def load_nrun(self,nrun):
        self.nrunLike=np.array(nrun[1])
        self.Xnrun=np.array(nrun[0])
        
        self.Nr=sp.interpolate.interp1d(self.Xnrun,self.nrunLike)
        return None 

    def load_nrunrun(self,nrunrun):
        self.nrunrunLike=np.array(nrunrun[1])
        self.Xnrunrun=np.array(nrunrun[0])
        
        self.Nrr=sp.interpolate.interp1d(self.Xnrunrun,self.nrunrunLike)

        
        return None 
    
    
    def load(self,ns,nrun=None,nrunrun=None):
        self.MAX=np.array([0,0,0],dtype=np.float64)
        self.MIN=np.array([0,0,0],dtype=np.float64)
        self.FUNCS={}
        
        if nrun is not None:
            self.nrunLike=np.array(nrun[1])
            self.Xnrun=np.array(nrun[0])
            self.MINnr=min(self.Xnrun)
            self.MAXnr=max(self.Xnrun)
            self.Nr=sp.interpolate.interp1d(self.Xnrun,self.nrunLike)
            self.MAX[1]=self.MAXnr
            self.MIN[1]=self.MINnr
            self.FUNCS['nr']=self.Nr
            
            if nrunrun is not None:
                self.nrrLike=np.array(nrunrun[1])
                self.Xnrr=np.array(nrunrun[0])
                self.MINnrr=min(self.Xnrr)
                self.MAXnrr=max(self.Xnrr)
                self.Nrr=sp.interpolate.interp1d(self.Xnrr,self.nrrLike)
                self.MAX[2]=self.MAXnrr
                self.MIN[2]=self.MINnrr
                self.FUNCS['nrr']=self.Nrr
            else:
                self.MAX=np.array(self.MAX.tolist()[0:2])
                self.MIN=np.array(self.MIN.tolist()[0:2])
                
        else:
            self.MAX=np.array(self.MAX.tolist()[0:2])
            self.MIN=np.array(self.MIN.tolist()[0:2])
            
            if nrunrun is not None:
                self.nrrLike=np.array(nrunrun[1])
                self.Xnrr=np.array(nrunrun[0])
                self.MINnrr=min(self.Xnrr)
                self.MAXnrr=max(self.Xnrr)
                self.Nrr=sp.interpolate.interp1d(self.Xnrr,self.nrrLike)
                self.MAX[1]=self.MAXnrr
                self.MIN[1]=self.MINnrr
                self.FUNCS['nrr']=self.Nrr
            else:
                self.MAX=np.array(self.MAX.tolist()[0:1])
                self.MIN=np.array(self.MIN.tolist()[0:1])

        
        self.nsLike=np.array(ns[1])
        self.Xns=np.array(ns[0])
        self.MINns=min(self.Xns)
        self.MAXns=max(self.Xns)
        self.Ns=sp.interpolate.interp1d(self.Xns,self.nsLike)
        self.MAX[0]=self.MAXns
        self.MIN[0]=self.MINns
        self.FUNCS['ns']=self.Ns
        
        
        return None
        
        
        return None
        
    def print_ns(self):
        plt.figure(1)
        plt.plot(self.Xns,self.nsLike)
        plt.show()
        return None



    def print_nr(self):
        plt.figure(2)
        plt.plot(self.Xnrun,self.nrunLike)
        plt.show()
        return None
    
    
    def print_nrr(self):
        plt.figure(3)
        plt.plot(self.Xnrunrun,self.nrunrunLike)
        plt.show()
        return None
        
    
    def __call__(self, x):        
        #ns=x[0]
        #nrun=x[1]
        #nrunrun=x[2]
        x0=x
        x=self.sc1.inverse_transform([self.estimator.predict(self.sc.transform([x0]))])[0]
        #print(x)
        if ((x0[0]<0.04016)or(x0[0]>0.1562)or(x0[1]<-0.4088)or(x0[1]>0.1697)or(x0[2]<-1.3882) or (x0[2]>0.1882)):
            return -1e6

        if (np.where(x<self.MIN)[0].size>0) or (np.where(x>self.MAX)[0].size>0): 
            return -1e6
        result=1
        for key in self.FUNCS.keys():
            if key=='ns':
                result=result*self.FUNCS[key](x[0])
            if 'nr' in self.FUNCS:
                if key=='nr':
                    result=result*self.FUNCS[key](x[1])
                if key=='nrr':
                    result=result*self.FUNCS[key](x[2])
            else:
                if key=='nrr':
                    result=result*self.FUNCS[key](x[1])

                            
            
            
        return np.log(result)
    

    
def GR_stat(chain,BurnIn):
        
        
        if len(chain[:,0])>BurnIn:
            L = len(chain[0,:])
            MEAN=np.zeros(shape=(L,1))
            VAR=np.zeros(shape=(L,1))
            N=len(chain[BurnIn:,0])
            for i in range(L):
                MEAN[i]=np.mean(chain[BurnIn:,i])
                VAR[i]=np.var(chain[BurnIn:,i])
            
            GMEAN=np.mean(MEAN)
            B=0
            W=0
            for i in range(L):
                B=B+(N/(L-1))*((GMEAN-MEAN[i])**2)
                W=W+VAR[i]/L 
            
            V=((N-1)/(N))*W +((L+1)/(L*N))*B
            R=np.sqrt(V/W)
            
            return(R)
        return(1)
        
 
def GR_stats(chain,BurnIn):
       
       
       if len(chain[:])>BurnIn:
           L = 1
           
           N=len(chain[BurnIn:])
           for i in range(L):
               MEAN=np.mean(chain[BurnIn:])
               VAR=np.var(chain[BurnIn:])
           
           GMEAN=np.mean(MEAN)
           B=0
           W=0
           for i in range(L):
               B=B+(N/(L))*((GMEAN-MEAN)**2)
               W=W+VAR/L 
           
           V=((N-1)/(N))*W +((L+1)/(L*N))*B
           R=np.sqrt(V/W)
           
           return(R)
       return(1)   
 
    


class  gaussian_3_proposal:
    """
    Gaussian proposal distribution.

    Draw new parameters from Gaussian distribution with
    mean at current position and standard deviation sigma.

    Since the mean is the current position and the standard
    deviation is fixed. This proposal is symmetric so the ratio
    of proposal densities is 1.

    :param x: Parameter array
    :param sigma:
        Standard deviation of Gaussian distribution. Can be scalar
        or vector of length(x)

    :returns: (new parameters, ratio of proposal densities)
    """
    
    def __init__(self):
        
        
        return None
    
    
    def define_Sigma(self,x,sigma=None):
        if sigma is None:
            #sigma=0.05
            #self.sigma=0.1*np.random.rand(len(x),)
            self.sigma=0.001*np.ones(shape=(len(x),))
            self.sigma[1]=0.00001
            #self.sigma=np.array([0.01,0.01,0.01])
        else:
            self.sigma = 0.0001*np.ones((x.shape))
            self.sigma[0]=0.0275/10
            self.sigma[1]=0.1546/10
            self.sigma[2]=0.4016/10
            
            
        self.step=0.0004*np.ones(shape=(len(x),))

        
        
        return None
        
        

    def move_sigma(self,accept_rate):
        RATE=accept_rate
        print(RATE)
        if RATE>0.6:
            if min(self.sigma<0.004):
                self.step=0.1*self.step
            self.sigma=self.sigma+self.step
        if RATE<0.23:
            if min(self.sigma<0.004):
                self.step=0.1*self.step
            self.sigma=self.sigma-self.step
        print(self.sigma)
        return None
    
    def __call__(self,x):
        # Draw x_star
        x_star = x + np.random.randn(len(x))*self.sigma
        #print(np.random.randn(len(x))*self.sigma)
        #Draw from gaussian priors
        # x_star[0]=np.random.normal(loc=0.1055,scale=0.0275)
        # x_star[1]=np.random.normal(loc=-0.1176,scale=0.1546)
        # x_star[2]=np.random.normal(loc=-0.63440,scale=0.4016)
        
        
        #Draw from uniform:
        # x_star[0]=np.random.uniform(low=0.0485,high= 0.1562)
        # x_star[1]=np.random.uniform(low=-0.4088,high= 0.1697)
        # x_star[2]=np.random.uniform(low=-1.3882,high= 0.1882)
            
        #print(sigma)    
    
        # proposal ratio factor is 1 since jump is symmetric
        qxx = 1

        return (x_star, qxx)
   