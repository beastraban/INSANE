# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 08:55:32 2014

@author: beastraban
"""
from __future__ import division
import time as TIME
import numpy as np
import sympy as sym
from scipy.integrate import ode
#from progressbar import ProgressBar,Percentage,Bar,AdaptiveETA
#import progressbar
#from findMax import findMax
#import Image 
import warnings

class BackgroundSolver:
    """
        A class that, given an inflationary potential, as well as a set of initial conditions, 
        solves the differential equation system governing evolution of the Hubble parameter as well 
        as the inflationary potential value at all times during inflation.
        
        As the starting conditions for the wave-exponential function U_k are crucial, 
        the starting precision is multiplied 100-fold for the first 20 seconds.
        Parameters:
        -----------
        Pot:        
                    a symbolic function of "x" , where "x" is the inflationary potential coordinate 
                    (phi in most relevant literature).
                    default=1
        
        Potd:       
                    a symbolic function of "x" that is the differential of POT with reference to "x",
                    if none is given it is computed by POT
                    
        Hubble0:         
                    Initial Hubble parameter,at onset of inflation. if a value is not given, one HAS to supply a value for phidot0
        
        Phi0:      
                    Initial coordinate value - required.
        
        Phidot0:    
                    Initial coordinate velocity. if a value is not given one HAS to supply an initial H0.
        
        PhiEnd:
                    Final coordinate value - required for physical inflations
                    default=1
        Tprecision: 
                    The precision in time in which to integrate.
                    default=0.1  
                    
        endTime:    
                    If inflation isn't physical, at what time to finish integration.
                    default=100
        
        Planck_Mass:        
                    Planck Mass.
                    default=1
        
        physical:   
                    Boolean variable, is the inflationary model a physical one?
                    or are you checking non-physical solutions?
        
        Attributes:
        -----------
        solution:
                    A solution to the background evolution problem.
                    Internal structure: [t,a,H,phi,phidot,epsilon] 
                        t: 
                            the time variable used to integrate over.
                        a: 
                      self._ETA(v,vdd)      the logarithm of the scale factor, actually log(a(t))
                        H: 
                            the Hubble parameter
                        phi: 
                            the coordinate evolving in time
                        phidot: 
                            coordinate velocity evolving in time
                        epsilon: 
                            the first slow roll parameter in time
        v:
                    the value of the potential in time
        vd:
                    the value of the potential first derivative in time
        integrationError:
                    a measure of the normalized integration error,
                    as assessed by a dt vs. 2X(dt/2) analysis
        
        Methods:
        --------
        Solve():
            This method kick-starts the solution of the evolution problem. 
        """
                    
                    
    def __init__(self,l1=0,l2=0,l3=0,l4=0,l5=0,Hubble0=None,Phi0=None,PhiEnd=1,Phidot0=None,Tprecision=0.1,endTime=200,Planck_Mass=1,physical=True,selfCheck=False,poly=True,Pot=(sym.Symbol('x'))**0,mode='silent'):
        if (poly):
            self._V=np.poly1d([l5,l4,l3,l2,l1,1])
            self._Vd=np.poly1d([5*l5,4*l4,3*l3,2*l2,l1])
            self._Vdd=np.poly1d([20*l5,12*l4,6*l3,2*l2])
            self._Vddd=np.poly1d([60*l5,24*l4,6*l3])
            self._Vd4=np.poly1d([120*l5,24*l4])
            self._Vd5=np.poly1d([120*l5])
        else:

            self._Pot=Pot
            print(type(self._Pot))
            
            try:        
                self._x=self._Pot.atoms(sym.Symbol).pop()
            except Exception as e:
                self._x=sym.Symbol('x')                
                self._Pot=self._x**0
            self._Potd=sym.diff(self._Pot,self._x)
            self._Potdd=sym.diff(self._Potd,self._x)
            self._Potddd=sym.diff(self._Potdd,self._x)
            self._Potd4=sym.diff(self._Potddd,self._x)
            self._Potd5=sym.diff(self._Potd4,self._x)

            print(self._Pot) 
            print(self._Potd)               
                
            self._V=self._evalV
            self._Vd=self._evalVd
            self._Vdd=self._evalVdd
            self._Vddd=self._evalVddd
            self._Vd4=self._evalVd4
            self._Vd5=self._evalVd5

            
            print(self._V(0))
            print(self._Vd(0))            
            
            #####################################
            # insert 
            #####################################
        self.mode=mode
        self._Mpl=Planck_Mass
        self._H0,self._phi0,self._phidot0=self._getInitialConditions(Hubble0,Phi0,Phidot0)
        self._PhiEnd=PhiEnd
        self._epsilonStop=physical
        self._v0=self._V(self._phi0)    
        self._vd0=self._Vd(self._phi0)  
        self._vdd0=self._Vdd(self._phi0)
        self._vddd0=self._Vddd(self._phi0)
        self._vd40=self._Vd4V(self._phi0,self._v0)
        self._vd50=self._Vd5V(self._phi0,self._v0)
        self._epsilon=(1/2)*(self._vd0/self._v0)**2
        self._precision=Tprecision
        self._endTime=endTime
        self.solution=None    
        self.integrationError=None
        self.v=None
        self.vd=None
        self._selfCheck=selfCheck
        self._eta=self._ETA(self._v0,self._vdd0)
        self._xisq=self._Xisq(self._v0,self._vd0,self._vddd0)
        self._vd4v=self._Vd4V(self._v0,self._vd40)
        self._vd5v=self._Vd5V(self._v0,self._vd50)
        self.message=None
        #plt.plot(np.linspace(0,4),self._V(np.linspace(0,4)))
        #plt.plot(np.linspace(0,4),self._Vd(np.linspace(0,4)))
        #plt.plot(np.linspace(0,4),self._Vdd(np.linspace(0,4)))
  
    def Solve(self):
        """
        This method solves the cosmic evolution problem, given a set of initial condtions,
        as well as an inflationary potential.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        solution:
            Tuple of arrays in this format: [t,H,phi,phidot,v,vd,epsilon,ERROR,a,eta,xisq] 
                        t: 
                            the time variable used to integrate over.
                        H: 
                            the Hubble parameter
                        phi: 
                            the phi coordinate 
                        phidot: 
                            coordinate velocity 
                        v:
                            the value of the potential as a function of time
                        vd:
                            the value of d(V)/d(phi) as a function of time
                        epsilon: 
                            the quantity 1/2 *(V'/V)^2 
                        ERROR:
                            if selfcheck is active, yields the error between full step
                            and two half step integrations
                        a: 
                            the logarithm of the scale factor,this is actually log(a(t)) (=N)
                        eta:
                            the quantity (V''/V) as a function of time
                        xisq: 
                            the quantity (V'*V'''/V^2) as a function of time

        """
        startTime=TIME.time()
        y0 = [self._H0, self._phi0,self._phidot0]       # initial condition vector
        T=self._endTime # number of seconds to run
        dt=np.longdouble(self._precision) 
        t0=0
        backtrack=100
        
        #decide on integration scheme:
        #backend='dopri5'
        #backend='vode'
        backend='dop853'
        #backend='lsoda'
        #
        r = ode(self._F).set_integrator(backend,nsteps=2005)
        # suppress Fortran-printed warning
        r._integrator.iwork[2] = -1
        if self._selfCheck:
            SSS = ode(self._F).set_integrator(backend)
            SSS._integrator.iwork[2] = -1
            warnings.filterwarnings("ignore", category=UserWarning)
            SSS.set_initial_value(y0,t0).set_f_params([self._v0,self._vd0,self._Mpl])

        
        r.set_initial_value(y0,t0).set_f_params([self._v0,self._vd0,self._Mpl])
        sol=np.ndarray(shape=(14,1),dtype=np.longdouble)
        IntCheck=0
        err11=0
        ERROR=0
        iterator=0
        changes=0
        for i in range(3):
            sol[i+1,0]=y0[i]
        sol[0,0]=t0
        sol[4,0]=self._v0
        sol[5,0]=self._vd0
        sol[6,0]=self._epsilon
        sol[7,0]=0
        sol[8,0]=0
        sol[9,0]=self._eta
        sol[10,0]=self._xisq
        sol[11,0]=self._vd4v
        sol[12,0]=self._vd5v
        sol[13,0]=-0.5*self._phidot0**2
        IntCheck=sol[1,0]
        acumErr=0
        CHANGEDT=False
        Tchange=0.5
        dt=dt/np.longdouble(100)
        epsilon=self._epsilon
        eta=self._eta

        #bar=ProgressBar(widgets=[Percentage(), Bar(),AdaptiveETA()], maxval=np.log(2)).start()
       # print(np.log(epsilon))
        
        while (r.successful()):
            if not (self.mode=='silent'):
                print('{0},{1},{2}'.format(r.t,epsilon,r.y[1]))
            if (self._epsilonStop):
                #bar.update(1-np.abs(r.y[1])/np.abs(r.y[2]))
                if (epsilon>1):
                    #bar.finish()
                    self.message="Inflation ended - epsilon~1"
                    break
            #    if (r.y[1]>2*self._PhiEnd):
           #         self.message="Too long"
          #          break
                if (iterator>5000):
                    checkStationary=np.var(sol[1,(iterator-4999):iterator])
                    if (checkStationary==0):
                        self.message="Eternal inflation"
                        break
#                if (np.abs(eta)>1):
#                    self.message="Inflation ended - |eta|~1"
#                    break
                
            else:
               # bar.update(1000*r.t/np.longdouble(T))
                
                if (r.t>T):
                    #bar.finish()
                    break
            if (iterator>1000)and(not CHANGEDT):
                dt=dt*np.longdouble(100)
                CHANGEDT=True
                
            if self._selfCheck:
                sss=SSS 
                sss.integrate(s.t+dt/np.longdouble(2))
                vs=self._V(sss.y[1])
                vds=self._Vd(sss.y[1])
                sss.set_f_params([vs,vds,self._Mpl])
                sss.integrate(s.t+dt/np.longdouble(2))
            
            s=r
            time=s.t
            PHIDOT=s.y[2]
            s.integrate(s.t+dt)
            
            
            if not(r.successful()):
                print('not successful')
                iterator=iterator-backtrack
            while (not(s.successful()) and i<24 and dt>1/float(10**6)):
                print(i)
                dt=float(dt)/(10**(i+1))        
                iterator=iterator-1
                s = ode(self._F).set_integrator(backend)#nsteps=1)
                # suppress Fortran-printed warning
                s._integrator.iwork[2] = -1
                warnings.filterwarnings("ignore", category=UserWarning)
                yy=[sol[1,iterator],sol[2,iterator],sol[3,iterator]]
                s.set_initial_value(yy,sol[0,iterator]).set_f_params([sol[4,iterator],sol[5,iterator],self._Mpl])
        
                s.integrate(s.t+dt)  
                if self._selfCheck:
                    sss = ode(self._F).set_integrator(backend)#nsteps=1)
                    # suppress Fortran-printed warning
                    sss._integrator.iwork[2] = -1
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sss.set_initial_value(yy,sol[0,iterator]).set_f_params([sol[4,iterator],sol[5,iterator],self._Mpl])
                    sss.integrate(s.t+dt/np.longdouble(2))
                    vs=self._V(sss.y[1])
                    vds=self._Vd(sss.y[1])
                    sss.set_f_params([vs,vds,self._Mpl])
                    sss.integrate(s.t+dt/np.longdouble(2))
                if s.successful():
        #            SOL=np.ndarray(shape=(7,iterator+(len(sol[0,:])-iterator)*(10)),dtype=float)
        #            SOL[:,:len(sol[0,:])]=sol
        #            sol=SOL
        #            del SOL
                    changes+=1
                i+=1
                
            r=s
            if self._selfCheck:
                SSS=sss
                ERROR=np.longdouble(r.y[0])-np.longdouble(sss.y[0])

            
            if not(r.successful()):
                print('not successful')

            iterator+=1
            warnings.resetwarnings()
            #evaluate, v, vd and epsilon
            v=self._V(r.y[1])
            vd=self._Vd(r.y[1])
            vdd=self._Vdd(r.y[1])
            vddd=self._Vddd(r.y[1])
            vd4=self._Vd4(r.y[1])
            vd5=self._Vd5(r.y[1])
            if self._selfCheck:
                vs=self._V(SSS.y[1])
                vds=self._Vd(SSS.y[1])
            epsilon=self._EPSILON(v,vd)
            eta=self._ETA(v,vdd)
            xisq=self._Xisq(v,vd,vddd)
            Vd4V=self._Vd4V(v,vd4)
            Vd5V=self._Vd5V(v,vd5)
            if not(epsilon<1):
                print('inflation ended')
                # time - Hubble - phi - phidot -v - vd
            #print('{0},{1},{2},{3},{4},{5}'.format(r.t,r.y[0],r.y[1],r.y[2],v,vd))
            
            SOL=np.ndarray(shape=(14,1),dtype=np.longdouble)
            for i in range(3):
                SOL[i+1,0]=np.longdouble(r.y[i])
            SOL[0,0]=r.t
            SOL[4,0]=v
            SOL[5,0]=vd
            SOL[6,0]=epsilon
            SOL[7,0]=ERROR
            SOL[8,0]=sol[8,iterator-1]+SOL[1,0]*dt
            SOL[9,0]=eta
            SOL[10,0]=xisq
            SOL[11,0]=Vd4V
            SOL[12,0]=Vd5V
            SOL[13,0]=-0.5*r.y[2]**2
            sol=np.hstack((sol,SOL))
            if self._selfCheck:
                IntCheck=np.hstack((IntCheck,SSS.y[0]))
                acumErr=acumErr+np.abs(r.y[0]-SSS.y[0])
                err11=np.hstack((err11,acumErr))
                SSS.set_f_params([vs,vds,self._Mpl])
            #update r 
            r.set_f_params([v,vd,self._Mpl])
            
        sol=sol[:,0:iterator]
        self.H=sol[1,:]
        self.phi=sol[2,:]
        self.phidot=sol[3,:]
        v=sol[4,:]
        vd=sol[5,:]
        self.epsilon=sol[6,:]
        ERROR=sol[7,:]
        self.a=sol[8,:]
        self.eta=sol[9,:]
        self.xisq=sol[10,:]
        self.vd4v=sol[11,:]
        self.vd5v=sol[12,:]
        self.eh=-sol[13,:]/self.H**2
        self.deltah=-3+np.sqrt(self.epsilon/self.eh)*(3-self.eh)
        self.delpp=3*self.eh -3*self.deltah-self.eta*(3-self.eh)
        #need to add: H dot, phi ddot, phidddot 
        if self._selfCheck:
            IntCheck=IntCheck[0:iterator]
            err11=err11[0:iterator]
        self.t=sol[0,:]
        self.integrationError=ERROR/(IntCheck+self.H)
        self.v=v
        self.vd=vd
        
        self.solution=[self.t,self.a,self.H,self.phi,self.phidot,self.epsilon,self.eta,self.xisq,self.vd4v,self.vd5v]
        self.tau=np.array(self.a)
        self.tau[0]=0
        for i in range(1,len(self.a)):
                self.tau[i]=self.tau[i-1]+(self.t[i]-self.t[i-1])/(np.exp(self.a[i]))
        
        endTime=TIME.time()
        print('elapsed bacground geometry work is {0} seconds'.format(endTime-startTime))
        return(self.solution)
    
    def _getInitialConditions(self,H0,phi0,phidot0):
        if phi0 is None:
            raise Exception('Cannot solve background equations without starting coordinate, specify phi_0')
        v=self._V(phi0)
        if (phidot0 is None) and (H0 is None):
             raise Exception('must specify either initial coordinate velocity phi0 or Initial Hubble parameter H0')
        if phidot0 is None:
             phidot0=np.sqrt((2*(3*(H0**2)*(self._Mpl**2)-v)))
        if H0 is None:
            H0=np.sqrt(((1/np.longdouble(6))*phidot0**2 +(1/np.longdouble(3))*v))

        return(H0,phi0,phidot0)            
            


    

    def _EPSILON(self,v,vd):
        return((1/float(2))*((vd/v)**2))
    
    def _ETA(self,v,vdd):
        return(vdd/v)
        
    def _Xisq(self,v,vd,vddd):
        return((vddd*vd)/(v**2))
        
    def _Vd4V(self,v,vd4):
        return((vd4/v))
    
    def _Vd5V(self,v,vd5):
        return((vd5/v))
    
    def _F(self,t,y,arg1):
        v=arg1[0]#float(V.evalf(subs={Phi:y[2],l1:L1,l2:L2,l3:L3,l4:L4,l5:L5,V0:VV}))
        vd=arg1[1]#float(Vd.evalf(subs={Phi:y[2],l1:L1,l2:L2,l3:L3,l4:L4,l5:L5,V0:VV}))
        Mpl=arg1[2]        
        ########################################################################
        # H=y[0]
        # phi=y[1]
        # phidot=y[2]
        #
        ########################################################################
        H=np.longdouble(y[0])
        phi=np.longdouble(y[1])
        phidot=np.longdouble(y[2])
        
        H_rhs=(-1/float(2*Mpl**2))*phidot**2
        phi_rhs=phidot
        phidot_rhs=-3*H*phidot -vd
        
        return[np.longdouble(H_rhs),np.longdouble(phi_rhs),np.longdouble(phidot_rhs)]

    def _jac(t,y,arg1):
        pass
    
    def _evalV(self,y):
        return(np.longdouble(self._Pot.evalf(subs={self._x:y})))

    def _evalVd(self,y):
        return(np.longdouble(self._Potd.evalf(subs={self._x:y})))
    
    def _evalVdd(self,y):
        return(np.longdouble(self._Potdd.evalf(subs={self._x:y})))
        
    def _evalVddd(self,y):
        return(np.longdouble(self._Potddd.evalf(subs={self._x:y})))
    
    def _evalVd4(self,y):
        return(np.longdouble(self._Potd4.evalf(subs={self._x:y})))
    
    def _evalVd5(self,y):
        return(np.longdouble(self._Potd5.evalf(subs={self._x:y})))
