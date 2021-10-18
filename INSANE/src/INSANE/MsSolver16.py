# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:30:02 2015

@author: spawn
"""
from __future__ import division
import numpy as np
import platform
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.interpolate import interp1d
import time as TIME
from scipy.constants import pi
from scipy import io
import gc 
"Multiprocessing added"
import multiprocessing as mp


class MukhanovException(Exception):
    def __init__(self, value):
        self.message = value
    def __str__(self):
        return repr(self.message) #return repr(self.value)

class MsSolver:
    """
        class members (external input):
        -----------------------

        self._loga         the original scale factor
        self._H         the original Hubble parameter
        self._t         the original time array
        self._phi       the original phi array
        self._phidot    the original phidot array
        self._eta       the original eta array (second slow roll parameter)
        self._epsilon   the original epsilon array (first slow roll parameter)
        self._xisq        an array of (V'''*V')/(V)^2 along inflation

        class members (derived members):
        ------------------------

        self._a
        self._Eta       concated eta array
        self._N         number of efolds present in inflation
        self._Z
        self._Zdot
        self._Zddot
        self._Wsq (not sure necessary)
        self._WsG (gravitational waves omega squared)
        self._effectiveSliceTime
        self._interZ
        self._interH
        self._interWsq
        self._interWsG
        self._timing
        self._Tnum      number of time cells in arrays
        self._Knum      number of k cells in UKT
        self._UKT
        self._absUKT
        self._K         the actual k modes we used
        self._fit
        self._concatT
        self.ns
        self.nrun
        self.slope
        self.message
        self.NSevolution  the analytical ns quantity over inflation
        self.NrunEvolution the analytical nrun quantity over inflation
        self.NSdelta        the difference betwenn first order ns and second order derived analytically
        self.analNs         the projected analytical expression for ns
        self.analNrun       the projected analytical expression for nrun
        operational definitions:
        ------------------------

        self._Kprc              K-precision - the number of pints in k modes to evaluate
        self._Tprc              T-precision
        self._scaleCorrection
        self._initialSliceTime
        self._efoldsAgo
        self._efoldsNum
        self._logarithmicSpread
        self._pivotK
        self._startK
        self._endK


        Methods:
        --------

        prepareToWork:
            description:
                1) take log(a),H,phidot, in order to create Z=a*phidot/H
                2) create a=exp(log(a)), and normalize it.
                3) create Z=a*phidot/H
                4) get the internal k that corresponds to the physical l=1
                    (the mode that is just now enetrenig the horizon)
                5) make some kind of an extrapolated line to allow
                    for timing issues.
                6) In order to find the quantity (Z''+HZ')/Z later,
                    we will differentiate Z, once, and twice in time,
                    we can make interpolated this quantity to reduce
                    number of calls during numerical integration.
                7) create interpolated instances of a, H as well

        findKnee:
            description:
                find the time for efficient slicing -
                i.e. when the largest K mode has left the horizon
                and add ~ 5 seconds for stabilization.
                1) look for the time when
                        (K_max/a)**2 -(Z''+HZ'/Z) ~0
                2) this is the new SliceTime.

        F:
        the integration function:
            we are using a logarithmic scheme,
            and separating phase (imaginary part),
            from amplitude (real part):
            -------------------------------------
            a_rhs=np.longdouble(A)
            A_rhs=np.longdouble(A*(h-2*B))

            b_rhs=np.longdouble(B)
            B_rhs=np.longdouble(A**2-wSq +B*(h-B))

            for each iteration we need the w(t) at that time,
            as well as H(t) at that time. that is why are interpolating
            H earlier.
            and for each k-mode we will also need to interpolate:
                w**2=(k/a)**2-((Z''-HZ)/Z)

        FindZ(time):
            return the correct Z for said time

        FindWsq(time):
            return Wsq (omega squared) for that time

        FindH(time):
            return H for that time

        solveMod:
            this function gets some k, and solves the Mukhanov-Sassaki
            equation for this mode.
            1) create the correct Wsq for that mode by using the array form
                of Z Z' Z'' H a, creating the array
                self._Wsq

        buildUKT:
            this function builds the U(k,t) joint function,
            which we can later slice at the efficient slice time
            to get the correct power spectrum

        Analysis:
            this function analyzes the resultent UKT (or absUKT):
            1)  slice UKT at an appropriate time (in our case
                this will be at T=self._concatT[len(self._concatT)-1]),
                produce a function U of k alone.

            2)  derive the power spectrum by taking this U,
                and dividing it by Z(T), and applying:
                PS (Power Spectrum)=(1/np.longdouble(2*pi**2))(K**3)(abs(U/Z(T)))**2


            3)  produce the log-log curve:
                log(PS) vs. log(K)
                if we want graphics (mode='graphic') we will
                additionally create and show the plot

            4)  create a polyfit object for the log-log curve
                by fit=np.polyfit(log(K),log(PS),deg=2)

                --  since we are only after ns, nrun, we need no further
                    powers of the polynomial

                Slope=fit[1]
                nrun=fit[2]
                ns=1-Slope

            5)  if mode='graphic'
                produce the graph for the fit=
                pfit=np.poly1d(fit)
                gfit=np.pfit(log(K))
                and plot this
                and return these quantities.

            6)  return Slope, nrun, ns

        """
    def __init__(self,a,H,phi,phidot,t,epsilon,eta,xisq,Vd4V,Vd5V,V,eh,delh,pp,Tprecision=0.0064,Kprecision=15,log=True,pivotK=240,efoldsAgo=60,efoldsNum=8,check=False,physical=True,mode='aprox',preintegration=3):

        startTime=TIME.time()
        print(__name__)
        self.platform = platform.system()
        self.pre_Integration=preintegration
        self.mode=mode
        self.check=check
        self._v=np.array(V,dtype=np.double)
        self._loga=np.array(a,dtype=np.double)
        self._H=np.array(H,dtype=np.double)
        self._phi=np.array(phi,dtype=np.double)
        self._phidot=np.array(phidot,dtype=np.double)
        self._t=np.array(t,dtype=np.double)
        self._Tprc=Tprecision
        self._Kprc=Kprecision
        self._logarithmicSpread=log
        self._pivotK=pivotK
        self._efoldsAgo=efoldsAgo
        self._efoldsNum=efoldsNum
        print(self._efoldsNum)
        self._normalization=1
        self._epsilon=np.array(epsilon,dtype=np.double)
        self._eta=np.array(eta,dtype=np.double)
        self._xisq=np.array(xisq,dtype=np.double)
        self._Vd4V=np.array(Vd4V,dtype=np.double)
        self._Vd5V=np.array(Vd5V,dtype=np.double)
        self.eh=np.array(eh,dtype=np.double)
        self.delh=np.array(delh,dtype=np.double)
        self.delpp=np.array(pp,dtype=np.double)
        self.message=""
        self.physical=physical
        C=-(-2+np.log(2)+np.euler_gamma)
        length=len(self._t)
        newtime=np.linspace(self._t[0],self._t[len(self._t)-1],length*10)
        #print(type(newtime[1]))
        ##########################
        " spline everything"
        ##########################
        self._loga=self.SPLINE(self._loga,self._t,newtime)
        self._H=self.SPLINE(self._H,self._t,newtime)
        self._phi=self.SPLINE(self._phi,self._t,newtime)
        self._phidot=self.SPLINE(self._phidot,self._t,newtime)

        self._epsilon=self.SPLINE(self._epsilon,self._t,newtime)
        self._eta=self.SPLINE(self._eta,self._t,newtime)
        self._xisq=self.SPLINE(self._xisq,self._t,newtime)
        self._Vd4V=self.SPLINE(self._Vd4V,self._t,newtime)
        self._Vd5V=self.SPLINE(self._Vd5V,self._t,newtime)
        self._v=self.SPLINE(self._v,self._t,newtime)
        self.eh=self.SPLINE(self.eh,self._t,newtime)
        self.delh=self.SPLINE(self.delh,self._t,newtime)
        self.delpp=self.SPLINE(self.delpp,self._t,newtime)
        self._t=newtime
        ###########################
        "Done splining"
        ###########################



        ###########################
        "Find analytic n_s n_run as arrays"
        ###########################

        self.NSdelta=2*((1/3)*self._eta**2+(8*C-1)*self._eta*self._epsilon -(5/3 +12*C)*self._epsilon**2 -(C-1/3)*self._xisq)
        self.NSevolution=1+2*self._eta-6*self._epsilon+self.NSdelta
        self._NrunEvolution2=16*self._eta*self._epsilon -24*self._epsilon**2 -2*self._xisq
        self._NrunEvolution3=-self._epsilon**3*(40*C+184/3) +(self._eta*self._epsilon**2)*(-36*C +176/3) + (self._epsilon*self._eta**2)*(50*C-46/3) +(self._eta**3)*(4/3 -4*C)+self._xisq*(self._eta*(4/3 -8*C)+self._epsilon*(2*C-4))
        self._Del_NRUN4=(self._epsilon**4)*(456*C +3880/9) +(self._eta**4)*(2*C +2/9) - (self._epsilon**3)*(self._eta)*(4476*C+4412)/9 +(self._epsilon**2)*(self._eta**2)*(470*C+454)/3 - self._epsilon*(self._eta**3)*(28*C +40/3) +self._xisq*((self._epsilon**2)*(10*C+250/9)+(4*self._eta**2)/9 +(self._eta*self._eta) *(12*C-28/3))


        self.analCheckNs=2*C*self.delh**2 - 10*C*self.delh*self.eh - 8*C*self.eh**2 - 2*C*self.delpp - 4*self.delh**2*self.eh/(-self.eh + 1) - 2*self.delh**2*np.log(-self.eh + 1) - 12*self.delh*self.eh**2/(-self.eh + 1) + 10*self.delh*self.eh*np.log(-self.eh + 1) - 4*self.delh*self.eh/(-self.eh + 1) - 2*self.delh - 8*self.eh**3/(-self.eh + 1) + 8*self.eh**2*np.log(-self.eh + 1) - 4*self.eh**2/(-self.eh + 1) - 4*self.eh + 2*self.delpp*np.log(-self.eh + 1) + 1
        self.analCheckNR1=(np.diff(self.analCheckNs)/np.diff(self._t))*(1/self._H[1:len(self._H)])
        self.analCheckNR=np.ndarray(np.shape(self._H),dtype=np.double)
        self.analCheckNR[0]=0
        self.analCheckNR[1:len(self.analCheckNR)]=self.analCheckNR1
        #plt.plot(self._t,self.analCheckNR)


        self.NrunEvolution= self._NrunEvolution2 + self._NrunEvolution3

        self.NrunEv=self.NrunEvolution+self._Del_NRUN4

        self.analNrunHalfEv=-8*self.eh**2-10*self.eh*self.delh+2*self.delh**2 -2*self.delpp

        ##############################
        "Find ns value in different analytic formulations"
        "1. Stewart-Gong "
        "2. Dodelson-stewart"
        "3. Schwartz-Escalante-Garcia - Growing horizons"
        "4.Schwartz-Escalante-Garcia - Constant horizons"

        ##############################
        "1. Stewart-Gong"
        U1=2*self._epsilon
        V1=self._eta
        V2=self._xisq
        V3=U1*self._Vd4V

        self._SGns=1-3*U1 +2*V1 +(6*C-5/6)*U1**2 + (-8*C-1)*U1*V1 +(2/3)*V1**2 +(2*C+2/3)*V2 +(-12*C**2 +13*C/3 - 1867/36 +(11*np.pi**2)/2)*U1**3 +(24*C**2 +C/3 +595/6 -11*np.pi**2)*V1*U1**2 +(-8*C**2 -6*C -371/9 +(14*np.pi**2)/3)*U1*V1**2 +(4*V1**3)/9 +(-6*C**2 -2*C -49/3 +2*np.pi**2)*U1*V2 +(C**2 +8*C/3 +28/3 -(13*np.pi**2)/12)*V1*V2 +(C**2 +2*C/3 +2/9 -(np.pi**2)/12)*V3

        "2. Dodelson-Stewart"
        self._DSns=1-3*U1 + 2*(self.Q(0)*V1 + self.Q(1)*V2 +self.Q(2)*V3 +self.Q(3)*((U1)**(3/2))*self._Vd5V)

        "3. Schwartz-Escalante-Garcia - Growing horizons"
        self._GH=1-2*(self.eh+self.delh) -2*self.eh/(1-self.eh)

        "4.Schwartz-Escalante-Garcia - Constant horizons"
        self._CH=1-4*self.eh -2*self.delh



        """
        Derive simple quantities:
        """

        self._a=np.exp(self._loga)
        self._N=self._loga[len(self._loga)-1]
        Num=(self._N-self._efoldsAgo+self._efoldsNum)  #"efolds at end of analysis"
        StartNum=self._N-self._efoldsAgo               #"efolds at start of analysis"
        if (((self._N-self._efoldsAgo)<0)and(self.physical)):
             self.message="efolds in this inflation to small for {0} efolds ".format(self._efoldsAgo)
             raise MukhanovException("efolds in this inflation to small for {0} efolds ".format(self._efoldsAgo))
        else:
            if self.mode=='exact':
                index=np.where(self._loga-StartNum>=0)[0][0] #get the index of the first instance where efolds is bigger than efolds at start.
            else:
                if self.mode=='aprox':
                    index=np.where(self._phi>=0)[0][0]-1
                    self._efoldsAgo=self._loga[len(self._loga)-1]-self._loga[index]
                    Num=(self._N-self._efoldsAgo+self._efoldsNum)
                    StartNum=self._N-self._efoldsAgo
                else:
                    self.message='no appropriate phi scheme supplied'
                    raise MukhanovException('no appropriate phi scheme supplied')

            self.efoldsago=self._efoldsAgo
            self.index=index
            self.analNs=self.NSevolution[index]
            self.analNSCheck=self.analCheckNs[index]
            self.analNRCheck=self.analCheckNR[index]
            self.analNrun2=self._NrunEvolution2[index]
            self.analNrun=self.NrunEv[index]
            self.analNrun3=self.NrunEvolution[index]
            self.analNrunHalf=self.analNrunHalfEv[index]
            self.eta=self._eta[index]
            self.epsilon=self._epsilon[index]
            self.xisq=self._xisq[index]
            self.epsilonh=self.eh[index]
            self.deltah=self.delh[index]
            self.third=self.delpp[index]
            self.vd4v=self._Vd4V[index]
            self.vd5v=self._Vd5V[index]
            self.SGns=self._SGns[index]
            self.DSns=self._DSns[index]
            self.GHns=self._GH[index]
            self.CHns=self._CH[index]
            #print(self.eh[index])
            #print(self.delh[index])
            #print(self.delpp[index])


            PHI=self._phi[index]
            self.phiCMB=PHI
            a6=0
            a5=self._Vd5V[index]/120;
            a4=(self._Vd4V[index]-120*a5*PHI)/24
            a3=(-self._xisq[index]/np.sqrt(2*self._epsilon[index]) -24*a4*PHI -60*a5*PHI**2)/6
            a2=(self._eta[index]-6*a3*PHI -12*a4*PHI**2-20*a5*PHI**3)/2
            a1=(-np.sqrt(self._epsilon[index]*2) -2*a2*PHI-3*a3*PHI**2 -4*a4*PHI**3 -5*a5*PHI**4)
            self.coeffs=[self._v[index],a1,a2,a3,a4,a5,a6]

            ###########################################
            "Find the time to start analysis at, and end it"
            ###########################################


            #self._startingTime_index=np.where(self._loga-StartNum+3.5>=0)[0][0]
            self._startingTime_index=np.where(self._loga-StartNum+self.pre_Integration+0.5>=0)[0][0]
            self._initialTimeSlice_index=np.where(self._loga-Num>=0)[0][0]
            self._initialTimeSlice=self._t[self._initialTimeSlice_index]
            self._initialModeTime_index=index
            self._initialModeTime=self._t[self._initialModeTime_index]
            print('initialModeTime={0}'.format(self._initialModeTime))

        endTime=TIME.time()
        elapsed=endTime-startTime
        if not(mode=='silent'):
            print('done with preliminaries')
            print('elapsed initiation time is {0} sec'.format(elapsed))

        ##############################
        "Done with that"
        ##############################







        #self.startTime=self._t[self._startingTime_index]
        #self.firstModeTime=self._t[self._initialStartTime_index]
            #self.out=self._initialStartTime

            #self.iniTime=self._initialTimeSlice


    def findKnee(self):
        k=self._endK
        WsqMax=((k/self._a)**2 -self._ZppOverasqZ)
        self.WsqMax=WsqMax
        if not self.check:
            try:
                WsqMax=np.nan_to_num(WsqMax)
                TRADE=np.max(WsqMax)
                WsqMax=np.nan_to_num(WsqMax,nan=TRADE)
                bareIndex=np.where(WsqMax<=0)[0][0]
            except Exception as e:
                bareIndex=10
        else:
            bareIndex=10
        #print(self._t[bareIndex])
        self.out=bareIndex
        #now, instead of using time as an indication for when we are after freeze out - use efolds!!!
        try:
            self._effectiveSliceEfolds=self._loga[bareIndex]+10
            sliceIndex=np.where(self._effectiveSliceEfolds-self._loga<=0)[0][0]
            self._effectiveSliceTime=self._t[sliceIndex]
        except Exception as e:
             self._effectiveSliceTime=self._initialTimeSlice
        self.out2=self._effectiveSliceTime
        self._effectiveSliceTimeIndex=np.where(self._t-self._effectiveSliceTime>=0)[0][0]
#            """
#                try this:
#            """
#        self._effectiveSliceTime= self._initialTimeSlice
#        self._effectiveSliceTimeIndex=self._initialTimeSlice_index


    def Q(self,n):
        "generating function for q"
        if (n==0):
            q=1
        if (n==1):
            q=1.06297048787186
        if (n==2):
            q=0.418550335773528
        if (n==3):
            q=0.602566218787830
        if (n==4):
            q=-0.535472134139112
        if (n==5):
            q=1.27745007188817
#        x=sym.Symbol('x')
#        generator=(2**(-x))*sym.cos(sym.pi*x/2)*(3*sym.gamma(2+x))/((1-x)*(3-x))
#        qq=sym.diff(generator,x,n)
#        q=qq.subs(x,0)
        return(q)

#    def D(self,n):
#        "generating function for q"
#        x=sym.Symbol('x')
#        generator=(2**(x))*sym.cos(sym.pi*x/2)*(sym.gamma(2-x))/(1+x)
#        qq=sym.diff(generator,x,n)
#        q=qq.subs(x,0)
#        return(sym.N(q))

    def prepareToWork(self,mode='silent'):
        startTime=TIME.time()
        """
            Truncate a, phi,phidot, H, t to the correct size
        """

#        self._a=self._a[self._initialStartTime_index:]
#        self._H=self._H[self._initialStartTime_index:]
#        self._t=self._t[self._initialStartTime_index:]
#        self._phidot=self._phidot[self._initialStartTime_index:]
#        self._eta=self._eta[self._initialStartTime_index:]

        self._a=self._a[self._startingTime_index:]
        self._H=self._H[self._startingTime_index:]
        self._t=self._t[self._startingTime_index:]
        self._phidot=self._phidot[self._startingTime_index:]
        self._eta=self._eta[self._startingTime_index:]
        self._epsilon=self._epsilon[self._startingTime_index:]
        self._xisq=self._xisq[self._startingTime_index:]
        self._Vd4V=self._Vd4V[self._startingTime_index:]
        self._Vd5V=self._Vd5V[self._startingTime_index:]
        self.slowROLL=self.delh[self._startingTime_index:]
        self.delh=self.delh[self._startingTime_index:]
        self.eh=self.eh[self._startingTime_index:]
        self.delpp=self.delpp[self._startingTime_index:]


        self._initialModeTime_index=self._initialModeTime_index-self._startingTime_index

        self._a=self._normalization*(self._a)/((self._a[self._initialModeTime_index]))      #normalize a#
        #self._a=self._normalization*(self._a)/((self._a[0]))      #normalize a#
        if not self.check:
            self._Z=((self._a**1)*self._phidot)/self._H
        else:
            self._Z=np.ones(np.shape(self._a))
        self._Z=np.abs(self._Z)




        """
            derive Zdot,Zddot - no need use straight result z''/z=2a^2H^2(1+3delta/2 +epsilon +epsilon*delta/2 + 0.5*(2*epsilon^2 +epsilon*delta -delta^2 +delpp))
        """

        self._Zdot=np.diff(self._Z)/np.diff(self._t)
        self._Zddot=(np.diff(self._Zdot))/(np.diff(self._t[:len(self._Zdot)]))

        self._ZppOverasqZ=(2*self._H**2)*(1+3*self.delh/2 +self.eh+2*self.eh*self.delh + self.eh**2 +0.5*self.delpp)
        self._appOverasqa=(self._H**2)*(2-self.eh)
        #self.Z=self._Z
        #self.Zdot=self._Zdot
        #self.Zddot=self._Zddot
#        self.t=self._t
        truncIndex=len(self._Zddot)-2
        """
            truncate all derived quantitis to the initial time Slice index
        """
        self._a=self._a[:truncIndex]
        self._H=self._H[:truncIndex]
        self._t=self._t[:truncIndex]
        self._Z=self._Z[:truncIndex]
        self._Zdot=self._Zdot[:truncIndex]
        self._Zddot=self._Zddot[:truncIndex]
        self._ZppOverasqZ=self._ZppOverasqZ[:truncIndex]
        self._appOverasqa=self._appOverasqa[:truncIndex]
        wsqtemp=(self._Zddot+self._H*self._Zdot)/self._Z
        self.errorZ=200*(wsqtemp-self._ZppOverasqZ)/(wsqtemp+self._ZppOverasqZ)
        self._eta=self._eta[:truncIndex]
        self._epsilon=self._epsilon[:truncIndex]
        self._xisq=self._xisq[:truncIndex]
        self._Vd4V=self._Vd4V[:truncIndex]
        self._Vd5V=self._Vd5V[:truncIndex]
        self.slowROLL=self.slowROLL[:truncIndex]
        if not self.check:
            self.k0=self._a[self._initialModeTime_index]*np.sqrt((self._Zddot[self._initialModeTime_index]+self._Zdot[self._initialModeTime_index]*self._H[self._initialModeTime_index])/self._Z[self._initialModeTime_index])
            #self.k0=self._a[0]*np.sqrt((self._Zddot[0]+self._Zdot[0]*self._H[0])/self._Z[0])
        else:
            self.k0=1

        #self._efoldsNum=self._efoldsNum-2
        #self.efoldsNum=self._efoldsNum

        numstart=np.log(0.6)#np.log(10)#np.log(6)#0
        self._startK= self.k0*np.exp(numstart)  # start the analysis at l=10, after cosmic variance, adn when error is less significant)
        self._endK=self.k0*np.exp(self._efoldsNum)
        """
            creat an array of k modes to draw from
        """
        self.sK=self._startK
        self.eK=self._endK
        self._kmodes=np.ndarray(shape=self._Kprc,dtype=np.float)
        lmbda=(self._efoldsNum-numstart)/np.float((self._Kprc -1))
        #print(lmbda)
        self.lmbda=lmbda
        delta=(self._endK-self._startK)/np.float((self._Kprc ))
        if (self._logarithmicSpread):
            self._kmodes[0]=self._startK
            for i in range(self._Kprc ):
                self._kmodes[i]= self._startK*np.exp(lmbda*(i))
                print(i,"  ",self._kmodes[i])
            if (self._kmodes[len(self._kmodes)-1]>self._endK):
                print("in")
                self._kmodes[len(self._kmodes)-1]=self._endK

        else:
            for i in range(self._Kprc):
                self._kmodes[i]=self._startK+i*delta
        self._K=self._kmodes
        self.EFatFO=np.array(self._K)
        #self.K=self._K
        #self.invK=self._K[::-1]

        if (mode=='graphic'):
            plt.figure(num=15)
    #        plt.plot(self._t,self._Z)
    #        plt.plot(self._t,self._Zdot)
    #        plt.plot(self._t,self._Zddot)
            plt.plot(self._t,self._a)
            plt.show()



        """
            find the knee
        """

        self.findKnee()

        """
            truncate all derived quantities to the new time Slice index
        """
        self._a=self._a[:self._effectiveSliceTimeIndex]
        self._Z=self._Z[:self._effectiveSliceTimeIndex]
        self._Zdot=self._Zdot[:self._effectiveSliceTimeIndex]
        self._Zddot=self._Zddot[:self._effectiveSliceTimeIndex]
        self._ZppOverasqZ=self._ZppOverasqZ[:self._effectiveSliceTimeIndex]
        self._appOverasqa=self._appOverasqa[:self._effectiveSliceTimeIndex]

        self._H=self._H[:self._effectiveSliceTimeIndex]
        self._t=self._t[:self._effectiveSliceTimeIndex]
        tau=np.ones(len(self._a))
        for i in range(len(self._a)):
            tau[i]=np.trapz(1/self._a[0:i],self._t[0:i])
        self.tau=tau-tau[len(tau)-1]
        self._eta=self._eta[:self._effectiveSliceTimeIndex]
        self._epsilon=self._epsilon[:self._effectiveSliceTimeIndex]
        self._xisq=self._xisq[:self._effectiveSliceTimeIndex]
        self._Vd4V=self._Vd4V[:self._effectiveSliceTimeIndex]
        self._Vd5V=self._Vd5V[:self._effectiveSliceTimeIndex]
        self.slowROLL=self.slowROLL[:self._effectiveSliceTimeIndex]
        self._concatIndex=np.where(self._t-(self._t[len(self._t)-1]-0.1)>0)[0][0]
        self._concatT=self._t[:self._concatIndex]
        a=self._a#/(self._a[0])
        self._loga=np.log(a[:self._concatIndex])
        #if (np.max(np.abs(self.slowROLL))>1):
        #    self.message="This is not a slow roll inflation "
        #    raise MukhanovException("This is not a slow roll inflation ")


        """
            create interpolation for H and Z
        """

        self._interZ=interp1d(self._t,self._Z)
        self._interH=interp1d(self._t,self._H)
        self._interlA=interp1d(self._concatT,self._loga)

        """
            define the size of containers
        """
        self._Tnum=max(np.int32(100*(self._t[-1]-self._t[0])/self._Tprc),self._t.size)




        endTime=TIME.time()
        if not(mode=='silent'):
            print('elapsed time for prep-work is {0}'.format(endTime-startTime))
        """
            Done for now, that is as much as we can do right now
        """
    def _F_win(self,t,y,args=None):

        wSq=self._interW(t)
        wSqG=self._interWG(t)
        h=self._interH(t)

        a=np.double(y[0])
        A=np.double(y[1])
        b=np.double(y[2])
        B=np.double(y[3])
        va=np.double(y[4])
        vA=np.double(y[5])
        vb=np.double(y[6])
        vB=np.double(y[7])

        a_rhs=np.double(A)
        A_rhs=np.double(-A*(h+2*B))

        b_rhs=np.double(B)
        B_rhs=np.double(-(B*(h+B)+(wSq-A**2)))

        va_rhs=np.double(vA)
        vA_rhs=np.double(-vA*(h+2*vB))

        vb_rhs=np.double(vB)
        vB_rhs=np.double(-(vB*(h+vB)+(wSqG-vA**2)))


        return([a_rhs,A_rhs,b_rhs,B_rhs,va_rhs,vA_rhs,vb_rhs,vB_rhs])


    def _F(self,t,y,K,args=None):
        k=K
        wSqk=self.WsqList[k]
        wSqGk=self.WgList[k]
        wSq=wSqk(t)
        wSqG=wSqGk(t)
        h=self._interH(t)

        a=np.double(y[0])
        A=np.double(y[1])
        b=np.double(y[2])
        B=np.double(y[3])
        va=np.double(y[4])
        vA=np.double(y[5])
        vb=np.double(y[6])
        vB=np.double(y[7])

        a_rhs=np.double(A)
        A_rhs=np.double(-A*(h+2*B))

        b_rhs=np.double(B)
        B_rhs=np.double(-(B*(h+B)+(wSq-A**2)))

        va_rhs=np.double(vA)
        vA_rhs=np.double(-vA*(h+2*vB))

        vb_rhs=np.double(vB)
        vB_rhs=np.double(-(vB*(h+vB)+(wSqG-vA**2)))


        return([a_rhs,A_rhs,b_rhs,B_rhs,va_rhs,vA_rhs,vb_rhs,vB_rhs])

    def buildUKT(self,mode='silent',kk=None):
        if self.platform.lower() =='linux':
            return(self.buildUKT_Linux(mode,kk))
        else:
            return(self.buildUKT_Win(mode,kk))
        


    def buildUKT_Win(self,mode='silent',kk=None):
        """
            this should be a loop that solves all the modes and fills
            a pre-made empty UKT,
            as well as the K array
            and timing array
        """
        startBuildTime=TIME.time()
        MODE=mode
        if  not (mode=='silent'):
                plt.figure(num=19)

        """
            Make sure the whole inflationary stage where we evolve modes for,
            is a slow-roll stage
        """
        if max(np.abs(self._eta))>1:
            self._UKT=None
            self._absUKT=None
            self.message="Not a slow roll evolution"
        """
            Do nothing
        """

        """
            first bulid UKT - the empty 2 dim array to hold the mods
        """

        self._UKT=np.ndarray(shape=(self._Kprc,self._Tnum),dtype=np.double)
        self._absUKT=np.ndarray(shape=(self._Kprc,self._Tnum),dtype=np.double)
        self._absVKT=np.ndarray(shape=(self._Kprc,self._Tnum),dtype=np.double)

        if not(kk is None):
            self._K=kk
            self.EFatFO=np.array(self._K)

        iterator=0
        for k in self._K[::-1]:
            startTime=TIME.time()
            tempU=self.solveMod_win(k,mode=MODE)
            endTime=TIME.time()
            if  not (mode=='silent'):
                print('elapsed time for this mode is {0} sec'.format(endTime-startTime))
            else:
                print('Working')
            self.EFatFO[iterator]=self.EFatFO_this
            iterator+=1
            u=tempU[0]
            absU=tempU[1]
            v=tempU[2]
            absV=tempU[3]
            self.utime=tempU[4]
            self._UKT[self._Kprc-iterator,:len(u)]=u
            self._absUKT[self._Kprc-iterator,:len(absU)]=absU
            self._absVKT[self._Kprc-iterator,:len(absV)]=absV
            if (np.abs(k-1)<0.05):
                self.u=u
                self.v=v
                self.outk=k
                self.tau=self.tau[0:len(u)]


            """
            now we need to change k according to logarithmic or linear
            """

        """
            done with loop, now slight adjustments to absUKT and UKT
        """
        self._UKT=self._UKT[self._Kprc-iterator :,:len(self._concatT)-1]
        self._absUKT=self._absUKT[self._Kprc-iterator:,:len(self._concatT)-1]
        self._absVKT=self._absVKT[self._Kprc-iterator:,:len(self._concatT)-1]
        self._K=self._K[self._Kprc-iterator:]
        endBuildTime=TIME.time()
        self.buildTime=(endBuildTime-startBuildTime)
        if not(mode=='silent'):
            print('Total elapsed build-time was {0} sec'.format(self.buildTime))



    def buildUKT_Linux(self,mode='silent',kk=None):
        """
            this should be a loop that solves all the modes and fills
            a pre-made empty UKT,
            as well as the K array
            and timing array
            
            NEED TO PARALLELIZE THIS!!!!
            
            
        """
        startBuildTime=TIME.time()
        MODE=mode
        if  not (mode=='silent'):
                plt.figure(num=19)

        """
            Make sure the whole inflationary stage where we evolve modes for,
            is a slow-roll stage
        """
        #if max(np.abs(self._eta))>1:
            #self._UKT=None
        #    self._absUKT=None
        #    self.message="Not a slow roll evolution"
        #else:

        """
            first bulid UKT - the empty 2 dim array to hold the mods
        """
        gc.collect()
        self._UKT=np.ndarray(shape=(self._Kprc,self._Tnum),dtype=np.double)
        self._absUKT=np.ndarray(shape=(self._Kprc,self._Tnum),dtype=np.double)
        self._absVKT=np.ndarray(shape=(self._Kprc,self._Tnum),dtype=np.double)

        if not(kk is None):
            self._K=kk
            self.EFatFO=np.array(self._K)
            
        self.WsqList=dict()
        self.WgList=dict()
        for k in self._K[::-1]:
            Wsq=((k**2)/(self._a)**2 -self._ZppOverasqZ)
            WsG=((k**2)/(self._a)**2 -self._appOverasqa)
            interW=interp1d(self._t,Wsq)
            interWG=interp1d(self._t,WsG)
            self.WsqList[k]=interW
            self.WgList[k]=interWG
        
        
        #self.Kque=mp.Queue()
        manager = mp.Manager()
        self.Quk=manager.Queue()
        self.QuU=manager.Queue()
        self.QuV=manager.Queue()
        self.QuAbsV=manager.Queue()
        self.QuAbsU=manager.Queue()
        self.Qtime=manager.Queue()
        self.QuFO=manager.Queue()
        
        self.processes=[]
        lock=mp.Lock()
        print("starting parallel processing of modes")
        for k in self._K[::-1]:
            p=mp.Process(target=self.solveMod,args=(k,self.Quk,self.QuU,self.QuAbsU,self.QuV,self.QuAbsV,self.Qtime,self.QuFO,lock,'dop853','silent'))
            self.processes.append(p)
            p.start()
            
            
        for p in self.processes:   
            print("joining process {0}".format(p))
            p.join()
        
        
        print("Done with modes")
        "Extract solutions and organize them according to K's"
        TempK=[]
        TempU=[]
        TempV=[]
        TempAbsU=[]
        TempAbsV=[]
        TempTime=[]
        TempEFatFO=[]
        while not(self.Quk.empty()):
            TempK.append(self.Quk.get())
            TempU.append(self.QuU.get())
            TempV.append(self.QuV.get())
            TempAbsU.append(self.QuAbsU.get())
            TempAbsV.append(self.QuAbsV.get())
            TempTime.append(self.Qtime.get())
            TempEFatFO.append(self.QuFO.get())


        Zipped=zip(TempK,TempU,TempV,TempAbsU,TempAbsV,TempTime,TempEFatFO)
        self.Zipped=list(Zipped)
        self.Zipped = sorted(self.Zipped, key = lambda x: x[0]) 
        
        
        iterator=0

        
        for k in self._K:
            TempVec=self.Zipped[iterator]
            KK=TempVec[0]
            u=TempVec[1]
            v=TempVec[2]
            absU=TempVec[3]
            absV=TempVec[4]
            utime=TempVec[5]
            self.EFatFO[iterator]=TempVec[6]
            
            self._UKT[iterator,:len(u)]=u
            self._absUKT[iterator,:len(absU)]=absU
            self._absVKT[iterator,:len(absV)]=absV
            self.utime=utime
            iterator+=1
            



            """
                now we need to change k according to logarithmic or linear
            """

        """
            done with loop, now slight adjustments to absUKT and UKT
        """
        self._UKT=self._UKT[self._Kprc-iterator :,:len(self._concatT)-1]
        self._absUKT=self._absUKT[self._Kprc-iterator:,:len(self._concatT)-1]
        self._absVKT=self._absVKT[self._Kprc-iterator:,:len(self._concatT)-1]
        self._K=self._K[self._Kprc-iterator:]
        endBuildTime=TIME.time()
        self.buildTime=(endBuildTime-startBuildTime)
        if not(mode=='silent'):
            print('Total elapsed build-time was {0} sec'.format(self.buildTime))
        return(0)
    
    def solveMod_win(self,k,backend='dop853',mode='silent'):
        """
            first calculate the Wsq for this mode
            and interpolate it
        """
        if not(mode=='silent'):
            print('starting work on mode k={0}'.format(k))
        self._Wsq=((k**2)/(self._a)**2 -self._ZppOverasqZ)
        self._WsG=((k**2)/(self._a)**2 -self._appOverasqa)
#        #self._Wsq=np.zeros(len(self._a))       #check
#        ###### check
#        self._Wsq=(25)*np.ones(len(self._a))
#        self._H=np.zeros(len(self._a))
#        self._interH=interp1d(self._t,self._H)
#
#
#        ######
        self._interW=interp1d(self._t,self._Wsq)
        self._interWG=interp1d(self._t,self._WsG)
        self.W=self._Wsq
        self.Wg=self._WsG
        print(k)
        if (mode=='graphic'):
            plt.figure(num=11)
            plt.plot(self._t,self._Wsq,label='{0}'.format(k))
            plt.plot(self._t,self._WsG,label='Tensor {0}'.format(k))
            plt.draw()
            plt.legend(loc=1)
            plt.show()
            TIME.sleep(0.01)

        """
            Now, find the freeze out time for this mode:
        """
        if not(self.check):
            FOindex=np.where(self.W<=0)[0][0]
            try: 
                FOGindex=np.where(self.Wg<=0)[0][0]
            except: 
                print('GW is screwed')
                FOGindex=10
        else:
            FOindex=10;
            FOGindex=10;
        self.out3=FOindex
        self.out4=FOGindex
        FOindex=min(FOindex,FOGindex)


        FOtime=self._t[FOindex]
        EFatFO=self._loga[FOindex]
        self.EFatFO_this=EFatFO

        """
            Now for this mode take 3 efolds prior to Freeze-out to do pre-integration
        """
       # if (k>5):
       #     PIindex=0
       #     PIindex=np.where((EFatFO-self._loga)<=3)[0][0]
       # else:
       #     PIindex=0
        PIindex=np.where((EFatFO-self._loga)<=self.pre_Integration)[0][0]
        #PIindex=0
        PItime=self._t[PIindex]


        """
            Now set the initial conditions:
        """

        t0=self._t[PIindex]
        a0=0
        b0=0
        B0=0
        A0=k/np.double(self._a[PIindex])
        U0=1/np.sqrt(2*k)

        y0=[a0,A0,b0,B0,a0,A0,b0,B0]
        #print(y0)

        U=np.ndarray(shape=(9,self._Tnum),dtype=np.double)
        U[0,0]=t0
        U[1,0]=a0
        U[2,0]=A0
        U[3,0]=b0
        U[4,0]=B0
        U[5,0]=a0
        U[6,0]=A0
        U[7,0]=b0
        U[8,0]=B0

        r=ode(self._F_win).set_initial_value(y0,t0).set_integrator(backend,nsteps=4000)
        iterator=0
        while ((r.t<(self._effectiveSliceTime-0.05)) and (r.successful())):
            r.integrate(r.t+self._Tprc)
            iterator+=1
            """
                assign result of this step to U
            """

            U[0,iterator]=r.t
            U[1,iterator]=r.y[0]
            U[2,iterator]=r.y[1]
            U[3,iterator]=r.y[2]
            U[4,iterator]=r.y[3]
            U[5,iterator]=r.y[4]
            U[6,iterator]=r.y[5]
            U[7,iterator]=r.y[6]
            U[8,iterator]=r.y[7]
            #print(r.t)
            #print(U[:,iterator])
        if r.successful():
            U=U[:,:iterator]
            utime=U[0,:]
            aa=U[1,:]
            aadot=U[2,:]
            bb=U[3,:]
            bbdot=U[4,:]
            va=U[5,:]
            vadot=U[6,:]
            vb=U[7,:]
            vbdot=U[8,:]
            firstKindex=np.where(self._K>1)[0][0]
            if (abs(k-self._K[firstKindex]<0.001)):
                io.savemat('KMODE.mat',dict(aa=aa,bb=bb,utime=utime,k=k))
            """
                find how many entries ther are in a second
            """
            density=round(len(utime)/(utime[len(utime)-1]-utime[0]))
            #print(density)
            number=np.int32((self._t[PIindex]-self._t[0])*density)


            """
                add zeros where we need
            """
            TimeAdd=np.linspace(self._t[0],self._t[PIindex],number)
            Uadd=np.zeros(shape=(1,len(TimeAdd)),dtype=np.double)

            utime=np.append(TimeAdd,utime)
            aa=np.append(Uadd,aa)
            aadot=np.append(Uadd,aadot)
            bb=np.append(Uadd,bb)
            bbdot=np.append(Uadd,bbdot)
            va=np.append(Uadd,va)
            vadot=np.append(Uadd,vadot)
            vb=np.append(Uadd,vb)
            vbdot=np.append(Uadd,vbdot)

            """
            Now we need to construct the function and cast it
            into the proper timing
            """
            u=U0*(np.cos(aa)+np.complex(0,1)*np.sin(aa))*np.exp(bb)
            #u=U0*np.cos(aa)*np.exp(bb)
            absu=U0*np.exp(bb)
            v=U0*(np.cos(va)+np.complex(0,1)*np.sin(va))*np.exp(vb)
            absv=U0*np.exp(vb)

            """
                we interpolate and recast
            """

            interU=interp1d(utime,u)
            interAbsU=interp1d(utime,absu)
            interV=interp1d(utime,v)
            interAbsV=interp1d(utime,absv)

            u=interU(self._concatT)
            absu=interAbsU(self._concatT)
            v=interV(self._concatT)
            absv=interAbsV(self._concatT)

            if ((mode=='graphic') or (mode=='verbose') or (mode=='internal')):
                plt.figure(num=19)
                plt.plot(self._concatT,np.log(absu),label='{0}'.format(k))
                plt.xlabel="t"
                plt.ylabel="ln(abs(U))"
                plt.draw()
                plt.legend(loc=4)
                plt.show()
                plt.figure(num=20)
                plt.plot(self._loga,np.log((k**3)*(absu)**2),label='{0}'.format(k))
                plt.draw()
                plt.legend(loc=4)
                plt.show()
                plt.figure(num=21)
                plt.plot(utime,aa,label='log(phase){0}'.format(k))
                plt.plot(utime,aadot,label='der (log(phase){0}'.format(k))
                plt.plot(utime,bb,label='log(amplitue) {0}'.format(k))
                plt.plot(utime,bbdot,label='der (log(amplitude)){0}'.format(k))
                plt.draw()
                plt.legend(loc=4)
                plt.show()
                plt.figure(num=23)
                plt.plot(utime,np.cos(aa)*np.exp(bb),label='u_k function k={0}'.format(k))
                plt.draw()
                plt.legend(loc=4)
                plt.show()
                TIME.sleep(0.01)

            return([u,absu,v,absv,utime,self._concatT])
        else:
            raise MukhanovException('integration was not successful, at mode {0}'.format(k))

    
    
    def solveMod(self,k,Quk,QuU,QuAbsU,QuV,QuAbsV,Qtime,QuFO,lock,backend='dop853',mode='silent'):
        """
            first calculate the Wsq for this mode
            and interpolate it
        """
        BEGINTIME=TIME.time();
        if not(mode=='silent_complete'):
            print('starting work on mode k={0}'.format(k))
        
#        #self._Wsq=np.zeros(len(self._a))       #check
#        ###### check
#        self._Wsq=(25)*np.ones(len(self._a))
#        self._H=np.zeros(len(self._a))
#        self._interH=interp1d(self._t,self._H)
#
#
#        ######

        W=self.WsqList[k]
        W=W(self._t)
        Wg=self.WgList[k]
        Wg=Wg(self._t)
        print(k)
        # if (mode=='graphic'):
        #     plt.figure(num=11)
        #     plt.plot(self._t,Wsq,label='{0}'.format(k))
        #     plt.plot(self._t,WsG,label='Tensor {0}'.format(k))
        #     plt.draw()
        #     plt.legend(loc=1)
        #     plt.show()
        #     TIME.sleep(0.01)

        """
            Now, find the freeze out time for this mode:
        """
        if not(self.check):
            W=np.nan_to_num(W)
            TRADE=np.max(W)
            W=np.nan_to_num(W,TRADE)
            
            
            Wg=np.nan_to_num(Wg)
            TRADE=np.max(Wg)
            Wg=np.nan_to_num(Wg,TRADE)
            
            
            FOindex=np.where(W<=0)[0][0]
            try: 
                FOGindex=np.where(Wg<=0)[0][0]
            except: 
                print('GW is screwed')
                FOGindex=10
            
        else:
            FOindex=10;
            FOGindex=10;
        #self.out3=FOindex
        #self.out4=FOGindex
        FOindex=min(FOindex,FOGindex)


        FOtime=self._t[FOindex]
        EFatFO=self._loga[FOindex]
        

        """
            Now for this mode if it is a later mode take 3 efolds prior to Freeze-out to do pre-integration
        """
       # if (k>5):
       #     PIindex=0
       #     PIindex=np.where((EFatFO-self._loga)<=self.pre_Integration)[0][0]
       # else:
       #     PIindex=0
       
        PIindex=np.where((EFatFO-self._loga)<=self.pre_Integration)[0][0]
        #PIindex=0
        PItime=self._t[PIindex]


        """
            Now set the initial conditions:
        """

        t0=self._t[PIindex]
        a0=0
        b0=0
        B0=0
        A0=k/np.double(self._a[PIindex])
        U0=1/np.sqrt(2*k)

        y0=[a0,A0,b0,B0,a0,A0,b0,B0]
        #print(y0)

        U=np.ndarray(shape=(9,self._Tnum),dtype=np.double)
        U[0,0]=t0
        U[1,0]=a0
        U[2,0]=A0
        U[3,0]=b0
        U[4,0]=B0
        U[5,0]=a0
        U[6,0]=A0
        U[7,0]=b0
        U[8,0]=B0

        r=ode(self._F).set_initial_value(y0,t0).set_integrator(backend,nsteps=4000).set_f_params(k)
        iterator=0
        while ((r.t<(self._effectiveSliceTime-0.05)) and (r.successful())):
            r.integrate(r.t+self._Tprc)
            iterator+=1
            """
                assign result of this step to U
            """

            U[0,iterator]=r.t
            U[1,iterator]=r.y[0]
            U[2,iterator]=r.y[1]
            U[3,iterator]=r.y[2]
            U[4,iterator]=r.y[3]
            U[5,iterator]=r.y[4]
            U[6,iterator]=r.y[5]
            U[7,iterator]=r.y[6]
            U[8,iterator]=r.y[7]
            #print(r.t)
            #print(U[:,iterator])
        if r.successful():
            U=U[:,:iterator]
            utime=U[0,:]
            aa=U[1,:]
            aadot=U[2,:]
            bb=U[3,:]
            bbdot=U[4,:]
            va=U[5,:]
            vadot=U[6,:]
            vb=U[7,:]
            vbdot=U[8,:]
            firstKindex=np.where(self._K>1)[0][0]
            if (abs(k-self._K[firstKindex]<0.001)):
                io.savemat('KMODE.mat',dict(aa=aa,bb=bb,utime=utime,k=k))
            """
                find how many entries ther are in a second
            """
            density=round(len(utime)/(utime[len(utime)-1]-utime[0]))
            #print(density)
            number=np.int32((self._t[PIindex]-self._t[0])*density)


            """
                add zeros where we need
            """
            TimeAdd=np.linspace(self._t[0],self._t[PIindex],number)
            Uadd=np.zeros(shape=(1,len(TimeAdd)),dtype=np.double)

            utime=np.append(TimeAdd,utime)
            aa=np.append(Uadd,aa)
            aadot=np.append(Uadd,aadot)
            bb=np.append(Uadd,bb)
            bbdot=np.append(Uadd,bbdot)
            va=np.append(Uadd,va)
            vadot=np.append(Uadd,vadot)
            vb=np.append(Uadd,vb)
            vbdot=np.append(Uadd,vbdot)

            """
            Now we need to construct the function and cast it
            into the proper timing
            """
            u=U0*(np.cos(aa)+np.complex(0,1)*np.sin(aa))*np.exp(bb)
            #u=U0*np.cos(aa)*np.exp(bb)
            absu=U0*np.exp(bb)
            v=U0*(np.cos(va)+np.complex(0,1)*np.sin(va))*np.exp(vb)
            absv=U0*np.exp(vb)

            """
                we interpolate and recast
            """

            interU=interp1d(utime,u)
            interAbsU=interp1d(utime,absu)
            interV=interp1d(utime,v)
            interAbsV=interp1d(utime,absv)

            u=interU(self._concatT)
            absu=interAbsU(self._concatT)
            v=interV(self._concatT)
            absv=interAbsV(self._concatT)

            if ((mode=='graphic') or (mode=='verbose') or (mode=='internal')):
                plt.figure(num=19)
                plt.plot(self._concatT,np.log(absu),label='{0}'.format(k))
                plt.xlabel="t"
                plt.ylabel="ln(abs(U))"
                plt.draw()
                plt.legend(loc=4)
                plt.show()
                plt.figure(num=20)
                plt.plot(self._loga,np.log((k**3)*(absu)**2),label='{0}'.format(k))
                plt.draw()
                plt.legend(loc=4)
                plt.show()
                plt.figure(num=21)
                plt.plot(utime,aa,label='log(phase){0}'.format(k))
                plt.plot(utime,aadot,label='der (log(phase){0}'.format(k))
                plt.plot(utime,bb,label='log(amplitue) {0}'.format(k))
                plt.plot(utime,bbdot,label='der (log(amplitude)){0}'.format(k))
                plt.draw()
                plt.legend(loc=4)
                plt.show()
                plt.figure(num=23)
                plt.plot(utime,np.cos(aa)*np.exp(bb),label='u_k function k={0}'.format(k))
                plt.draw()
                plt.legend(loc=4)
                plt.show()
                TIME.sleep(0.01)
            #self.Kque.put(k)
            lock.acquire()
            Quk.put(k)
            QuV.put(v)
            QuU.put(u)
            QuAbsU.put(absu)
            QuAbsV.put(absv)
            Qtime.put(utime)
            QuFO.put(EFatFO)
            print("Elapsed time for mode {} was {} seconds".format(k,TIME.time()-BEGINTIME))
            lock.release()
            #return([u,absu,v,absv,utime,self._concatT,k])
            return(0)
        else:
            raise MukhanovException('integration was not successful, at mode {0}'.format(k))

    def analysis(self,mode='silent',deg=None,pivotScale=None,Threshold=10**(-4)):
        K=(self._K/self.k0)
        DEG=deg
        
        
        
        if (pivotScale is None):
            self.pivot=1.2*10**(-4)
        else:
            self.pivot=pivotScale

        shift=self.pivot/(1.2*10**(-4))
        self.lk0=np.log(K)
        """
            Define slice timing
        """


        index= len(self._concatT)-2
        T=self._concatT[index]
        N=self._loga[:index]
        """
            get UKT at time T, and Z at time T
        """

        U=self._absUKT[:,index]
        V=self._absVKT[:,index]
        Z=self._interZ(T)
        A=np.exp(self._interlA(T))

        self.UKT=self._UKT[:,:index]
        self.T=self._concatT[:index]
        self.N=N
        self.zz=Z
        """
            prodoce U over Z, and the power spectrum
        """

        UoverZ=U/Z
        PS=(1/np.double(2*pi**2))*(K**3)*(UoverZ)**2
        VoverA=V/A
        PG=(4/np.double(2*pi**2))*(K**3)*(VoverA)**2


        """
            create the fit:
            we need to recast to float64 because fit doesnt accept
            float128
        """
        self.shift=shift
        K=K/shift
        K=np.array(K,dtype=np.float64)
        PS=np.array(PS,dtype=np.float64)
        PG=np.array(PG,dtype=np.float64)
        "Normalize PS"
        #self.As=np.exp(3.089)/(10**10)
        #PS=(PS/PS[0])*self.As
        self.lk=np.log(K)
        self.lps=np.log(PS)
        self.lgs=np.log(PG)
#        l_range=2500
#        prec=0.01

#        self.cls=self.get_CLs(PS,K,l_range,prec)
#        ll=range(1,l_range+1)
#        ll=np.array(ll)
#        self.ells=ll
#        if (mode=='graphic'):
#            plt.figure(num=151)
#            plt.plot(np.log(ll),(ll*(ll+1)*self.cls))

        #for i in range(0,len(self.lk)-1):
        #    weights=1/np.sqrt((1+np.abs(np.exp(np.abs(self.lk-self.lk[i])))));
        #    fit,cov=np.polyfit(np.log(K),np.log(PS),w=weights,deg=DEG,cov=True)
        #   """
        #        errors in ns and nrun by covariance matrix
        #    """
         #   project1=[1,0,0]
         #   project2=[0,1,0]

            #self.nrun_error=np.dot((project1*fit).T,np.dot(cov,(project1*fit)))
            #self.ns_error=np.dot((project2*fit).T,np.dot(cov,(project2*fit)))
         #   error=np.sum((np.polyval(fit,np.log(K)) - np.log(PS))**2)
         #   if (i==0):
         #       ERROR=error
         #   else:
         #       ERROR=min(ERROR,error)
         #       j=i

        #weights=1/np.sqrt((1+np.abs(np.exp(np.abs(self.lk-self.lk[j])))));
        #fit,cov=np.polyfit(np.log(K),np.log(PS),w=weights,deg=DEG,cov=True)
        if (DEG is None):
            maxDeg=20
        else:
            maxDeg=DEG
        self.maxDeg=maxDeg
        self.error=1
        DEG=0
        while ((self.error>(Threshold))and(DEG<maxDeg)):
            DEG=DEG+1
            fit=np.polyfit(np.log(K),np.log(PS),deg=DEG)
            self.error=(np.sqrt(np.sum((np.polyval(fit,np.log(K)) - np.log(PS))**2)))/np.double(len(self._K))
    
    
        self.As=1
        self.As_pivot=1
        self.ns=0
        self.nrun=0
        self.nrunrun=0
        if (not(self.error>(Threshold))):
            self.Slope=fit[DEG-1]
            if (DEG>1):
                self.nrun=2*fit[DEG-2]
            if (DEG>2):
                self.nrunrun=6*fit[DEG-3]
            self.ns=1+self.Slope
            self.As_pivot=np.exp(fit[DEG])
    
        self.scalarDEG=DEG
    
    
        self.At_pivot=1
        self.errorTensor=1
        DEG=0
        while ((self.errorTensor>(Threshold))and(DEG<maxDeg)):
            DEG=DEG+1
            fitTens=np.polyfit(np.log(K),np.log(PG),deg=DEG)
            self.errorTensor=np.sum((np.polyval(fitTens,np.log(K)) - np.log(PG))**2)
        self.nT=3
        if (not(self.errorTensor>(Threshold))):
            self.nT=fit[DEG-1]
            self.At_pivot=np.exp(fitTens[DEG])
        self.tensorDeg=DEG
        
        """
            if mode='graphic' plot log(PS) vs. log(K)
        """

        self.r_pivot=self.At_pivot/self.As_pivot
        stretch=np.linspace(-2,9,10000)
        poly=np.poly1d(fit)
        pfit=poly(stretch)
        polyTens=np.poly1d(fitTens)
        pfitTens=polyTens(stretch)

        self.As=np.exp(poly(-np.log(shift)))
        self.At=np.exp(polyTens(-np.log(shift)))
        self.r=self.At/self.As

        if (mode=='graphic'):
            plt.figure(num=43)
            plt.plot(np.log(K),np.log(PS),marker='x',label='PS at time {0}'.format(T))
            plt.plot(np.log(K),np.log(PG),marker='+',label='PT at time {0}'.format(T))




            analfit_interim=np.poly1d([self.analNrun/2,self.analNs-1,0])
            zero_order=analfit_interim(np.log(K)[0])-pfit[0]
            analfit=np.poly1d([self.analNrun2/2,self.analNs-1,-zero_order])
            afit=analfit(np.log(K))
            plt.plot(stretch,pfit,label='PS fitting: Slope={0:.4f} ,ns={1:.4f} ,nrun={2:.6f}'.format(self.Slope,self.ns,self.nrun))
            plt.plot(stretch,pfitTens,label='PT fitting')
            plt.plot(np.log(K),afit,label='analytic curve:ns={0:.4f},nrun={1:.6f}'.format(self.analNs,self.analNrun2))
            plt.title("Power Spectrum vs. K & fitting; log-log")
#            plt.xlabel("ln(K)")
#            plt.ylabel="ln(Power Spectrum)"
            plt.legend(loc=1)
            self.image=plt.gcf()
            plt.show()

    def SPLINE(self,array,time,newtime):
        tempArray=array

        f=interp1d(time,tempArray)
        newArray=f(newtime)
        return(newArray)

#    def get_CLs(self,PK,K,l_range,prec):
#        self.As=np.exp(3.089)/(10**10)
#        kk=K
#        self.KD=0.14
#        self.Keq=0.01
#        CLS=np.zeros(l_range)
#        k=np.linspace(np.log(kk[0]),np.log(kk[len(kk)-1]),len(kk)/prec)
#        k=np.exp(k)
#        self.clK=k
#        x=k/self.Keq
#        Transfer=(np.log(np.e+0.171*x)/(0.171*x))*(1+0.284*x +(1.18*x)**2 +(0.399*x)**3 +(0.49*x)**4)**(-0.25)
#        self.Transfer=Transfer
#        dk=k[1]-k[0]
#        pk=self.SPLINE(np.log(PK),np.log(kk),np.log(k))
#        pk=np.exp(pk)
#        self.pk=pk
#        pkmat=((9/10)**2)*self.As*pk*(Transfer)**2
#        self.pkmat=pkmat
#        #TABLE=np.zeros((l_range,len(k)))
#        #fill the table
#        for l in range(l_range):
#            cl=0
#            for j in range(len(k)):
#                bes=np.sqrt((2/np.pi)*k[j]/self.KD)*sp.special.jv(l+1/2,k[j]/self.KD)
#                cl =cl + (4*np.pi/25)*((pkmat[j]*(bes)**2)/k[j]**2)*dk
#            CLS[l]=cl
#        return(CLS)


#    def get_CLs_depricated(self,PK,K,l_range,prec):
#        self.KD=0.14
#        self.Keq=0.01
#
#        CLS=np.zeros(l_range)
#        k=np.linspace(K[0],K[len(K)-1],len(K)/prec)
#        x=k/self.Keq
#        Transfer=(np.log(1+0.171*x)/0.171*x)*(1+0.284*x +(1.18*x)**2 +(0.399*x)**3 +(0.49*x)**4)**(-0.25)
#        dk=k[1]-k[0]
#        pk=self.SPLINE(PK,K,k)
#        pkmat=((9/10)**2)*pk*(Transfer)**2
#
#        TABLE=np.zeros((l_range,len(k)))
#        #fill the table
#        for j in range(len(k)):
#            bes=sp.special.sph_jn(l_range,k[j]/self.KD)
#            BES=np.array(bes[0])
#            for l in range(l_range):
#                TABLE[l,j]=BES[l]
#        for l in range (l_range):
#            cl=0
#            for j in range(len(k)):
#                cl =cl + ((pkmat[j]*(TABLE[l,j])**2)*k[j]**2)*dk
#            CLS[l]=cl
#        return(CLS)

    def get_CLs2(self,PK,K,l_range,H0):
        prec=10**(-5)
        CLS=np.zeros(l_range-1)
        k=np.linspace(K[0],K[len(K)-1],len(K)/prec)
        pk=self.SPLINE(PK,K,k)
        c=3*10**8
        for l in range(1,l_range):
            kk=l*H0/(2*c);
            J=np.where(k>kk)[0][0]
            pp=pk[J-1]+(pk[J]-pk[J-1])*(kk-k[J-1])
            CLS[l-1]=(pp)/(l*(l+1)*np.pi)
        return(CLS)


    def analysis2(self,mode='silent',deg=None,pivotScale=None,Threshold=10**(-4)):
        K=(self._K/self.k0)
        DEG=deg
        if (pivotScale is None):
            self.pivot=1.2*10**(-4)
        else:
            self.pivot=pivotScale

        shift=self.pivot/(1.2*10**(-4))
        """
            Define slice timing
        """


        index= len(self._concatT)-2
        T=self._concatT[index]
        N=self._loga[:index]
        """
            get UKT at time T, and Z at time T
        """

        U=self._absUKT[:,index]
        V=self._absVKT[:,index]
        Z=self._interZ(T)
        A=np.exp(self._interlA(T))

        self.UKT=self._UKT[:,:index]
        self.T=self._concatT[:index]
        self.N=N
        self.zz=Z
        """
            prodoce U over Z, and the power spectrum
        """

        UoverZ=U/Z
        PS=(1/np.double(2*pi**2))*(K**3)*(UoverZ)**2
        VoverA=V/A
        PG=(4/np.double(2*pi**2))*(K**3)*(VoverA)**2


        """
            create the fit:
            we need to recast to float64 because fit doesnt accept
            float128
        """
        self.shift=shift
        K=K/shift
        K=np.array(K,dtype=np.float64)
        PS=np.array(PS,dtype=np.float64)
        PG=np.array(PG,dtype=np.float64)
        self.lk=np.log(K)
        self.lps=np.log(PS)
        self.lgs=np.log(PG)
        l_range=2500
        prec=0.01

        """
        Skeleton code for translating P(k) to matter PS into cls to work with HEALPIX (comment out, next version!)
        """
#        self.cls=self.get_CLs(PS,self._K,l_range,prec)
#        ll=range(1,l_range+1)
#        ll=np.array(ll)
#        self.ells=ll
#        if (mode=='graphic'):
#            plt.figure(num=151)
#            plt.plot(np.log(ll),(ll*(ll+1)*self.cls))
        """

        """
        #for i in range(0,len(self.lk)-1):
        #    weights=1/np.sqrt((1+np.abs(np.exp(np.abs(self.lk-self.lk[i])))));
        #    fit,cov=np.polyfit(np.log(K),np.log(PS),w=weights,deg=DEG,cov=True)
        #   """
        #        errors in ns and nrun by covariance matrix
        #    """
         #   project1=[1,0,0]
         #   project2=[0,1,0]

            #self.nrun_error=np.dot((project1*fit).T,np.dot(cov,(project1*fit)))
            #self.ns_error=np.dot((project2*fit).T,np.dot(cov,(project2*fit)))
         #   error=np.sum((np.polyval(fit,np.log(K)) - np.log(PS))**2)
         #   if (i==0):
         #       ERROR=error
         #   else:
         #       ERROR=min(ERROR,error)
         #       j=i

        #weights=1/np.sqrt((1+np.abs(np.exp(np.abs(self.lk-self.lk[j])))));
        #fit,cov=np.polyfit(np.log(K),np.log(PS),w=weights,deg=DEG,cov=True)
        if (DEG is None):
            self.error=1
            DEG=0
            while ((self.error>(Threshold))and(DEG<20)):
                DEG=DEG+1
                fit=np.polyfit(np.log(K),np.log(PS),deg=DEG)
                self.error=(np.sqrt(np.sum((np.polyval(fit,np.log(K)) - np.log(PS))**2)))/np.double(len(self._K))


            #self.pivotK=np.exp(self.lk[j])
            self.As=1
            self.As_pivot=1
            self.ns=0
            self.nrun=0
            self.nrunrun=0
            if (not(self.error>(Threshold))):
                self.Slope=fit[DEG-1]
                if (DEG>1):
                    self.nrun=2*fit[DEG-2]
                if (DEG>2):
                    self.nrunrun=6*fit[DEG-3]
                self.ns=1+self.Slope
                self.As_pivot=np.exp(fit[DEG])

            self.scalarDEG=DEG


            self.At_pivot=1
            self.errorTensor=1
            DEG=0
            while ((self.errorTensor>(Threshold))and(DEG<20)):
                DEG=DEG+1
                fitTens=np.polyfit(np.log(K),np.log(PG),deg=DEG)
                self.errorTensor=np.sum((np.polyval(fitTens,np.log(K)) - np.log(PG))**2)
            self.nT=3
            if (not(self.errorTensor>(Threshold))):
                self.nT=fit[DEG-1]
                self.At_pivot=np.exp(fitTens[DEG])
            self.tensorDeg=DEG
        else:
            fit=np.polyfit(np.log(K),np.log(PS),deg=DEG)
            self.error=np.sum((np.polyval(fit,np.log(K)) - np.log(PS))**2)
            fitTens=np.polyfit(np.log(K),np.log(PG),deg=DEG)
            self.errorTensor=np.sum((np.polyval(fitTens,np.log(K)) - np.log(PG))**2)
            self.ns=1
            self.nrun=0
            self.nrunrun=0
            self.Slope=fit[DEG-1]
            if (DEG>1):
                self.nrun=2*fit[DEG-2]
            if (DEG>2):
                self.nrunrun=6*fit[DEG-3]
            self.ns=1+self.Slope
            self.As_pivot=np.exp(fit[DEG])
            self.nT=3
            self.nT=fit[DEG-1]
            self.At_pivot=np.exp(fitTens[DEG])
            self.scalarDEG=DEG
        """
            if mode='graphic' plot log(PS) vs. log(K)
        """

        self.r_pivot=self.At_pivot/self.As_pivot
        stretch=np.linspace(-2,9,10000)
        poly=np.poly1d(fit)
        pfit=poly(stretch)
        polyTens=np.poly1d(fitTens)
        pfitTens=polyTens(stretch)

        self.As=np.exp(poly(-np.log(shift)))
        self.At=np.exp(polyTens(-np.log(shift)))
        self.r=self.At/self.As

        if (mode=='graphic'):
            plt.figure(num=43)
            plt.plot(np.log(K),np.log(PS),marker='x',label='PS at time {0}'.format(T))
            plt.plot(np.log(K),np.log(PG),marker='+',label='PT at time {0}'.format(T))




            analfit_interim=np.poly1d([self.analNrun/2,self.analNs-1,0])
            zero_order=analfit_interim(np.log(K)[0])-pfit[0]
            analfit=np.poly1d([self.analNrun2/2,self.analNs-1,-zero_order])
            afit=analfit(np.log(K))
            plt.plot(stretch,pfit,label='PS fitting: Slope={0:.4f} ,ns={1:.4f} ,nrun={2:.6f}'.format(self.Slope,self.ns,self.nrun))
            plt.plot(stretch,pfitTens,label='PT fitting')
            plt.plot(np.log(K),afit,label='analytic curve:ns={0:.4f},nrun={1:.6f}'.format(self.analNs,self.analNrun2))
            plt.title("Power Spectrum vs. K & fitting; log-log")
#            plt.xlabel("ln(K)")
#            plt.ylabel="ln(Power Spectrum)"
            self.image=plt.gcf()

            plt.legend(loc=1)
            plt.show()


