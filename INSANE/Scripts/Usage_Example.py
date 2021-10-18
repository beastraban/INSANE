# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:18:31 2015

@author: Spawn
"""
from __future__ import division
from __future__ import print_function
import sys
import platform
from INSANE.BackgroundGeometry import BackgroundSolver
import os
"""
Here we need to define the parameters to run the script:
I am defining a range for, say l4, and I draw random values from this range.
"""
from INSANE.MsSolver16 import MsSolver
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import time as TIME
import sympy as sym
from configparser import ConfigParser


def main():
    
    Platform=platform.system()
    WD=os.getcwd()
    
    if len(sys.argv)<2:
        filename=(r'./params.ini')
    else:
        filename=sys.argv[1]
    
    
    config = ConfigParser()
    if Platform.lower() =='linux':
        config.read(filename)
        WD=WD+'/'
    else:
        config.read(filename)
        WD=WD+'\\'
    
    print(WD)
    input("Press Enter to continue...")
    
    """
    Define initial conditions and other background quantities from params.ini file
    """
    symbolic=eval(config.get('main', 'symbolic'))    
    V0=np.float64(config.get('main', 'V0'))
    l1=np.float64(config.get('main', 'l1'))
    l2=np.float64(config.get('main', 'l2'))
    l3=np.float64(config.get('main', 'l3'))
    l4=np.float64(config.get('main', 'l4'))
    l5=np.float64(config.get('main', 'l5'))
    l6=np.float64(config.get('main', 'l6'))
    l7=np.float64(config.get('main', 'l7'))
    phi=eval(config.get('main', 'phi'))
    V=eval(config.get('main', 'V'))
    H0=np.float64(eval(config.get('main', 'H0')))
    phidot0=np.float64(config.get('main', 'phidot0'))
    efolds=np.float64(eval(config.get('main', 'efolds')))
    efoldsago=np.float64(config.get('main', 'efoldsago'))
    phi0=np.float64(config.get('main', 'phi0'))
    Tprecision=np.float64(config.get('main', 'Tprecision'))
    from_list=eval(config.get('main', 'from_list'))
    feed=eval(config.get('main', 'feedback'))
    if (feed==0):
        feedback='silent'
    else:
        if (feed==1):
            feedback='verbose'
        else:
            feedback='graphic'
    
    physical=eval(config.get('main', 'physical'))
    EndK=np.float64(eval(config.get('main', 'EndK')))
    StartK=np.float64(eval(config.get('main', 'StartK')))
    
    
    """
    Define initial conditions and other quantities dor MS solver from params.ini file
    """
    MStimePrec=np.float64(eval(config.get('MS', 'MStimeprec')))
    MSkPrec=np.int32(eval(config.get('MS', 'MSkPrec')))
    Klog=eval(config.get('MS', 'Klog'))
    MSfeed=np.int32(eval(config.get('MS', 'MSfeedback')))
    if MSfeed==0:
        MSfeedback='silent'
    else:   
        if MSfeed==1:
            MSfeedback='verbose'
        else:
            MSfeedback='graphic'
    
    Anfeed=np.int32(eval(config.get('MS', 'AnFeedback')))
    if Anfeed==0:
        AnFeedback='silent'
    else:   
        if Anfeed==1:
            AnFeedback='verbose'
        else:
            AnFeedback='graphic'
    
    CMBpoint=config.get('MS', 'CMBpoint')
    pivotScale=np.float64(eval(config.get('MS', 'pivotScale')))
    deg=config.get('MS', 'deg')
    
    logfile=config.get('LOGFILE','logfile')
    if not logfile=='':
        sys.stdout = open('./'+logfile, 'w') 
        
    
    if deg=='None':
        deg=None
    else:
        deg=np.int32(deg)
            
    if from_list:
        NumOfFiles=10
    
        number=sys.argv[1]
        if (len(sys.argv)>2):
            degree=np.int32(sys.argv[2])
        else:
            degree=None;
    
    def sendMail(to,n):
        """
        you may insert a method that sends a mail to your email when done.
        """
        print('redacted')
    
    
    def parseCoeffs(string):
            """
            returns a tuple of the coefficients l1...l5,phi0 that have been stored in
            txt format as "l1;l2;l3;l4;l5;l6;phi0"
    
            Parameters
            -----------
            string the string to parse
    
            Returns
            -----------
            a 5 element list [l1,l2,l3,l4,l5,l6,phi0] of floats after parsing
    
            """
            STR=string
            print(STR)
            if isinstance(STR, str):
                STR=str(STR)
                STR=STR.split(";")
                return([np.double(STR[0]),np.double(STR[1]),np.double(STR[2]),np.double(STR[3]),np.double(STR[4]),np.double(STR[5]),np.double(STR[6])])
            else:
                e=Exception()
                e.message="not a proper string"
                raise e
            return([])
    
    
    
    plat=platform.system()
    
    """
        read the line from the list, and delete it from the list
    """
    if from_list:
        fileread="/home/spawn/Dropbox/PEGASUS/list{0}.txt".format(number)
        filewrite="/home/spawn/Dropbox/PEGASUS/PEGASUS results/results{0}.txt".format(number)
        ii=0
        while not (os.stat(fileread).st_size == 0):
        
            fileid=open(fileread,"r")
            try:
        
                content=fileid.readlines()
                line=content[0]
                Vars=parseCoeffs(line)
                fileid.close()
                content=content[1:]
                fileid=open(fileread,"w")
                for line in content:
                   fileid.write(line)
                fileid.flush()
                fileid.close()
            except Exception as e:
                fileid.close()
                print(e.message)
        
            l1=Vars[0]
            l2=Vars[1]
            l3=Vars[2]
            l4=Vars[3]
            l5=Vars[4]
            l6=Vars[5]
            l7=0
            phi0=Vars[6]
            V0=1
        
            phi=sym.Symbol('phi')
            V=1+l1*phi+l2*phi**2 +l3*phi**3 +l4*phi**4 +l5*phi**5 +l6*phi**6 +l7*phi**7
    
    
    
    
    
        try:
            solver=BackgroundSolver(l1,l2,l3,l4,l5,H0,phi0,1,phidot0,Tprecision=Tprecision,poly=(not symbolic),Pot=V,mode=feedback)
            solver.Solve()
            if (solver.message=="Inflation ended - epsilon~1"):#or(solver.message=="Too long"):
    
                #execfile("MS_script1.py")
                if (solver.a[len(solver.a)-1]>50):
                    try:
                       Solver=MsSolver(solver.a,solver.H,solver.phi,solver.phidot,solver.t,solver.epsilon,solver.eta,solver.xisq,solver.vd4v,solver.vd5v,solver.v,solver.eh,solver.deltah,solver.delpp,Tprecision=MStimePrec,Kprecision=MSkPrec,efoldsAgo=efoldsago,efoldsNum=efolds,log=Klog,mode=CMBpoint)
                       Solver.prepareToWork(mode=MSfeedback)
                       Solver.buildUKT(mode=MSfeedback)
                       if (Solver.message=="Not a slow roll evolution"):
                           result="{0};{1};{2};{3};{4};{5};{6};{7};{8}; -- {9}".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0,Solver.message)
                       else:
                           try:
                                Solver.analysis(mode=AnFeedback,pivotScale=pivotScale,deg=deg)
                                assymNRfull=200*np.abs(Solver.analNrun2-Solver.nrun)/np.abs(Solver.analNrun2+Solver.nrun)
                                assymNRhalf=200*np.abs(Solver.analNrunHalf-Solver.nrun)/np.abs(Solver.analNrunHalf+Solver.nrun)
                                assymNS=200*np.abs((Solver.analNs-Solver.ns)/(Solver.analNs+Solver.ns))
                                result="{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13};{14};{15};{16};{17};{18};{19};{20};{21};{22};{23}".format(l1,l2,l3,l4,l5,l6,Solver.coeffs[0],Solver.coeffs[1],Solver.coeffs[2],Solver.coeffs[3],Solver.coeffs[4],Solver.coeffs[5],Solver.coeffs[6],Solver.phiCMB,phi0,phidot0,H0,efolds,efoldsago,Solver.Slope,Solver.ns,Solver.nrun,Solver.nrunrun,Solver.error)#,assymNS,assymNRfull,assymNRhalf)
                           except Exception as e:
                                print(e.message)
                                result=e.message
                    except Exception as e:
                        result="{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};error message:'{11}'".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0,efolds,efoldsago,e.message)
    
                else:
                    if (solver.message=="Too long"):
                        print("Too long")
                        result="{0};{1};{2};{3};{4};{5};{6};{7};{8}; -- Too long".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0)
    
                    else:
                        print("not enough efolds for physical inflation")
                        result="{0};{1};{2};{3};{4};{5};{6};{7};{8}; -- not enough efolds for physical inflation".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0)
            else:
                    if (solver.message=="Too long"):
                        print("Too long")
                        result="{0};{1};{2};{3};{4};{5};{6};{7};{8}; -- Too long".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0)
    
                    else:
                        print(solver.message)
                        result="{0};{1};{2};{3};{4};{5};{6};{7};{8};{9}".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0,solver.message)
            #            break
    
    
    
    
    
    
        #;{10};{11},{12}
            if (plat=='Linux'):
                results=open(filewrite,"a")
                FORMAT=open("/home/spawn/Dropbox/PEGASUS/PEGASUS results/FORMAT.txt","a")
            else:
                results =open("E:\\Dropbox\\PEGASUS\\PEGASUS results\\results.txt","a")
                FORMAT=open("E:\\Dropbox\\PEGASUS\\PEGASUS results\\FORMAT.txt","a")
    
            results.write(result+"\n")
            results.flush()
            results.close()
            if (ii==0):
               __format="l1,l2,l3,l4,l5,,l6,V0,a1,a2,a3,a4,a5,a6,phiCMB,phi0,phidot0,H0,efolds,efoldsago,slope,ns,nrun,nrunrun,error"
               FORMAT.write(__format+"\n")
               FORMAT.flush()
               FORMAT.close()
            TIME.sleep(1)
    
        #    except Exception as e:
        #        print("something went wrong")
        #        print(e.message)
        #        print("Moving on then...")
        except Exception as e:
            print(e.message)
            print('didnt go that well')
        ii=ii+1
    else:
        solver=BackgroundSolver(l1,l2,l3,l4,l5,H0,phi0,1,phidot0,Tprecision=Tprecision,poly=(not symbolic),Pot=V,mode=feedback)
        solver.Solve()
        if (solver.message=="Inflation ended - epsilon~1"):#or(solver.message=="Too long"):
        
                    #execfile("MS_script1.py")
                    if (solver.a[len(solver.a)-1]>50):
                        try:
                           Solver=MsSolver(solver.a,solver.H,solver.phi,solver.phidot,solver.t,solver.epsilon,solver.eta,solver.xisq,solver.vd4v,solver.vd5v,solver.v,solver.eh,solver.deltah,solver.delpp,Tprecision=MStimePrec,Kprecision=MSkPrec,efoldsAgo=efoldsago,efoldsNum=efolds,log=Klog,mode=CMBpoint)
                           Solver.prepareToWork(mode=MSfeedback)
                           Solver.buildUKT(mode=MSfeedback)
                           if (Solver.message=="Not a slow roll evolution"):
                               result="{0};{1};{2};{3};{4};{5};{6};{7};{8}; -- {9}".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0,Solver.message)
                           else:
                               try:
                                    Solver.analysis(mode=AnFeedback,pivotScale=pivotScale,deg=deg)
                                    plt.savefig(WD+'OutputPlot.png')
                                    asymNRfull=200*np.abs(Solver.analNrun2-Solver.nrun)/np.abs(Solver.analNrun2+Solver.nrun)
                                    asymNRhalf=200*np.abs(Solver.analNrunHalf-Solver.nrun)/np.abs(Solver.analNrunHalf+Solver.nrun)
                                    asymNS=200*np.abs((Solver.analNs-Solver.ns)/(Solver.analNs+Solver.ns))
                                    asymNR=200*np.abs((Solver.nrun-Solver.analNrun)/(Solver.nrun+Solver.analNrun))
                                    if Solver.nrunrun is not None:
                                        nrr=Solver.nrunrun
                                    else:
                                        nrr=0
                                    result="{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13};{14};{15};{16};{17};{18};{19};{20};{21};{22};{23}".format(l1,l2,l3,l4,l5,l6,Solver.coeffs[0],Solver.coeffs[1],Solver.coeffs[2],Solver.coeffs[3],Solver.coeffs[4],Solver.coeffs[5],Solver.coeffs[6],Solver.phiCMB,phi0,phidot0,H0,efolds,efoldsago,Solver.Slope,Solver.ns,Solver.nrun,Solver.nrunrun,Solver.error)#,assymNS,assymNRfull,assymNRhalf)
                                    io.savemat(WD+'OUTPUT.mat',dict(analNS=Solver.analNs,analNR=Solver.analNrun,ns=Solver.ns,nr=Solver.nrun,nrr=Solver.nrunrun,lk=Solver.lk,lps=Solver.lps,asymNS=asymNS,asymNR=asymNR))
    
                               except Exception as e:
                                    print(e.message)
                                    result=e.message
                        except Exception as e:
                            result="{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};error message:'{11}'".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0,efolds,efoldsago,e.message)
        
                    else:
                        if (solver.message=="Too long"):
                            print("Too long")
                            result="{0};{1};{2};{3};{4};{5};{6};{7};{8}; -- Too long".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0)
        
                        else:
                            print("not enough efolds for physical inflation")
                            result="{0};{1};{2};{3};{4};{5};{6};{7};{8}; -- not enough efolds for physical inflation".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0)
        else:
            if (solver.message=="Too long"):
                print("Too long")
                result="{0};{1};{2};{3};{4};{5};{6};{7};{8}; -- Too long".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0)
        
            else:
                print(solver.message)
                result="{0};{1};{2};{3};{4};{5};{6};{7};{8};{9}".format(V0,l1,l2,l3,l4,l5,phi0,phidot0,H0,solver.message)  
    
    
    print('done')
    input("Press Enter to continue...")