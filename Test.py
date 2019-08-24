# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:36:23 2019

@author: Zach
"""

import numpy as np
import func
import Newton
import DG_Basis1D as DG
import matplotlib.pyplot as plt
import math
#plt.close('all')
#n=2
#Coef=DG.Basis_Func(n)
#xspan=DG.Linspace(-1,1,101)
#Q=np.matrix(np.zeros((101,n+1),dtype=float))
#Y=np.matrix(np.zeros((101,n+1),dtype=float))
#Z=np.matrix(np.zeros((101,n+1),dtype=float))
#def myFunction(x):
#    return DG.Poly(x,Coef[:,1])
#
#for j in range(0,n+1):
#    for i in range(0,101):
#        Y[i,j]=DG.Poly(xspan[i],Coef[:,j])
#        Z[i,j]=Newton.Derivative(xspan[i],DG.Poly,Coef[:,j])
#        Q[i,j]=DG.Poly(xspan[i],DG.Poly_Diff(Coef[:,j]))
#Diff_Coeff=DG.Poly_Diff(Coef[:,0])
#
#Test=DG.Poly_Mult(Coef[:,0],Diff_Coeff)
#
#for i in range(0,n+1):   
#    plt.figure(0)
#    plt.plot(xspan,Y[:,i])
#    plt.figure(1)
#    plt.plot(xspan,Z[:,i])
Mass=DG.Steady_State(1,1)