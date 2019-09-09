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
plt.close('all')
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
runs=1
avg=0
time=np.matrix(np.zeros((runs,1),dtype=float))
for i in range(0,runs):
    Sol7,time[i]=DG.Steady_State(10,2)
    avg=time[i]+avg
avg=avg/runs
#Finish Implemented the Momentum equation need d(pu^2)/dx term
#Try to fix continuity equation implementation (want generic version)
#Implement a Fast Jacobian Computation
#Define Area
#n=1
#deg=2
#A=1
##Define offset for State Vector
#off=(deg+1)*n
##Define State Vector
#u=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
#rho=np.matrix(np.ones(((deg+1)*n,1),dtype=float))
#T=np.matrix(np.ones(((deg+1)*n,1),dtype=float))
#h=np.matrix(np.ones(((deg+1)*n,1),dtype=float))
#P=np.matrix(np.ones(((deg+1)*n,1),dtype=float))
# 
#u_offset=0*off
#rho_offset=1*off
#P_offset=2*off
#T_offset=3*off
#h_offset=4*off
#offset=[u_offset,rho_offset,T_offset,h_offset,P_offset]
##state=np.r_[u,rho,T,h,P]
#state=np.r_[u,rho,P]
##Define State Vector
#phi=DG.Basis_Func(deg)
#mesh=DG.Linspace(0,1,n+1)
#BC=4
#Coeff_3_0=np.matrix(np.zeros((3*deg+1,(deg+1)**3),dtype=float))
#Coeff_2_1=np.matrix(np.zeros((3*deg,(deg+1)**3),dtype=float))
#Coeff_1_1=np.matrix(np.zeros((2*deg,(deg+1)**2),dtype=float))
#Ints_3_0=np.matrix(np.zeros(((deg+1)**3,1),dtype=float))
#Ints_2_1=np.matrix(np.zeros(((deg+1)**3,1),dtype=float))
#Ints_1_1=np.matrix(np.zeros(((deg+1)**2,1),dtype=float))
#P0=100
#for j in range(0,deg+1):
#    for m in range(0,deg+1):
#        num2=m+(deg+1)*j
#        Coeff_1_1[:,num2]=DG.Poly_Mult(phi[:,m],DG.Poly_Diff(phi[:,j]))
#        Ints_1_1[num2,0]=DG.Gauss(-1,1,DG.Poly,Coeff_1_1[:,num2])
#        for k in range(0,deg+1):
#            num=k+(deg+1)*m+(deg+1)**2*j
#            Coeff_2_1[:,num]=DG.Poly_Mult(phi[:,k],phi[:,m],DG.Poly_Diff(phi[:,j]))
#            Ints_2_1[num,0]=DG.Gauss(-1,1,DG.Poly,Coeff_2_1[:,num])
#Tester=DG.Momentum(Sol,n,deg,phi,mesh,offset,P0,Ints_1_1,A)
#Basis_Functions=DG.Basis_Func(2)
#point=DG.Poly(0,Basis_Functions[:,0])