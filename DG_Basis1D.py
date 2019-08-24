# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:59:00 2019

@author: Zach
"""
import numpy as np
import Newton
import copy
import math
def Steady_State(n,deg):
    #Define Area
    A=1
    #Define offset for State Vector
    off=(deg+1)*n
    #Define State Vector
    u=np.matrix(np.ones(((deg+1)*n,1),dtype=float))
    u_offset=0*off
    rho=np.matrix(np.ones(((deg+1)*n,1),dtype=float))
    rho_offset=1*off
    T=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    T_offset=2*off
    h=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    h_offset=3*off
    P=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    P_offset=4*off
    offset=[u_offset,rho_offset,T_offset,h_offset,P_offset]
    state=np.r_[u,rho,T,h,P]
    #Define State Vector
    phi=Basis_Func(deg)
    mesh=Linspace(0,1,n+1)
    BC=5
    Coeff=np.matrix(np.zeros((3*deg,(deg+1)**3),dtype=float))
    Ints=np.matrix(np.zeros(((deg+1)**3,1),dtype=float))
    for j in range(0,deg+1):
        for m in range(0,deg+1):
            for k in range(0,deg+1):
                num=k+(deg+1)*m+(deg+1)**2*j
                Coeff[:,num]=Poly_Mult(phi[:,k],phi[:,m],Poly_Diff(phi[:,j]))
                Ints[num,0]=Gauss(-1,1,Poly,Coeff[:,num])
                
    def Navier(state,n,deg,phi,mesh,offset,A,BC,Coeff,Ints):
        extra=Continuity(state,n,deg,phi,mesh,offset,A,BC,Coeff,Ints)
        fills=Filling(state,n,deg,phi,mesh,offset,A,BC,Coeff,Ints)
        return np.r_[extra,fills]
    Sol=Newton.Newton_Solve(state,Navier,n,deg,phi,mesh,offset,A,BC,Coeff,Ints)
    return Sol

def Continuity(state,n,deg,basis,mesh,offset,A,BC,Coeff,Ints):
    #Define offsets from offset vector
    u_offset=offset[0]
    rho_offset=offset[1]
    Mass=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    count=1
    #Deal with Boundary Conditions
    Mass[0]=state[u_offset]-BC
    for j in range(1,deg+1):
        for m in range(0,deg+1):
            for k in range(0,deg+1):
                num=k+(deg+1)*m+(deg+1)**2*j
                if m == k and m==j and k ==deg:
                    Flux=A*state[rho_offset+k]*state[u_offset+m]
                else:
                    Flux=0
                
                Mass[count]=Mass[count]+(Flux-
                        A*state[rho_offset+k]*state[u_offset+m]*Ints[num,0])
                
        count=count+1
    for i in range(1,n):
        for j in range(0,deg+1):
            for m in range(0,deg+1):
                for k in range(0,deg+1):
                    num=k+(deg+1)*m+(deg+1)**2*j
                    if m==k and m==j and k == 0:
                        Flux=-A*state[rho_offset+k+(i-1)*(deg+1)]*state[u_offset+m+(i-1)*(deg+1)]
                    elif m==k and m==j and k == deg:
                        Flux=A*state[rho_offset+k+i*(deg+1)]*state[u_offset+m+i*(deg+1)]
                    else:
                        Flux=0
                    Mass[count]=Mass[count]+(Flux-
                        A*state[rho_offset+k+(deg+1)*i]*state[u_offset+m+(deg+1)*i]*
                        Ints[num,0])
            count=count+1
    return Mass

def Filling(state,n,deg,basis,mesh,offset,A,BC,Coeff,Ints):
    #Define offsets from offset vector
    rho_offset=offset[1]
    T_offset=offset[2]
    h_offset=offset[3]
    P_offset=offset[4]
    output=np.matrix(np.zeros((4*(deg+1)*n,1),dtype=float))
    for i in range(0,(deg+1)*n):
        output[rho_offset-(deg+1)*n+i,0]=state[rho_offset+i]-3
        output[T_offset-(deg+1)*n+i,0]=state[T_offset+i]-3
        output[h_offset-(deg+1)*n+i,0]=state[h_offset+i]-3
        output[P_offset-(deg+1)*n+i,0]=state[P_offset+i]-3
    output[rho_offset-(deg+1)*n,0]=1/BC-state[rho_offset]
    return output
#==============================================================================

def Linspace(x_start,x_end,n):
    #Generates a vector of n linearly spaced points between x_start and x_end
    #Inputs:    x_start=lower bound
    #           x_end=upper bound
    #           n=number of equally spaced points
    #Outputs:   output: Vector of n equally spaced points between x_start and
    #                   x_end
    #==========================================================================
    #Initialize output vector
    output=np.matrix(np.zeros((n,1),dtype=float))
    #Set first point of the output vector to x_start
    output[0]=x_start
    #Determine step size for linearly spaced points
    dx=(x_end-x_start)/(n-1)
    #Add dx to each point to determine output vector
    for i in range(1,n):
        output[i]=output[i-1]+dx
    return output
#==============================================================================
    
def Basis_Func(n):
    #Generates the polynomial basis functions for the reference space [-1,1] of
    #degree n
    #Inputs:    n=degree of basis functions to be generated
    #Outputs:   Coeff: Coefficient matrix of polynomial basis functions
    #                  Each column of Coeff corresponds to a unique basis
    #==========================================================================
    vec=Linspace(-1,1,n+1)
    V=Vandermonde(vec)
    b=np.matrix(np.zeros((n+1,1),dtype=float))
    Coeff=np.matrix(np.zeros((n+1,n+1),dtype=float))
    for i in range(0,n+1):
        b[i]=1
        Coeff[:,i]=Newton.Linear_Solve(V,b)
        b[i]=0
    return Coeff
#==============================================================================
    
def Real_2_Ref(a,b,x):
    #Projects x in [a,b] to eta in [-1,1]
    #Inputs: a=lower bound of real interval
    #        b=upper bound of real interval
    #        x=point in [a,b] to be projected
    #Outputs:projection of x into [-1,1]
    #==========================================================================    
    return (2*x-a-b)/(b-a)
#==============================================================================
    
def Ref_2_Real(a,b,z):
    #Projects z in [-1,1] to x in [a,b]
    #Inputs: a=lower bound of real interval
    #        b=upper bound of real interval
    #        z=reference point in [-1,1] to be projected into [a,b]
    #Outputs:projection of z into [a,b]
    #==========================================================================
    return ((b-a)*z+b+a)/2
#==============================================================================
    
def Poly(x, coeff):
    #Evaluates a 1D polynomial at point x: p(x)=c_0+c_1*x+c_2*x^2+...+c_n*x^n
    #Input:    x: point for p(x) to be evaluated at
    #          coeff: vector of polynomial coefficients defined as
    #                 coeff[0]=c_0,coeff[1]=c_1,...,coeff[n]=c_n
    #output:   Value of p(x) at x
    #==========================================================================
    n=len(coeff)
    output=0
    for i in range(0,n):
        output=output+coeff[i]*x**(i)
    return output
#==============================================================================
    
def Poly_Mult(*coeff):
    #Multiplies 2 or more polynomials together
    #Input: *coeff: Coefficinets of polynomials to be multiplied together
    #               The coefficients for each polynomial should be a separate 
    #               vector
    #Output: output: Coefficient vector of the resulting polynomial
   #===========================================================================
    n=len(coeff)
    output=copy.deepcopy(coeff[0])
    for i in range(1,n):
        p=len(output)
        q=len(coeff[i])
        output_new=np.matrix(np.zeros((p+q-1,1),dtype=float))
        for k in range(0,p):
            for l in range(0,q):
                output_new[k+l]=output_new[k+l]+output[k]*coeff[i][l]
        output=output_new
    return output
#==============================================================================
    
def Poly_Diff(coeff):
    #Input: Coefficients of polynomial p(x)=c0+c_1*x+c_2*x^2+...+c_n*x^n
    #Output: Coefficients of differentiated polynomial of p(x)
    #        p'(x)=c_1+2c_2*x+...+n*c_n*x^{n-1}
    #Function to differentiate a 1D polynomial. Returns the coefficients of the 
    #differentiated polynomial
    #==========================================================================
    #Determine the size of the problem
    n=len(coeff)
    #Preallocate space for the output matrix
    output=np.matrix(np.zeros((n-1,1),dtype=float))
    #Differentiates the polynomial
    for i in range(0,n-1):
        output[i]=coeff[i+1]*(i+1)
    #Returns differentiated polynomial coefficients
    return output
#==============================================================================
    
def Vandermonde(x):
    #Generates a Vandermonde matrix for a vector point x
    #Input: vector x of real values
    #Output: nxn Vandermonde matrix
    #==========================================================================
    #Determine size of x
    n=len(x)
    #Initialize space for Vandermonde matrix
    V=np.matrix(np.ones((n,n),dtype=float))
    #GEnerate Vandermonde Matrix
    for i in range(0,n):
        for j in range(1,n):
            V[i,j]=x[i]**j
    return V
#==============================================================================
    
def Gauss(a,b,func,*arg):
    #Numerically integrates a real 1D function from a to b using 9pt Gaussian
    #Quadrature. 
    #Inputs: a=lower integral bound
    #        b=upper integral bound
    #        func= real 1D function to be integrated
    #        *arg= additional inputs to be passed into func
    #==========================================================================
    #Initialize Gaussian quadrature weights
    w=np.matrix([[0.4179591836734694],[0.3818300505051189],[0.3818300505051189],[0.2797053914892766],[0.2797053914892766],[0.1294849661688697],[0.1294849661688697]])
    #Initialize Gaussian quadrature abissica
    ab=np.matrix([[0.0000000000000000],[0.4058451513773972],[-0.4058451513773972],[-0.7415311855993945],[0.7415311855993945],[-0.9491079123427585],[0.9491079123427585]])
    #Initialize the integral value at zero
    Int=0
    #Initialize relative error in the integral
    error=1
    #Initialize refinement counter
    count = 0
    #Relative error counter
    tol = 10**(-3)
    #determine if there are additional input arguments for func
    s=len(arg)
    #Initialize number of intervals at 2
    n=2
    x=Linspace(a,b,n)
    #Cycle through all Gaussian Quadrature points
    for i in range(0,7):
        #Transform real points to reference space
        loc=Ref_2_Real(x[0],x[1],ab[i])
        #Evaluate gaussian quadrature pointss
        if s>0:
            Int=Int+w[i]*(x[1]-x[0])/2*func(loc,*arg)
        else:
            Int=Int+w[i]*(x[1]-x[0])/2*func(loc)
    #Mesh refinement
    while error > tol and count < 5:
        #Define new integral
        Int_new=0
        #Refine Mesh
        n=n*2
        #Define new mesh x
        x=Linspace(a,b,n)
        #Cycle through all points
        for j in range(1,n):
            #Cycle through Gauassian Quadrature
            for i in range(0,7):
                #Transform integration interval to reference space
                loc=Ref_2_Real(x[j-1],x[j],ab[i])
                #Evaluate Gaussian Quadrature POints
                if s > 0:
                    Int_new=Int_new+w[i]*(x[j]-x[j-1])/2*func(loc,*arg)
                else:
                    Int_new=Int_new+w[i]*(x[j]-x[j-1])/2*func(loc)
        #Determine Relative Error
        if Int_new <= tol:
            error=abs(Int-Int_new)
        else:
            error=abs(Int-Int_new)/abs(Int_new)
        #Assign Current Integral estimate to previous estimate
        Int=Int_new
        #Iterate maximum iteration counter
        count=count+1
    return Int
#==============================================================================

def Simpson(a,b,func):
    #Performs adaptive quadrature until convergence occurs
    tol=10**-4
    n=2
    x_vec=Linspace(a,b,n+1)
    h=(b-a)/n
    Int=1/3*h*(func(x_vec[0])+4*func(x_vec[1])+func(x_vec[2]))
    count = 0
    error=1
    while count < 12 and error>tol:
        Even=0
        Odd=0
        n=n*2
        h=(b-a)/n
        x_vec=Linspace(a,b,n+1)
        for i in range(1,n-1):
            if i%2 == 0:
                Even=Even+func(x_vec[i])
            else:
                Odd=Odd+func(x_vec[i])    
        Int_new=1/3*h*(func(a)+2*Odd+4*Even+func(b))
        if Int_new == 0:
            error=abs(Int-Int_new)
        else:
            error=abs(Int-Int_new)/abs(Int_new)
        count=count+1
        Int=Int_new
    return Int
#==============================================================================
    
def Lagrange(x_vec,y_vec,z):
    n=len(x_vec)
    l_basis=np.matrix(np.ones((n,1),dtype=float))
    output=0
    for i in range(0,n):
        for j in range(0,n):
            if i != j:
                l_basis[i]=l_basis[i]*(z-x_vec[j])/(x_vec[i]-x_vec[j])
        output=output+l_basis[i]*y_vec[i]
    return output
#==============================================================================