# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:59:00 2019

@author: Zach
"""
import numpy as np
import Newton
import copy
import math
import time
import matplotlib.pyplot as plt
def Steady_State(n,deg):
    Initial_start=time.time()
    #Define Area
    A=1
    #Define offset for State Vector
    off=(deg+1)*n
    #Define initial conditions
    u=np.matrix(np.ones(((deg+1)*n,1),dtype=float))*1
    rho=np.matrix(np.ones(((deg+1)*n,1),dtype=float))*1
    T=np.matrix(np.ones(((deg+1)*n,1),dtype=float))*300
    h=np.matrix(np.ones(((deg+1)*n,1),dtype=float))*300
    P=np.matrix(np.ones(((deg+1)*n,1),dtype=float))*101000
    
    u_offset=0*off
    rho_offset=1*off
    P_offset=2*off
    T_offset=3*off
    h_offset=4*off
    offset=[u_offset,rho_offset,T_offset,h_offset,P_offset]
    state=np.r_[u,rho,P,T,h]
    #Define State Vector
    phi=Basis_Func(deg)
    mesh=Linspace(0,1,n+1)
    #Mass Flow Rate kg/s
    m_flow=1
    #Molecular weight kg/mol
    MW=.01801528
    #Ideal gas constant
    R=8.314
    #Inlet Pressure
    P0=101000
    #Enthalpy inlet condition
    h_bc=300
    Cp=1.05
    U=10
    print("Pre-Calculculating Integrals for Steady State Solver\n")
    Integration_start=time.time()
    Coeff_3_1=np.matrix(np.zeros((4*deg,(deg+1)**4),dtype=float))
    Coeff_3_0=np.matrix(np.zeros((3*deg+1,(deg+1)**3),dtype=float))
    Coeff_2_1=np.matrix(np.zeros((3*deg,(deg+1)**3),dtype=float))
    Coeff_2_0=np.matrix(np.zeros((2*deg+1,(deg+1)**2),dtype=float))
    Coeff_1_1=np.matrix(np.zeros((2*deg,(deg+1)**2),dtype=float))
    Coeff_1_m1=np.matrix(np.zeros((3*deg,(deg+1)**3),dtype=float))
    Ints_3_1=np.matrix(np.zeros(((deg+1)**4,1),dtype=float))
    Ints_3_0=np.matrix(np.zeros(((deg+1)**3,1),dtype=float))
    Ints_2_1=np.matrix(np.zeros(((deg+1)**3,1),dtype=float))
    Ints_2_0=np.matrix(np.zeros(((deg+1)**2,1),dtype=float))
    Ints_1_1=np.matrix(np.zeros(((deg+1)**2,1),dtype=float))
    Ints_1_m1=np.matrix(np.zeros(((deg+1)**3,1),dtype=float))
    T_ints=Enthalpy_T_Int(phi,mesh,n,deg)
    #Pre-Calculate Integration Parameters
    for j in range(0,deg+1):
        for m in range(0,deg+1):
            num=m+(deg+1)*j
            Coeff_1_1[:,num]=Poly_Mult(phi[:,m],Poly_Diff(phi[:,j]))
            Coeff_2_0[:,num]=Poly_Mult(phi[:,m],phi[:,j])
            Ints_1_1[num,0]=Gauss(-1,1,Poly,Coeff_1_1[:,num])
            Ints_2_0[num,0]=Gauss(-1,1,Poly,Coeff_2_0[:,num])
            for k in range(0,deg+1):
                num2=k+(deg+1)*m+(deg+1)**2*j
                Coeff_2_1[:,num2]=Poly_Mult(phi[:,k],phi[:,m],Poly_Diff(phi[:,j]))
                Coeff_3_0[:,num2]=Poly_Mult(phi[:,k],phi[:,m],phi[:,j])
                Coeff_1_m1[:,num2]=Poly_Mult(phi[:,m],Poly_Diff(Poly_Mult(phi[:,k],phi[:,j])))
                Ints_2_1[num2,0]=Gauss(-1,1,Poly,Coeff_2_1[:,num2])
                Ints_3_0[num2,0]=Gauss(-1,1,Poly,Coeff_3_0[:,num2])
                Ints_1_m1[num2,0]=Gauss(-1,1,Poly,Coeff_1_m1[:,num2])
                for l in range(0,deg+1):
                    num3=l+k*(deg+1)+m*(deg+1)**2+j*(deg+1)**3
                    Coeff_3_1[:,num3]=Poly_Mult(phi[:,l],phi[:,k],phi[:,m],Poly_Diff(phi[:,j]))
                    Ints_3_1[num3,0]=Gauss(-1,1,Poly,Coeff_3_1[:,num3]) 
    Integration_end=time.time()
    print("Completed Pre-Integration for Steady State Solver in ",Integration_end-Integration_start, "seconds\n")
    def Navier(state,n,deg,phi,mesh,offset,A,m_flow,Coeff_2_1,Ints_1_1,Ints_2_1,Ints_3_1,
               Ints_1_m1,P0,T_ints,R,MW,h_bc,Cp,U):
        Cont=SS_Continuity(state,n,deg,offset,A,m_flow)
        Dense=Density(state,n,deg,offset,R,MW)
        Press=Momentum(state,n,deg,offset,P0,Ints_1_1,Ints_3_1,A)
        Temp=Temperature(state,n,deg,offset,Cp)
        Enth=Enthalpy(state,phi,mesh,n,deg,offset,A,Ints_3_1,Ints_1_m1,Ints_2_0,Ints_2_1,T_ints,h_bc,U)
        return np.r_[Cont,Dense,Press,Temp,Enth]
    
    Solver_start=time.time()
    
    Sol=Newton.Fast_Newton_Solve(state,Navier,A,n,deg,phi,mesh,offset,A,m_flow,Coeff_2_1,Ints_1_1,Ints_2_1,Ints_3_1,
                                Ints_1_m1,P0,T_ints,R,MW,h_bc,Cp,U)
    Solver_end=time.time()
    print("Newton Solver converged in ",(Solver_end-Solver_start), "seconds\n")
    print("Total run time is ",(Solver_end-Initial_start), "seconds")
    return Sol

def Continuity(state,n,deg,basis,mesh,offset,A,BC,Coeff,Ints):
    #This implementation needs to be fixed before I move onto the temporal discretization
    #Define offsets from offset vector
    u_offset=offset[0]
    rho_offset=offset[1]
    Mass=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    count=0
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
#==============================================================================
    
def SS_Continuity(state,n,deg,offset,A,m_flow):
    u_offset=offset[0]
    rho_offset=offset[1]
    Boundary=np.matrix(np.ones(((deg+1)*n,1),dtype=float))*m_flow
    Output=A*np.multiply(state[u_offset:n*(deg+1)],state[rho_offset:rho_offset+n*(deg+1)])-Boundary
    return Output
#==============================================================================
    
def Momentum(state,n,deg,offset,P0,Ints_1_1,Ints_3_1,A):
    puu=Momentum_puu(state,n,deg,offset,Ints_3_1,A)
    dpdx=Momentum_dp(state,n,deg,offset,P0,Ints_1_1,A)
    return dpdx-puu
#==============================================================================
def Momentum_puu(state,n,deg,offset,Ints_3_1,A):
    u_offset=offset[0]
    rho_offset=offset[1]
    output=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    Flux=np.matrix(np.zeros(((deg+1)**4,1),dtype=float))
    #Cycle through all of the elements
    for i in range(0,n):
        #Pull out the relavent state vectors for the element
        u_vec=state[u_offset+i*(deg+1):u_offset+(i+1)*(deg+1)]
        rho_vec=state[rho_offset+i*(deg+1):rho_offset+(i+1)*(deg+1)]
        if i != 0:
            Flux[0]=-rho_u_u_tensor[-1]
        #Compute tensor product
        u_u_tensor=np.reshape(np.outer(u_vec,u_vec),((deg+1)**2,1))
        rho_u_u_tensor=np.reshape(np.outer(rho_vec,u_u_tensor),((deg+1)**3,1))
        #Create copies for calculating integration
        full_tensor=np.tile(rho_u_u_tensor,(deg+1,1))
        #Calculate integration
        Integral=np.multiply(full_tensor,Ints_3_1)
        Flux[-1]=rho_u_u_tensor[-1]
        Out=Flux-Integral
        #Compute action of jth basis function
        for j in range(0,deg+1):
            output[i*(deg+1)+j]=np.sum(Out[j*(deg+1)**3:(j+1)*(deg+1)**3])
    output[0]=0
    return output
#==============================================================================
def Momentum_dp(state,n,deg,offset,P0,Ints_1_1,A):
    p_offset=offset[4]
    output=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    Flux=np.matrix(np.zeros(((deg+1)**2,1),dtype=float))
    #Cycle through all of the elements
    for i in range(0,n):
        if i != 0:
            Flux[0]=-p_vec[-1]
        #pull out the relavent state vector
        p_vec=state[p_offset+i*(deg+1):p_offset+(i+1)*(deg+1)]
        #Create copies for calculating integration
        full_tensor=np.tile(p_vec,(deg+1,1))
        #Calculate integration
        Integral=np.multiply(full_tensor,Ints_1_1)
        #Finish computing the flux
        Flux[-1]=p_vec[-1]
        Out=Flux-Integral
        #Compute the action of the jth basis function
        for j in range(0,deg+1):
            output[i*(deg+1)+j]=np.sum(Out[j*(deg+1):(j+1)*(deg+1)])
    #Enforce pressure boundary condition
    output[0]=P0-state[p_offset]
    return output
#==============================================================================
def Density(state,n,deg,offset,R,MW):
    #Define offsets from offset vector
    rho_offset=offset[1]
    T_offset=offset[2]
    P_offset=offset[4]
    output2=(MW*state[P_offset:P_offset+n*(deg+1)]
            -R*np.multiply(state[rho_offset:rho_offset+n*(deg+1)],state[T_offset:T_offset+n*(deg+1)]))
    return output2
#==============================================================================

def Temperature(state,n,deg,offset,Cp):
    #Define offsets from offset vector
    #Computes the Temperature as a function of enthalpy
    T_offset=offset[2]
    h_offset=offset[3]
    output=Cp*state[T_offset:T_offset+n*(deg+1)]-state[h_offset:h_offset+n*(deg+1)]
    return output
#==============================================================================
    
def Enthalpy(state,phi,mesh,n,deg,offset,A,Ints_3_1,Ints_1_m1,Ints_2_0,Ints_2_1,T_ints,h_bc,U):
    #Computes the 1D Enthalpy equation
    output=(Enthalpy_Term(state,n,deg,offset,A,Ints_3_1,h_bc)
            -Enthalpy_Press(state,n,deg,offset,A,Ints_1_m1)
            -Enthalpy_Heat(state,phi,T_ints,mesh,n,deg,offset,A,Ints_2_0,U))
    return output
#==============================================================================
    
def Enthalpy_Term(state,n,deg,offset,A,Ints_3_1,h_bc):
    #Computes d(rho*h*u)/dx term in enthalpy equation
    u_offset=offset[0]
    rho_offset=offset[1]
    h_offset=offset[3]
    output2=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    Flux=np.matrix(np.zeros(((deg+1)**4,1),dtype=float))
    for i in range(0,n):
        h_vec=state[h_offset+i*(deg+1):h_offset+(i+1)*(deg+1)]
        u_vec=state[u_offset+i*(deg+1):u_offset+(i+1)*(deg+1)]
        rho_vec=state[rho_offset+i*(deg+1):rho_offset+(i+1)*(deg+1)]
        #Calculate flux (reuse previous tensor to avoid extra calcs)
        if i != 0:
            Flux[0]=-rho_u_h_tensor[-1]
        u_h_tensor=A*np.reshape(np.outer(u_vec,h_vec),((deg+1)**2,1))
        rho_u_h_tensor=A*np.reshape(np.outer(rho_vec,u_h_tensor),((deg+1)**3,1))
        #Create copies for calculating integration
        full_tensor=np.tile(rho_u_h_tensor,(deg+1,1))
        #Calculate integration
        Integral=np.multiply(full_tensor,Ints_3_1)
        Flux[-1]=rho_u_h_tensor[-1]
        Out=Flux-Integral
        for j in range(0,deg+1):
            output2[i*(deg+1)+j]=np.sum(Out[j*len(rho_u_h_tensor):(j+1)*len(rho_u_h_tensor)])
    #Enforce enthalpy boundary condition
    output2[0]=h_bc-state[h_offset]
    return output2
#==============================================================================
def Enthalpy_Press(state,n,deg,offset,A,Ints_1_m1):
    #Computes u*dp/dx term in enthalpy equation
    u_offset=offset[0]
    p_offset=offset[4]
#    output=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    output2=np.matrix(np.zeros((n*(deg+1),1),dtype=float))
    Flux=np.matrix(np.zeros(((deg+1)**3,1),dtype=float))
    #Cycle through all elements
    for i in range(0,n):
        #Pull the pressure and velocity vectors for this element
        p_vec=state[p_offset+i*(deg+1):p_offset+(i+1)*(deg+1)]
        u_vec=state[u_offset+i*(deg+1):u_offset+(i+1)*(deg+1)]
        #Calculate flux
        if i != 0:
            Flux[0]=-p_u_tensor[-1]
        #Calculate outer product of p and u
        p_u_tensor=A*np.reshape(np.outer(p_vec,u_vec),((deg+1)**2,1))
        #Create copies for calculating integration
        full_tensor=np.tile(p_u_tensor,(deg+1,1))
        #Calculate integration
        Integral=np.multiply(full_tensor,Ints_1_m1)
        Flux[-1]=p_u_tensor[-1]
        #Calculate the action of integration
        Out2=Flux-Integral
        #Calculate the integral for each basis vector j on element i
        for j in range(0,deg+1):
            output2[i*(deg+1)+j]=np.sum(Out2[j*(deg+1)**2:(j+1)*(deg+1)**2])
    #Set first term to zero for boundary equation
    output2[0]=0
    return output2
#==============================================================================
    
def Enthalpy_Heat(state,phi,T_ints,mesh,n,deg,offset,A,Ints_2_0,U):
    #Calculate the perimiter of the pipe
    Perim=2*math.sqrt(A*np.pi)
    T_offset=offset[2]
    #Reshape the surrounding temperature function into a vector
    T_ints2=np.reshape(T_ints,(n*(deg+1),1))
    output2=np.matrix(np.zeros(((deg+1)*n,1),dtype=float))
    #Cycle through all elements in the mesh
    for i in range(0,n):
        #Calculate the length of the element (for non-uniform meshes)
        dx=mesh[i+1]-mesh[i]
        #Pull relavent state variable for the element
        T_vec=state[T_offset+i*(deg+1):T_offset+(i+1)*(deg+1)]
        #Create copies for calculating integration
        full_tensor=np.tile(T_vec,(deg+1,1))
        #Calculate integration
        Integral2=np.multiply(full_tensor,Ints_2_0)
        #Calculate the action of the j'th basis function
        for j in range(0,deg+1):
            output2[j+i*(deg+1)]=T_ints2[j+i*(deg+1)]-dx/2*np.sum(Integral2[j*(deg+1):(j+1)*(deg+1)])
    #multiply through by U and the perimeter
    output2=output2*U*Perim
    #Set the first term to zero for the boundary equation
    output2[0]=0
    return output2
#==============================================================================

def T_Surr(x):
    #Defines the surrounding temerature as a function of x
    return 290
#==============================================================================
    
def Enthalpy_T_Int(phi,mesh,n,deg):
    #Integrates the surrounding temperature function along the mesh
    output=np.matrix(np.zeros((n,deg+1),dtype=float))
    def func(z,T_func,phi,a,b):
        return T_func(Ref_2_Real(a,b,z))*Poly(z,phi)
    for i in range(0,n):
        a=mesh[i]
        b=mesh[i+1]
        for j in range(0,deg+1):
            output[i,j]=(b-a)/2*Gauss(-1,1,func,T_Surr,phi[:,j],a,b)
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
def DG_Mesh(mesh,deg):
    Elements=len(mesh)-1
    dg_mesh=np.matrix(np.zeros(((deg+1)*Elements,1),dtype=float))
    for i in range(0,Elements):
        a=mesh[i]
        b=mesh[i+1]
        for j in range(0,deg+1):
            dg_mesh[j+i*(deg+1)]=mesh[i]+j*(b-a)/deg
    return dg_mesh
#==============================================================================
def DG_Plot(mesh,state):
    Elements=len(mesh)-1
    deg=int(len(state)/Elements-1)
    pts=101
    phi=Basis_Func(deg)
    refmesh=Linspace(-1,1,pts)
    store=np.matrix(np.zeros((101,1),dtype=float))
    for i in range(0,Elements):
        func=np.multiply(state[i*(deg+1):(i+1)*(deg+1)].T,phi)
        Coeff=sum(func.T).T
        real_mesh=Linspace(mesh[i],mesh[i+1],pts)
        for j in range(0,pts):
            store[j]=Poly(refmesh[j],Coeff)
        plt.plot(real_mesh,store,'C0')