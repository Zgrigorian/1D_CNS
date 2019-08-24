# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:46:02 2019

@author: Zach
"""
import numpy as np
import copy 
import math

def Newton_Opt(x0,func):
    count=0
    tol=10 **(-9)
    x=copy.deepcopy(x0)
    def F(arg):
        return 1/2*func(arg).T*func(arg)
    error=Two_Norm(F(x))
    while count < 100 and error > tol:
        J=Derivative(x,F).T
        H=Hessian(x,F)
        p=Linear_Solve(H,-J)
        x=x+p
        count = count +1
        error=Two_Norm(F(x))
        print("Iteration", count, "\n")
        print("Error", "{:.2e}".format(error), "\n")
    return x

def Newton_Solve(x0,func,*arg):
    #This uses newton's method to compute the solution to a non-linear system
    #of equations. This does not employ a quadratic cost function. The 
    #algorithm employed is x_{n+1}=x_{n}+J^{-1}f(x_n). This is transformed into
    #a solvable linear system via: J*(x_{n+1}-x_{n})=f(x_n)
    #==========================================================================
    #Initialize the counter for the maximum number of iterations allowed
    count = 0
    s=len(arg)
    #Copy my initial starting value to avoid changing the original
    x=copy.deepcopy(x0)
    #Initialize my tolerance criteria
    tol = 10**(-5)
    #Initialize my error
    if s > 0:
        error = Two_Norm(func(x0,*arg))
    else:
        error = Two_Norm(func(x0))
    #Perform newton's method
    print("Beginning Newton's Method\n")
    while count < 100 and abs(error) > tol:
        print("Iteration",count,"\n")
        if s > 0:
            #Compute the Jacobian at the current x value
            print("Computing Numerical Jacobian\n")
            J=Derivative(x,func,*arg)
            #Solve the resulting linear system for the change in x
            print("Solving Linear System\n")
            p=Linear_Solve(J,-func(x,*arg))
        else:
            #Compute the Jacobian at the current x value
            print("Computing Numerical Jacobian\n")
            J=Derivative(x,func)
            #Solve the resulting linear system for the change in x
            print("Solving Linear System\n")
            p=Linear_Solve(J,-func(x))
        #Update x
        x=x+p
        #increment the iteration counter
        count = count +1
        #Print results so I know its working
        if s > 0:
            error = Two_Norm(func(x,*arg))
        else:
            error = Two_Norm(func(x))
        print("Error", "{:.2e}".format(error), "\n")
    #Output the root of the system
    return x

def Grad_Descent(x0,func):
    #Implements Gradient descent with momentum and a line search. A 
    #backtracking line search is utilized. The algorithm is based on the 
    #Armijo–Goldstein condition. Given a search direction p, and step length 
    #alpha the condition is satisfied once:
    #f(x+alpha*p)<f(x)+alpha*c*p^T*grad(f(x))
    #With momentum, the previous step is remembered so we can speed up the
    #convergence and avoid saddle points (since hopefull the previous step will
    #push us through the saddle point). The amount of momentum can be adjusted
    #with the hyperparameter beta
    #==========================================================================
    #Copy the original point to avoid changing it
    #Real step
    x=copy.deepcopy(x0)
    #Momentum remember
    z=copy.deepcopy(x0)
    #Initialize Jacobian of the function
    J=Derivative(x0,func)
    #Initialize counter for stopping infinite loops
    count = 0
    #Initialize error calculation
    error=Two_Norm(func(x))
    #Hyperparameters
    #Initial step size when starting line search
    alpha=1
    #Momemtum hyper parameter
    beta=0
    #Line search hyper parameter (alpha fractional reduction)
    tau=0.5
    #Line search hyper parameter for stopping criteria
    c=10**(-4)
    #Error tolerance for the system
    Tol=10**(-5)
    #Initialize m and t for the back tracking line search (keep compactness)
    m=5;
    t=c*m
    #Begin the gradient descent
    while count < 1000 and error > Tol:
        #Backtracking Linesearch in descent direction
        alpha=1;
        p=-J.T*func(x)
        p=p/Two_Norm(p)
        m=p.T*J.T*func(x)
        #I think I can take out t=cm but im not sure
        t=-c*m
        #Backtracking line search
        while func(x).T*func(x)-func(x+alpha*p).T*func(x+alpha*p) <= alpha*t  and alpha>10**-9:
            alpha=alpha*tau
        #Momentum Update
        z=beta*z+p
        #Update new x
        x=x+alpha*z
        #Determine new descent direction
        J=Derivative(x,func)
        #Compute error of the system
        error=Two_Norm(func(x))
        #Increment counter
        count = count +1
        #Print stuff so I know the program is working
        print("Iteration", count, "\n")
        print("Error", "{:.2e}".format(error), "\n")
        print("Step Size","{:.3e}".format(alpha),"\n")
    #Return the solution to the problem
    return x

def Hessian(x0,func,*arg):
    h=10**-6
    #Determine size of the problem
    n=len(x0)
    #Copy x coordinate so original isn't changed
    x_up=copy.deepcopy(x0)
    x_low=copy.deepcopy(x0)
    #Allocates memory for hessian matrix
    Hess=np.matrix(np.zeros((n,n),dtype=float))
    for j in range(0,n):
        for i in range(0,n):
            #Modify x_up/x_low for finite difference of 2nd derivative
            x_up[j]=x_up[j]+h
            x_low[j]=x_low[j]-h
            #Calculate 1st derivative of func at x_up/x_low
            J_up=Derivative(x_up,func,*arg)
            J_low=Derivative(x_low,func,*arg)
            #Calculate the second derivative of func
            Hess[i,j]=(J_up[0,i]-J_low[0,i])/(2*h)
            #Return x_up/x_low to their original status
            x_up[j]=x_up[j]-h
            x_low[j]=x_low[j]+h
    
    #Return the Hessian as the result        
    return Hess
#==============================================================================

def Derivative(x0,func,*arg):
    #Assign small step size for derivative
    s=len(arg)
    h=float(10**(-6));
    #Determine the size of the problem
    test=[];
    for i in range(0,s):
        test.append(arg[i])
    if s>0:
        n=len(func(x0, *arg))
    else:
        n=len(func(x0))
    m=len(x0)
    #Allocate memory for Jacobian
    J=np.matrix(np.zeros((n,m),dtype=float))
    #Copy x0 for finite differences
    x_up=copy.deepcopy(x0);
    x_low=copy.deepcopy(x0);
    #Cycle through all rows
    for i in range(0,n):
        #Cycle through all columns
        for j in range(0,m):
            #Modify x_up/x_low for coordinate of differentiation
            x_up[j]=x_up[j]+h;
            x_low[j]=x_low[j]-h;
            #Calculate derivative
            if s>0:
                J[i,j]=(func(x_up,*arg)[i]-func(x_low,*arg)[i])/(2*h)
            else:
                J[i,j] = (func(x_up)[i]-func(x_low)[i])/(2*h)
            #Return x_up/x_low to their original status
            x_up[j]=x_up[j]-h
            x_low[j]=x_low[j]+h
    #Return the Jacobian as the output
    return J


def Two_Norm(u):
    #Computes the two norm of a vector u. Does not work for matrices
    return math.sqrt(np.dot(u.T,u))
#==============================================================================

def Norm2(u):
    #Computes the two norm of a np.matrix vector u
    n=len(u)
    tot=0
    for i in range(0,n):
        tot=tot+u[i]**2
    return math.sqrt(tot)

def QR(Input):
    #Computes the QR decompsotion of a matrix A using Householder reflections.
    #This algorithm is recommended for solving linear equations due to its 
    #numerical stability
    #==========================================================================
    #Determine the size of the problem
    n=len(Input)
    #Copy the input so we don't change it
    R=copy.deepcopy(Input)
    print(R)
    #Initialize Q
    Q=np.matrix(np.identity(n),dtype=float)
    alpha=0
    for i in range(0,n-1):
        #Initialize Householder matrix
        H=np.matrix(np.identity(n),dtype=float)
        #Generate the identity vector
        e=np.matrix(np.zeros((n-i,1),dtype=float))
        e[0]=1
        #Pull a column vector from A
        x=R[i:n+1,i]
        alpha=np.sign(x.item(0))*Two_Norm(x)
        if alpha - x[0] < 10**-7:
            u=x
        else:
            u=x-alpha*e
        v=u/Two_Norm(u)
        H=np.matrix(np.identity(n),dtype=float)
        H[i:n+1,i:n+1]=H[i:n+1,i:n+1]-2*v*v.T
        R=H*R
        Q=Q*H.T
    #Return Q and R
    return Q,R
#==============================================================================

def Linear_Solve(A,b):
    #This will solve a linear system of equations using QR decomposition. The
    #QR algorithm uses Householder reflections. After the QR decomposition 
    #occurs, the Q matrix is multiplied through and back substitution is used
    #to compute the vector x
    #==========================================================================
    #Use Householder reflections to perform QR decomposition
    Q,R=QR(A)
    #Determine the size of the problem
    n=len(A)
    #Multiply the Q over to the right hand side
    b_hat=Q.T*b
    #Initialze memory for the solution x
    x=np.matrix(np.zeros((n,1),dtype=float))
    #Perform back substitution to solve for x
    for i in range(n-1,-1,-1):
        #Initialize the eventual RHS element
        RHS=b_hat[i]
        for j in range(n-1,i,-1):
            #Move everything but diagonal element of R over
            RHS=RHS-R[i,j]*x[j]
        #Compute this element of x
        x[i]=RHS/R[i,i]
    #Return x
    return x
#==============================================================================
    
def QR_GS(A):
    #This is a function that computes the QR decompositon of a matrix A using
    #the Gram Schmidt algorithm. This is not suggested for real computations as
    #it is numerically unstable
    #==========================================================================
    import math
    #Determine size of our system
    n=len(A)
    #Initialize matrix to store U vectors
    U=np.matrix(np.zeros((n,n),dtype=float))
    #Initialize matrix for Q vectors
    Q=np.matrix(np.zeros((n,n),dtype=float))
    #Cycle through all the columns of the matrix
    for i in range(0,n):
        #Copy column vector of A
        U[:,i]=copy.deepcopy(A[:,i])
        for j in range(0,i):
            #Project the column vector of A onto U
            U[:,i]=U[:,i]-Project(U[:,j],A[:,i])  
        Q[:,i]=U[:,i]/math.sqrt(np.dot(U[:,i].T,U[:,i]))
    return Q
#==============================================================================
    
def Project(u,a):
    #This is a helper function for the Gram Schmidt algorithm
    return u*np.dot(u.T,a)/np.dot(u.T,u)
#==============================================================================