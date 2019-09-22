5# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:36:23 2019
The purpose of this program is to solve the 1D Compressible Navier Stokes(CNS)
equations for 1D pipe flow. All calculations are based on the assumption that 
the flow is turbulent and a circular pipe. A uniform mesh is automatically 
generated, but the solver can handle a user defined non-uniform mesh

The user must define the following inlet boundary conditions:
    P0: pressure
    m_flow: Mass flow rate
    h_bc: Inlet Enthalpy
The user must specify the following pipe parameters:
    A: Area
    D: Diameter
    U: Heat transfer coefficient
    rough: pipe roughness
    T_ambient: Temperature of surrounding air
The user must specify the following gas parameters:
    MW: Molecular weight
    R: Ideal gas constant
    Cp: Heat capacity
    mu: Dynamic Viscosity
The user must specify the following system hyper parameters:
    elements: The number of elements that the discretized mesh will be broken
              up into
    deg: The polynomial degree used to approximate the true solution on each
         element
As a general rule of thumb, the greater the polynomial degree, the fewer
elements needed to achieve accurate results.

The solver uses the 1D discontinuous galerkin method to solve the CNS. A 
derivation of the equations used in this program can be found in the following
thesis: "Modeling, Discontinuous Galerkin Approximation and Simulation of the 
1-D Compressible Navier Stokes Equations" by Zachary Grigorian
https://vtechworks.lib.vt.edu/handle/10919/93197

In this solver no viscous or diffusive effects are used. Additionally, unlike
in the thesis, a non-linear pressure profile is used and the momentum equation
is actually solved.

The system of equations is solved using a Newton solver found in the "Newton" 
package. The Newton solver in the "Newton" package is a vanilla Newton's 
method. A QR factorization method is used to solve all intermediate linear
Systems.
All relevant state variables are plotted after calculations are finished
In the simplified view, only each polynomial coefficient is plotted and 
connected with a straight line.
In the DG view, the actual polynomial on each element is constructed and 
plotted.
@author: Zachary Grigorian
"""

import numpy as np
import DG_Basis1D as DG
import matplotlib.pyplot as plt
import math
plt.close('all')
#Set hyper parameters for the system
elements=12
deg=1
plot_points=101
#==============================================================================
#Set the system parameters here
#Set Boundary Conditions:
#Mass Flow Rate kg/s
m_flow=1
#Inlet Pressure
P0=101000
#Enthalpy inlet condition
h_bc=330
#------------------------------------------------------------------------------
#Set pipe parameters
#Start point of the pipe
start=0
#Ending point of the pipe
end=.5
#Define the mesh for the pipe
mesh=DG.Linspace(start,end,elements+1)
#Define Area
A=.5
#Define Diameter
D=2*math.sqrt(A/math.pi)
#Heat Transfer Coefficient
U=5
#Roughness
rough=.045
#Abmient Temperature
T_ambient=290
#------------------------------------------------------------------------------
#Set gas flow parameters
#Molecular weight kg/mol
MW=.01801528
#Ideal gas constant
R=8.314
#Heat Capacity
Cp=1.0
#viscosity
mu=1.8*10**-5
#==============================================================================
#Define offsets for plotting purposes
off=(deg+1)*elements
u_offset=0*off
rho_offset=1*off
p_offset=2*off
T_offset=3*off
h_offset=4*off
f_offset=5*off
#==============================================================================
#Solve the Steady state CNS equations 
Sol=DG.Steady_State(elements,deg,A,D,mesh,m_flow,MW,R,P0,h_bc,Cp,U,rough,mu,T_ambient)
#==============================================================================
#Plot the basis functions in reference space
xspan=DG.Linspace(-1,1,plot_points)
Y=np.matrix(np.zeros((plot_points,deg+1),dtype=float))
Coef=DG.Basis_Func(deg)

for j in range(0,deg+1):
    for i in range(0,plot_points):
        Y[i,j]=DG.Poly(xspan[i],Coef[:,j])
        
for i in range(0,deg+1):   
    plt.figure(0)
    plt.plot(xspan,Y[:,i])
    plt.title("Basis Functions in Reference Space")
    plt.xlabel("Reference Space")
    plt.ylabel("Function Value")
#==============================================================================
dg_mesh=DG.DG_Mesh(mesh,deg)
#==============================================================================
#Plot Velocity
plt.figure(1)
plt.plot(dg_mesh,Sol[u_offset:u_offset+elements*(deg+1)])
plt.title("Velocity Profile; Simplified View")
plt.xlabel("x-coordinate")
plt.ylabel("Velocity in m/s")
#==============================================================================
#Plot Density
plt.figure(2)
plt.plot(dg_mesh,Sol[rho_offset:rho_offset+elements*(deg+1)])
plt.title("Density Profile; Simplified View")
plt.xlabel("x-coordinate")
plt.ylabel("Density in kg/m^3")
#==============================================================================
#Plot Pressure
plt.figure(3)
plt.plot(dg_mesh,Sol[p_offset:p_offset+elements*(deg+1)])
plt.title("Pressure Profile; Simplified View")
plt.xlabel("x-coordinate")
plt.ylabel("Pressure in Pa")
#==============================================================================
#Plot Temperature
plt.figure(4)
plt.plot(dg_mesh,Sol[T_offset:T_offset+elements*(deg+1)])
plt.title("Temperature Profile; Simplified View")
plt.xlabel("x-coordinate")
plt.ylabel("Pressure in K")
#==============================================================================
#Plot Enthalpy
plt.figure(5)
plt.plot(dg_mesh,Sol[h_offset:h_offset+elements*(deg+1)])
plt.title("Specific Enthalpy Profile; Simplified View")
plt.xlabel("x-coordinate")
plt.ylabel("Specific Enthalpy in kJ/kg")
#==============================================================================
#plot Friction
plt.figure(6)
plt.plot(dg_mesh,Sol[f_offset:f_offset+elements*(deg+1)])
plt.title("Friction Factor Profile; Simplified View")
plt.xlabel("x-coordinate")
plt.ylabel("Friction Factor")
#==============================================================================
#Plot Velocity
plt.figure(7)
DG.DG_Plot(mesh,Sol[u_offset:u_offset+elements*(deg+1)])
plt.title("Velocity Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Velocity in m/s")
#==============================================================================
#Plot Density
plt.figure(8)
DG.DG_Plot(mesh,Sol[rho_offset:rho_offset+elements*(deg+1)])
plt.title("Density Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Density in kg/m^3")
#==============================================================================
#Plot Pressure
plt.figure(9)
DG.DG_Plot(mesh,Sol[p_offset:p_offset+elements*(deg+1)])
plt.title("Pressure Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Pressure in Pa")
#==============================================================================
#Plot Temperature
plt.figure(10)
DG.DG_Plot(mesh,Sol[T_offset:T_offset+elements*(deg+1)])
plt.title("Temperature Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Temperature in K")
#==============================================================================
plt.figure(11)
DG.DG_Plot(mesh,Sol[h_offset:h_offset+elements*(deg+1)])
plt.title("Specific Enthalpy Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Specific Enthalpy in kJ/kg")
#==============================================================================
plt.figure(12)
DG.DG_Plot(mesh,Sol[f_offset:f_offset+elements*(deg+1)])
plt.title("Friction Factor Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Friction Factor")