# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:36:23 2019

@author: Zach
"""

import numpy as np
import DG_Basis1D as DG
import matplotlib.pyplot as plt
plt.close('all')
#Set parameters for the solver
elements=10
deg=3
plot_points=101
a=0
b=1
#==============================================================================
#Generate uniform mesh over [a,b]
mesh=DG.Linspace(a,b,elements+1)
#==============================================================================
off=(deg+1)*elements
u_offset=0*off
rho_offset=1*off
p_offset=2*off
T_offset=3*off
h_offset=4*off
#==============================================================================
#Solve the Steady state CNS equations (no friction implementation) 
Sol=DG.Steady_State(elements,deg)
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
plt.title("Velocity Profile")
plt.xlabel("x-coordinate")
plt.ylabel("Velocity in m/s")
#==============================================================================
#Plot Density
plt.figure(2)
plt.plot(dg_mesh,Sol[rho_offset:rho_offset+elements*(deg+1)])
plt.title("Density Profile")
plt.xlabel("x-coordinate")
plt.ylabel("Density in kg/m^3")
#==============================================================================
#Plot Pressure
plt.figure(3)
plt.plot(dg_mesh,Sol[p_offset:p_offset+elements*(deg+1)])
plt.title("Pressure Profile")
plt.xlabel("x-coordinate")
plt.ylabel("Pressure in Pa")
#==============================================================================
#Plot Temperature
plt.figure(4)
plt.plot(dg_mesh,Sol[T_offset:T_offset+elements*(deg+1)])
plt.title("Temperature Profile")
plt.xlabel("x-coordinate")
plt.ylabel("Pressure in K")
#==============================================================================
#Plot Enthalpy
plt.figure(5)
plt.plot(dg_mesh,Sol[h_offset:h_offset+elements*(deg+1)])
plt.title("Specific Enthalpy Profile")
plt.xlabel("x-coordinate")
plt.ylabel("Specific Enthalpy in kJ/kg")
#==============================================================================
#Plot Velocity
plt.figure(6)
DG.DG_Plot(mesh,Sol[u_offset:u_offset+elements*(deg+1)])
plt.title("Velocity Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Velocity in m/s")
#==============================================================================
#Plot Density
plt.figure(7)
DG.DG_Plot(mesh,Sol[rho_offset:rho_offset+elements*(deg+1)])
plt.title("Density Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Density in kg/m^3")
#==============================================================================
#Plot Pressure
plt.figure(8)
DG.DG_Plot(mesh,Sol[p_offset:p_offset+elements*(deg+1)])
plt.title("Pressure Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Pressure in Pa")
#==============================================================================
#Plot Temperature
plt.figure(9)
DG.DG_Plot(mesh,Sol[T_offset:T_offset+elements*(deg+1)])
plt.title("Temperature Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Pressure in K")
#==============================================================================
plt.figure(10)
DG.DG_Plot(mesh,Sol[h_offset:h_offset+elements*(deg+1)])
plt.title("Specific Enthalpy Profile; DG Solution")
plt.xlabel("x-coordinate")
plt.ylabel("Specific Enthalpy in kJ/kg")