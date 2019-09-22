# 1D_CNS
The purpose of this program is to solve the 1D Steady State Compressible Navier Stokes(CNS)
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

To use this program open the main.py file. The user can edit all of the
necessary parameters for the solver through the main function. If the
user would like to change the intial guess for the fluid flow it can 
be changed in the DG_Basis1D.py file in the function Steady_State where
the state vector is initially defined. 
