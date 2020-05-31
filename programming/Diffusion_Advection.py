###################################################################
###################### NOTE ######################################
# This script doesn't work (needs to implement Neumann boundary condition on outflow)
# The transient loop breaks with the error "UFL conditions cannot be evaluated as bool in a Python context."

################################### NASA COVID-19 CHALLENGE ########################################


import time
import os
import math
from dolfin import *
import matplotlib.pyplot as plt

def ApplyBoundaryCondition(W, C, U_inflow, C_inflow):
    
    U = W.sub(0)
    P = W.sub(1)
    inflow   = 'near(x[0], 0)'
    outflow  = 'near(x[0], 10)'
    walls    = 'near(x[1], 0) || near(x[1], 10)'

    bcu_noslip  = DirichletBC(U, Constant((0, 0)), walls)

    bcp_inflow  = DirichletBC(P, Constant(U_inflow), inflow)
    bcp_outflow = DirichletBC(P, Constant(C_inflow), outflow)

    # Concentration

    bcc_inflow = DirichletBC(C, Constant(0.0), inflow)
    bcc_outflow = DirichletBC(C, Constant(1.0), outflow)

    bcu = [bcu_noslip]
    bcp = [bcp_inflow, bcp_outflow]
    bcc = [bcc_inflow, bcc_outflow]
    
    return bcu, bcp, bcc


def ConvectionDiffusionSolution(C,c0,dt,u1,D,rho,mu,alphaC,mesh,boundaries, bcc, dx, ds, n):

    ## Trial and Test function(s)
    c = TrialFunction(C)
    l = TestFunction(C)
    
    ## Result Functions
    c1 = Function(C)

    # Concentration Equation
          # Transient Term   #                 Advection Term                         # Diffusion Term                            
    F = rho*inner((c - c0)/dt,l)*dx() + alphaC*(rho*inner(u1,(grad(c ))*l) + D*dot(grad(c ), grad(l)))*dx() #
                                 # + (1-alphaC)*(rho*inner(u1,(grad(c0))*l) + D*dot(grad(c0), grad(l)))*dx() # Relaxation
    a, L = lhs(F), rhs(F)
    
    solve(a == L, c1, bcc)

    return c1


def MomentumSolution(W, w0, t,dt,rho,mu,alpha,mesh, boundaries, U_inflow, bcu, bcp, dx, ds, n):    
        
    # Defining unknown and test functions

    dw = TrialFunction(W)
    (v, q) = TestFunctions(W)
    w = Function(W)
    (u, p) = split(w)

    U = W.sub(0)
    P = W.sub(1)
    (u0, p0) = split(w0)

    L1 = assemble(Constant(1.0)*ds(1))
    L2 = assemble(Constant(1.0)*ds(2))
    L3 = assemble(Constant(1.0)*ds(3))
    L4 = assemble(Constant(1.0)*ds(4))
    # Equations
        # Linear Momentum Conservation
            
            # Transient Term            # Inertia Term             # Surface Forces Term           # Pressure Force
    a1 = inner((u-u0)/dt,v)*dx() + alpha*(inner(grad(u)*u , v) + (mu/rho)*inner(grad(u), grad(v)) - div(v)*p /rho)*dx() #+ \
    #                             (1-alpha)*(inner(grad(u0)*u0,v) + (mu/rho)*inner(grad(u0),grad(v)) - div(v)*p/rho)*dx()    # Relaxation
                        
    # Body Forces Term: Gravity         
    L1 = 0

     # Continuity Equation
    a2 = (q*div(u))*dx() 
    L2 = 0
 
    
    # Weak Complete Form
    F = a1 + a2 - (L1 + L2)
    #  Jacobian
    J = derivative(F,w,dw)
    
    problemU = NonlinearVariationalProblem(F,w,bcu,J)
    solverU = NonlinearVariationalSolver(problemU)

    prmU = solverU.parameters
    prmU['nonlinear_solver'] = 'newton'
    prmU['newton_solver']['absolute_tolerance'] = 1e-12
    prmU['newton_solver']['relative_tolerance'] = 1e-10
    prmU['newton_solver']['maximum_iterations'] = 500
    prmU['newton_solver']['linear_solver'] = 'mumps'

    (no_iterations,converged) = solverU.solve()
  
    return w,no_iterations,converged

# Parameters
tol = 1e-10 

C_int = 0.23
C_inflow = 0
U_inflow = 0.25

D = Constant(0.3)
rho = Constant(1.0)
mu = Constant(1.0)
t_end = 10 
dt = Constant(0.1)



# Room Dimensions
Dimx = 10
Dimy = 10

# Mesh resolution
Nx = 50
Ny = 50

alpha = Constant(1.0)
alphaC = Constant(1.0)

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(Dimx, Dimy), Nx, Ny, 'crossed')

# FEM Parameters
velocityElementfamily = 'Lagrange'
velocityElementOrder = 2
pressureElementfamily = 'Lagrange'
pressureElementOrder = 1
scalarFieldElementfamily = 'Lagrange'
scalarFieldElementOrder = 1

# Mesh Measures
elementShape = mesh.ufl_cell()
markers = MeshFunction('size_t', mesh, mesh.topology().dim()) # Domain
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() -1) 

dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

n = FacetNormal(mesh)

# Set Mesh Elements
Uel = VectorElement(velocityElementfamily, elementShape, velocityElementOrder) # Velocity vector field
Pel = FiniteElement(pressureElementfamily, elementShape, pressureElementOrder) # Pressure field
UPel = MixedElement([Uel,Pel]) # Mixed Finite Element 
Cel = FiniteElement(scalarFieldElementfamily, elementShape, scalarFieldElementOrder) # Scalar Field

# Mixed Function Space: Pressure and Velocity
W = FunctionSpace(mesh,UPel)
C = FunctionSpace(mesh,Cel)
w0 = Function(W)

# Initializing Concentration Field
init = Expression('C0','C0',C0 = C_int,degree=2)
c0 = Function(C)
c0.assign(project(init,C))

# Apply boundary Conditions
bcu, bcp, bcc = ApplyBoundaryCondition(W, C, U_inflow, C_inflow)


lastStep = False
results = []
concentration = []
t = 0

while t <= t_end:
    # Solving Momentum Equations
    begin('Flow - Time:{:.3f}s'.format(t))

    (w,no_iterations,converged) = MomentumSolution(W, w0, t,dt,rho,mu,alpha,mesh, boundaries, U_inflow, bcu, bcp, dx, ds, n)


    if converged:

        (u1, p1) = w.leaf_node().split()
        begin('Solving Diffusion-Convection Equation')

        c1 = ConvectionDiffusionSolution(C,c0,dt,u1,D,rho,mu, alphaC,mesh,boundaries, bcc, dx, ds, n)
        end()

        results.append(c1)
    
    w0.assign(w)
    c0.assign(c1)

    t += dt




x =3
