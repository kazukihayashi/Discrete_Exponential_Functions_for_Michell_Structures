import numpy as np
from numba import njit, f8, i4, b1
from numba.types import Tuple
import matplotlib.pyplot as plt

CACHE = True
PARALLEL = False

# @njit(f8[:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL)
def Length(node,member):
    L = np.array([np.sum((node[member[i,1]]-node[member[i,0]])**2)**0.5 for i in range(member.shape[0])],dtype=np.float64)
    return L

# @njit(f8[:,:](f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL)
def Grad_Length(node,member):
    '''
    (input)
    node[nn,3]<float> : Nodal coordinates
    member[nm,2]<int> : Connectivity

    (output)
    L_g[3*nn,nm]: Gradient of member lengths with respect to nodal coordinates
    '''
    nn = node.shape[0]
    nm = member.shape[0]
    L = Length(node,member)
    L_g = np.zeros((nn*3,nm),dtype=np.float64)
    for i in range(nm):
        L_g[3*member[i,0]:3*member[i,0]+3,i] = -(node[member[i,1]]-node[member[i,0]])/L[i]
        L_g[3*member[i,1]:3*member[i,1]+3,i] = (node[member[i,1]]-node[member[i,0]])/L[i]
    return L_g

# @njit(Tuple((f8[:,:,:],f8[:]))(f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL)
def Transformation_Matrix(node,member):

    '''
    (input)
    node[nn,3]<float> : Nodal locations (x,y,z coordinates) [mm]
    member[nm,2]<int> : Member connectivity

    (output)
    tt[nm,6,6]<float> : Transformation matrices (local to global)
    L[nm]<float> : Member lengths [mm]
    '''
    nn = len(node) # number of nodes
    nm = len(member) # number of members

    xyz = node[member[:,1]]-node[member[:,0]] # Direction vectors of members
    L = Length(node,member) # member lengths
  
    r3 = np.zeros((len(member),3,3)) # Initialize 3x3 rotation matrix
    r3[:,0,:] = xyz/np.expand_dims(L,-1) # local x
  
    tt = np.zeros((nm,6,6),dtype=np.float64)
    tt[:,0:3,0:3] = tt[:,3:6,3:6] = r3 # Copy r3 diagonally 
  
    return tt, L

# @njit(f8[:,:,:,:](f8[:,:],i4[:,:],f8[:,:]),cache=CACHE,parallel=PARALLEL)
def Grad_Transformation_Matrix(node,member,dL):
    '''
    (input)
    node[nn,3]<float> : Nodal locations (x,y,z coordinates)
    member[nm,2]<int> : Member connectivity
    dL[3*nn,nm]<float> : Sensitivity of member lengths

    (output)
    dtt[3*nn,nm,6,6]<float> : Sensitivity of transformation matrices (local to global)
    '''
    nn = len(node) # number of nodes
    nm = len(member) # number of members

    xyz = node[member[:,1]]-node[member[:,0]] # direction vectors of members
    L = Length(node,member) # member lengths
    Lxy = np.array([np.linalg.norm(node[member[i,1],:2]-node[member[i,0],:2]) for i in range(nm)]) # member lengths in x-y plane view

    dr3 = np.zeros((3*nn,nm,3,3)) # sensitivity of rotation matrices (local to global)

    flag = [-1,1] # -1 if start point, 1 if end point

    for i in range(2): # start and end points
        for j in range(nm): # number of members

            ## Sensitivity with respect to x coordinate of member ends
            dr3[3*member[j,i]+0,j,0,0] = -xyz[j,0]*dL[3*member[j,i]+0,j]/L[j]**2+flag[i]/L[j]
            dr3[3*member[j,i]+0,j,0,1] = -xyz[j,1]*dL[3*member[j,i]+0,j]/L[j]**2
            dr3[3*member[j,i]+0,j,0,2] = -xyz[j,2]*dL[3*member[j,i]+0,j]/L[j]**2

            ## Sensitivity with respect to y coordinate of member ends
            dr3[3*member[j,i]+1,j,0,0] = -xyz[j,0]*dL[3*member[j,i]+1,j]/L[j]**2
            dr3[3*member[j,i]+1,j,0,1] = -xyz[j,1]*dL[3*member[j,i]+1,j]/L[j]**2+flag[i]/L[j]
            dr3[3*member[j,i]+1,j,0,2] = -xyz[j,2]*dL[3*member[j,i]+1,j]/L[j]**2
        
            ## Sensitivity with respect to z coordinate of the start point of all members
            dr3[3*member[j,i]+2,j,0,0] = -xyz[j,0]*dL[3*member[j,i]+2,j]/L[j]**2
            dr3[3*member[j,i]+2,j,0,1] = -xyz[j,1]*dL[3*member[j,i]+2,j]/L[j]**2
            dr3[3*member[j,i]+2,j,0,2] = -xyz[j,2]*dL[3*member[j,i]+2,j]/L[j]**2+flag[i]/L[j]

    dtt = np.zeros((3*nn,nm,6,6),dtype=np.float64)
    dtt[:,:,0:3,0:3] = dtt[:,:,3:6,3:6] = dr3 # Copy dr3 diagonally 

    return dtt

# @njit(f8[:,:,:](f8[:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def Local_Element_Stiffness_Matrix(E,A,L):

    '''
    (input)
    E[nm]<float>: Young's modulus
    A[nm]<float>: Cross-sectional area
    L[nm]<float>:  Member length
  
    (output)
    kel[nm,6,6]<float>: Local element stiffness matrices
    '''
    kel_i = lambda a,l,e:(e*a/l)*np.array(
        [
            [1,0,0,-1,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [-1,0,0,1,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ])
 
    kel = np.zeros((len(A),6,6),dtype=np.float64)
    for i in range(len(A)):
        kel[i] = kel_i(A[i],L[i],E[i]) 

    return kel

# @njit(f8[:,:,:,:](f8[:],f8[:],f8[:],f8[:,:]),cache=CACHE,parallel=PARALLEL)
def Grad_Local_Element_Stiffness_Matrix(E,A,L,dL):

    '''
    (input)
    E[nm]<float>: Young's modulus
    A[nm]<float>: Cross-sectional area
    L[nm]<float>:  Member length
    dL[3*nn,nm]<float>: Sensitivity of member length
  
    (output)
    dkel[nn*3,nm,6,6]<float>: Sensitivity of local element stiffness matrices
    '''
  
    dkel_i = lambda a,l,e,dl: (e*a/l**2)*np.outer(dl,
        [
            [-1,0,0,1,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [1,0,0,-1,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ]).reshape((dl.shape[0],6,6))

    dkel = np.zeros((*dL.shape,6,6),dtype=np.float64)
    for i in range(len(A)):
        dkel[:,i,...] = dkel_i(A[i],L[i],E[i],dL[:,i])

    return dkel

# @njit(Tuple((f8[:,:],f8[:,:],f8[:,:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:],f8[:],f8[:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def Linear_Stiffness_Matrix(node,member,support,A,E,L,tt):
    '''
    (input)
    node[nn,3]: Nodal coordinates
    member[nm,2]: Member connectivity
    support[nn,3]: True if supported, else False
    A[nm]: Cross-sectional area.
    E[nm]: Young's modulus.
    L[nm]: Member lengths.
    tt[nm,6,6]: Transformation matrices

    (output)
    Kl_free[nn,nn]: Global linear stiffness matrix with respect to DOFs only
    Kl[nn,nn]: Global linear stiffness matrix
    kl_el_local[nm,6,6]: Element stiffness matrices (local coordinate system)
    '''

    ## Organize input model
    nn = node.shape[0] # number of nodes
    nm = member.shape[0] # number of members
    free = np.logical_not(support.flatten()) # DOFs are True, otherwise False

    ## Linear element stiffness matrices (local coordinate system)
    kl_el_local = Local_Element_Stiffness_Matrix(E,A,L)

    ## Make the tensor contiguous so that numba can compute quickly
    tt = np.ascontiguousarray(tt)
    kl_el_local = np.ascontiguousarray(kl_el_local)

    ## Transform element stiffness matrices into global coordinate system
    kl_el_global = np.zeros_like(kl_el_local)
    for i in range(nm):
        kl_el_global[i] = tt[i].T@kl_el_local[i]@tt[i]

    ## Assembling element matrices to the global matrix
    Kl = np.zeros((3*nn,3*nn),np.float64)
    for i in range(nm): # assemble element matrices into one matrix
        Kl[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kl_el_global[i,0:3,0:3]
        Kl[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kl_el_global[i,0:3,3:6]
        Kl[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kl_el_global[i,3:6,0:3]
        Kl[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kl_el_global[i,3:6,3:6]

    Kl_free = Kl[free][:,free] # Extract DOFs	

    return Kl_free, Kl, kl_el_local

# @njit(Tuple((f8[:,:,:],f8[:,:,:]))(i4[:,:],b1[:,:],f8[:,:,:],f8[:,:,:,:],f8[:,:,:],f8[:,:,:,:]),cache=CACHE,parallel=PARALLEL)
def Grad_Linear_Stiffness_Matrix(member,fix,tt,dtt,kl_el_local,dkl_el_local):
    '''
    (input)
    nn<int>: Number of nodes
    fix[nn,3]: True if supported, else False
    tt[nm,6,6]: Transformation matrices
    dtt[3*nn,nm,6,6]: Sensitivity of transformation matrices
    kl_el_local[nm,6,6]: Element stiffness matrices (local coordinate system)
    dkl_el_local[3*nn,nm,6,6]: Sensitivity of element stiffness matrices (local coordinate system)

    (output)
    dKl_free[3*nn,ndof,ndof]: Sensitivity of global linear stiffness matrix with respect to DOFs only
    dKl[3*nn,nn*3,nn*3]: Sensitivity of global linear stiffness matrix
    '''

    nn = fix.shape[0] # number of nodes
    nm = tt.shape[0] # number of members
    free = ~fix.flatten() # DOFs are True, otherwise False

    ## Make the tensor contiguous so that numba can compute quickly
    tt = np.ascontiguousarray(tt)
    dtt = np.ascontiguousarray(dtt)
    kl_el_local = np.ascontiguousarray(kl_el_local)
    dkl_el_local = np.ascontiguousarray(dkl_el_local)

    ## Transform the sensitivity of linear element stiffness matrices into the global coordinate system
    dkl_el_global = np.zeros_like(dkl_el_local)
    for i in range(3*nn):
        for j in range(nm):
            dkl_el_global[i,j] = dtt[i,j].T@kl_el_local[j]@tt[j] + tt[j].T@dkl_el_local[i,j]@tt[j] + tt[j].T@kl_el_local[j]@dtt[i,j]

    ## Assembling element matrices to the global matrix
    dKl = np.zeros((dtt.shape[0],3*nn,3*nn))
    for k in range(dtt.shape[0]):
        for i in range(nm):
            dKl[k,3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += dkl_el_global[k,i,0:3,0:3]
            dKl[k,3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += dkl_el_global[k,i,0:3,3:6]
            dKl[k,3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += dkl_el_global[k,i,3:6,0:3]
            dKl[k,3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += dkl_el_global[k,i,3:6,3:6]
    
    dKl_free = dKl[:,free][:,:,free] # Extract DOFs	

    return dKl_free, dKl

# @njit(Tuple((f8[:,:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:,:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def Linear_Stiffness_Matrix_using_Element_Stiffness_Matrix(node,member,fix,kl_el_local,tt):
    '''
    (input)
    node[nn,3]: Nodal coordinates
    member[nm,2]: Member connectivity
    fix[nn,3]: True if supported, else False
    kl_el_local[nm,6,6]: Element stiffness matrices (local coordinate system)
    tt[nm,6,6]: Transformation matrices

    (output)
    Kl_free[nfree,nfree]: Global linear stiffness matrix with respect to DOFs only
    Kl[nn*3,nn*3]: Global linear stiffness matrix
    '''

    ## Organize input model
    nn = node.shape[0] # number of nodes
    nm = member.shape[0] # number of members
    free = ~fix.flatten() # DOFs are True, otherwise False

    ## Make the tensor contiguous so that numba can compute quickly
    tt = np.ascontiguousarray(tt)
    kl_el_local = np.ascontiguousarray(kl_el_local)

    ## Transform element stiffness matrices into global coordinate system
    kl_el_global = np.zeros_like(kl_el_local)
    for i in range(nm):
        kl_el_global[i] = tt[i].T@kl_el_local[i]@tt[i]

    ## Assembling element matrices to the global matrix
    Kl = np.zeros((3*nn,3*nn),np.float64)
    for i in range(nm): # assemble element matrices into one matrix
        Kl[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kl_el_global[i,0:3,0:3]
        Kl[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kl_el_global[i,0:3,3:6]
        Kl[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kl_el_global[i,3:6,0:3]
        Kl[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kl_el_global[i,3:6,3:6]

    Kl_free = Kl[free][:,free] # Extract DOFs	

    return Kl_free, Kl

def Grad_Displacement(Kl_free,dKl_free,u_free):

    '''
    (input)
    Kl_free[ndof,ndof]: Global linear stiffness matrix with respect to DOFs only
    dKl_free[3*nn,ndof,ndof]: Sensitivity of global linear stiffness matrix with respect to DOFs only
    u_free[ndof]: Displacement vector with respect to DOFs only

    (output)
    du_free[3*nn,ndof]: Sensitivity of displacement vector with respect to DOFs only
    '''

    du_free = np.linalg.solve(Kl_free,-(dKl_free@u_free).T).T

    return du_free

# @njit(f8[:,:](f8[:,:,:],f8[:,:,:,:],f8[:,:,:],f8[:,:,:,:],f8[:,:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def Grad_Section_Force(T,dT,kel,dkel,dm,ddm):

    '''
    (input)
    T[nm,6,6]<float>: Transformation matrices
    dT[nn*3,nm,6,6]<float>: Sensitivity of transformation matrices
    kel[nm,6,6]<float>: Local element stiffness matrices
    dkel[nn*3,nm,6,6]<float>: Sensitivity of local element stiffness matrices
    dm[nm,6]<float>: Displacements of member ends in the global coordinate system
    ddm[nn*3,nm,6]<float>: Sensitivity of displacements of member ends in the global coordinate system
  
    (output)
    ds[nn*3,nm]<float>: Sensitivity of section forces
    '''

    # ds = np.squeeze(dkel@T@np.expand_dims(dm,-1) + kel@dT@np.expand_dims(dm,-1) + kel@T@np.expand_dims(ddm,-1),3)

    '''
    When enabling numba, please use the following code instead. 
    '''
    T = np.ascontiguousarray(T)
    dT = np.ascontiguousarray(dT)
    kel = np.ascontiguousarray(kel)
    dkel = np.ascontiguousarray(dkel)
    dm = np.ascontiguousarray(dm)
    ddm = np.ascontiguousarray(ddm)

    ds = np.zeros_like(ddm)
    for i in range(ddm.shape[0]):
        for j in range(ddm.shape[1]):
            ds[i,j] = dkel[i,j]@T[j]@dm[j] + kel[j]@dT[i,j]@dm[j] + kel[j]@T[j]@ddm[i,j]

    return ds[:,:,3] # 0: axial (positive in compression), 3: axial (positive in tension), 1,2,5,6: shear (0 for truss elements)

# @njit(Tuple((f8[:,:],f8[:,:],f8[:,:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:],f8[:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def Geometry_Stiffness_Matrix(node,member,support,N,L,tt):
    '''
    (input)
    node[nn,3]: Nodal coordinates
    member[nm,2]: Member connectivity
    support[nn,3]: True if supported, else False
    N[nm]: Axial forces (positive for tension, negative for compression).
    L[nm]: Member lengths.
    tt[nm,6,6]: transformation matrices.

    (output)
    Kg_free[ndof,ndof]: Global geometry stiffness matrix with respect to DOFs only
    Kg[nn*3,nn*3]: Global geometry stiffness matrix
    kg_el_local[nm,12,12]<float> : Element geometry stiffness matrices (local coordinate system)
    '''
    ## Organize input model
    nn = node.shape[0] # number of nodes
    nm = member.shape[0] # number of members
    free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
    tt = np.ascontiguousarray(tt)

    keg_i = lambda n,l:np.array(
        [
            [n/l,0,0,-n/l,0,0],
            [0,6*n/(5*l),0,0,-6*n/(5*l),0],
            [0,0,6*n/(5*l),0,0,-6*n/(5*l)],
            [-n/l,0,0,n/l,0,0],
            [0,-6*n/(5*l),0,0,6*n/(5*l),0],
            [0,0,-6*n/(5*l),0,0,6*n/(5*l)],
        ])

    ## Geometry element stiffness matrices
    kg_el_local = np.zeros((nm,6,6),dtype=np.float64)
    for i in range(nm):
        kg_el_local[i] = keg_i(N[i],L[i])

    ## Transformation from local to global coordinate system
    kg_el_global = np.zeros((nm,6,6),dtype=np.float64)
    for i in range(nm):
        kg_el_global[i] = tt[i].T@kg_el_local[i]@tt[i]

    ## Assembling element matrices to the global matrix
    Kg = np.zeros((3*nn,3*nn),np.float64) # geometry stiffness matrix
    for i in range(nm): # assemble element matrices into one matrix
        Kg[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kg_el_global[i,0:3,0:3]
        Kg[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kg_el_global[i,0:3,3:6]
        Kg[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kg_el_global[i,3:6,0:3]
        Kg[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kg_el_global[i,3:6,3:6]

    Kg_free = Kg[free][:,free]

    return Kg_free, Kg, kg_el_local

# @njit(Tuple((f8[:],f8[:,:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def Stiffness_Matrix_Eig(node0,member,support,A,E):
    '''
    (input)
    node[nn,3]: nodal coordinates
    member[nm,2]: member connectivity
    support[nn,3]: True if supported, else False
      (note) Isolated nodes can be ignored by setting the "support" values associated with them to True.
    A[nm]: Cross-sectional area.
      (note) Assign exactly 0 to vanishing members so as to correctly compute the rank of the stifness matrix

    (output)
    eig_vals[nDOF]: eigen-values
    eig_modes[nDOF,nn,3]: eigen-modes
    '''

    ## Organize input model
    nn = node0.shape[0] # number of nodes
    free = np.logical_not(support.flatten()) # DOFs are True, otherwise False

    ## Transformation matrices (tt) and initial lengths (ll0)
    tt,ll0 = Transformation_Matrix(node0,member)
    tt = np.ascontiguousarray(tt)

    ## Linear stiffness matrix 
    Kl_free, _, _ = Linear_Stiffness_Matrix(node0,member,support,A,E,ll0,tt)
    eig_vals,eig_vecs = np.linalg.eigh(Kl_free)

    ## Reshape eig_vecs to obtain eigen-modes
    u = np.zeros((len(eig_vals),nn*3),dtype=np.float64)
    for i in range(len(eig_vals)):
        uu = u[i] # This is a shallow copy, and u also changes in the next line
        uu[free] = eig_vecs[:,i]
    eig_modes = u.reshape((len(eig_vals),nn,3))

    return eig_vals,eig_modes

# @njit(Tuple((f8[:,:],f8[:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def Linear_Structural_Analysis(node0,member,support,load,A,E):

    '''
    (input)
    node[nn,3]: Nodal coordinates
    member[nm,2]: Member connectivity
    support[nn,3]: True if supported, else False
    load[nn,3]: Load magnitude. 0 if no load is applied.
    A[nm]: Cross-sectional area.
    E[nm]: Young's modulus.

    (output)
    deformation[nn,3]: nodal deformations
    force[nm]: axial forces
    reaction[nn,3]: reaction forces
    (note): Only supported coordinates can take a non-zero value. The other coordinates (i.e., DOFs) takes 0.
    '''

    ## Organize input model
    nn = node0.shape[0] # number of nodes
    nm = member.shape[0] # number of members
    free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
    p = load.flatten() # load vector
    p_free = p[free] # load vector with respect to DoFs

    ## Transformation matrices (tt) and initial lengths (ll0)
    tt,ll0 = Transformation_Matrix(node0,member)
    tt = np.ascontiguousarray(tt)

    ## Linear stiffness matrix 
    Kl_free, Kl, kl_el_local = Linear_Stiffness_Matrix(node0,member,support,A,E,ll0,tt)

    ## Solve the stiffness equation (Kl_free)(Up) = (pp) to obtain the deformation
    u_free = np.linalg.solve(Kl_free,p_free) # Compute displacement Up (size:nDOF), error occurs at this point when numba is not in use
    # u_free = sp_solve(Kl_free,pp,assume_a ='sym',check_finite=False) # Use this for better precision when numba is not in use

    ## Deformation
    d = np.zeros(nn*3,dtype=np.float64)
    d[free] = u_free
    deformation = d.reshape((nn,3))

    ## Section forces
    dm = np.hstack((deformation[member[:,0]],deformation[member[:,1]]))
    force = np.zeros((nm,6),np.float64)
    for i in range(nm):
        force[i] = kl_el_local[i]@tt[i]@dm[i]
    force = force[:,3] # 0: axial (positive in compression), 3: axial (positive in tension), 1,2,5,6: shear (0 for truss elements)
    
    ## Reaction forces
    rfix = np.dot(Kl[~free][:,free],u_free)
    r = np.zeros(nn*3,dtype=np.float64)
    r[~free] = rfix - p[~free]
    reaction = r.reshape((nn,3))

    return deformation, force, reaction

# @njit(Tuple((f8[:,:],f8[:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def Linear_Structural_Analysis_using_Element_Stiffness_Matrix(node0,member,fix,load,kl_el_local):

    '''
    (input)
    node[nn,3]: Nodal coordinates
    member[nm,2]: Member connectivity
    fix[nn,6]: True if supported, else False
    load[nn,6]: Load magnitude. 0 if no load is applied
    kl_el_local[nm,12,12]: Element stiffness matrices (local coordinate system)

    (output)
    deformation[nn,6]: nodal deformations
    force[nm,12]: section forces (0-6: start point, 7-12: end point, 0:axial, 1-2:shear, 3:torsion, 4-5:bending)
    reaction[nn,6]: reaction forces
    '''

    ## Organize input model
    nn = node0.shape[0] # number of nodes
    nm = member.shape[0] # number of members
    free = ~fix.flatten() # DOFs are True, otherwise False
    p = load.flatten() # load vector
    p_free = p[free] # load vector with respect to DoFs

    ## Transformation matrices
    tt,_ = Transformation_Matrix(node0,member)

    ## Make the tensor contiguous so that numba can compute quickly
    tt = np.ascontiguousarray(tt)
    kl_el_local = np.ascontiguousarray(kl_el_local)

    ## Linear stiffness matrix 
    Kl_free, Kl = Linear_Stiffness_Matrix_using_Element_Stiffness_Matrix(node0,member,fix,kl_el_local,tt)

    ## Solve the stiffness equation (Kl_free)(u_free) = (p_free) to obtain the deformation
    u_free = np.linalg.solve(Kl_free,p_free) # Compute displacement u_free (size:nDOF)
    
    ## Deformation
    d = np.zeros(nn*3,dtype=np.float64)
    d[free] = u_free
    deformation = d.reshape((nn,3))

    ## Section forces
    dm = np.hstack((deformation[member[:,0]],deformation[member[:,1]]))
    force = np.zeros((nm,6),np.float64)
    for i in range(nm):
        force[i] = kl_el_local[i]@tt[i]@dm[i]
    force = force[:,3] # 0: axial (positive in compression), 3: axial (positive in tension), 1,2,5,6: shear (0 for truss elements)
    
    ## Reaction forces
    rfix = np.dot(Kl[~free][:,free],u_free)
    r = np.zeros(nn*3,dtype=np.float64)
    r[~free] = rfix - p[~free]
    reaction = r.reshape((nn,3))

    return deformation, force, reaction

# @njit(f8(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def Strain_Energy(node0,member,fix,load,A,E):

    '''
    (input)
    node[nn,3]: Nodal coordinates
    member[nm,2]: Member connectivity
    fix[nn,3]: True if supported, else False
    load[nn,3]: Load magnitude. 0 if no load is applied.
    A[nm]: Cross-sectional area.
    E[nm]: Young's modulus.

    (output)
    strain_energy<float>: Total axial strain energy
    '''

    _, N, _ = Linear_Structural_Analysis(node0,member,fix,load,A,E)
    L = Length(node0,member)
    strain_energy = np.sum(L*N**2/(2*E*A)) # axial strain energy

    return strain_energy

# @njit(f8[:](f8[:],f8[:,:],f8[:],f8[:],f8[:],f8[:,:]),cache=CACHE,parallel=PARALLEL)
def Grad_Strain_Energy(N,dN,A,E,L,dL):

    '''
    Compute the sensitivity of axial strain energy and bending
    strain energy with respect to the nodal coordinates.

    (input)
    N[nm]<float> : Section forces
    dN[3*nn,nm]<float> : Sensitivity of section forces
    A[nm]: Cross-sectional area
    E[nm]: Young's modulus
    L[nm]: Member lengths
    dL[3*nn,nm]<float> : Sensitivity of member lengths

    (output)
    d_strain_energy[3*nn]<float>: Sensitivity of total axial strain energy
    '''

    d_strain_energy = np.zeros(dN.shape[0])
    for i in range(len(A)):
        d_strain_energy += (2*L[i]*dN[:,i]+N[i]*dL[:,i])*N[i]/(2*E[i]*A[i])

    return d_strain_energy

# @njit(Tuple((f8[:],f8[:,:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def LinearBucklingAnalysis(node0,member,fix,load,A,E):
    '''
    (input)
    node[nn,3]: Nodal coordinates
    member[nm,2]: Member connectivity
    fix[nn,3]: True if supported, else False
    load[nn,3]: Load magnitude. 0 if no load is applied.
    A[nm]: Cross-sectional area
    E[nm]: Young's modulus

    (output)
    eig_vals[nDOF]: eigen-values (load factors that cause the buckling)
    eig_modes[nDOF,nn,3]: eigen-modes (buckling modes)
    '''

    ## Organize input model
    nn = node0.shape[0] # number of nodes
    free = np.logical_not(fix.flatten()) # DOFs are True, otherwise False

    ## Transformation matrices (tt) and initial lengths (ll0)
    tt,ll0 = Transformation_Matrix(node0,member)
    tt = np.ascontiguousarray(tt)

    ## Linear stiffness matrix 
    Kl_free, _, kl_el_local = Linear_Stiffness_Matrix(node0,member,fix,A,E,ll0,tt)

    ## Structural analysis to obtain axial forces
    _, N, _ = Linear_Structural_Analysis_using_Element_Stiffness_Matrix(node0,member,fix,load,kl_el_local)

    ## Geometric stiffness matrix using the axial forces
    Kg_free, _ = Geometry_Stiffness_Matrix(node0,member,fix,N,ll0,tt)

    ## Solve the eigenvalue problem
    eig_vals_comp, eig_vecs_comp = np.linalg.eig(np.dot(np.ascontiguousarray(-np.linalg.inv(Kg_free)),np.ascontiguousarray(Kl_free)))
    # eig_vals_comp, eig_vecs_comp = sp_eig(-Kl_free,Kg_free) # Use this for better precision, but numba cannot be used
    eig_vals = eig_vals_comp.real.astype(np.float64) # Extract real numbers
    eig_vecs = eig_vecs_comp.real.astype(np.float64) # Extract real numbers
    eig_modes = np.empty((len(eig_vals),nn*3),dtype=np.float64)
    for i in range(len(eig_vals)):
        eig_modes[i,free] = eig_vecs[:,i]
    eig_modes = eig_modes.reshape((nn,3))

    return eig_vals, eig_modes

# @njit(Tuple((f8[:,:],f8[:,:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],b1),cache=CACHE,parallel=PARALLEL)
def Self_Equilibrium_Matrix(node,member,fix,display):
    '''
    (input)
    node[nn,3]<float> : nodal coordinates
    member[nm,2]<int> : connectivity
    fix[nn,3]<bool> : True if fixed, False otherwise

    (output)
    aa[nfree,nm]: Equilibrium matrix with respect to unconstrained dofs
    aa_full[3*nn,nm]: Equilibrium matrix with respect to all dofs
    self_force[Deg. of statical indeterminacy,nm]: Bases of self-equilibrium forces
    '''
    ## Number of nodes and members
    nn = node.shape[0]
    nm = member.shape[0]

    ## Construct equilibrium matrix
    aa_full = np.zeros((3*nn,nm),dtype=np.float64)
    for i in range(nm):
        dd = node[member[i,1]] - node[member[i,0]]
        dd /= np.linalg.norm(dd)
        aa_full[3*member[i,0]:3*member[i,0]+3,i] = -dd
        aa_full[3*member[i,1]:3*member[i,1]+3,i] = dd

    ## Retrieve row vectors with respect to DoFs
    aa = aa_full[~fix.flatten()]

    ## Singular value decomposition of the equilibrium matrix
    zz, ss, hht = np.linalg.svd(aa, full_matrices=True)
    hh = hht.T
    rank = np.sum(ss>1.e-10)

    ## Bases of loads that can be resolved and corrsponding internal forces without self-equilibrium forces
    load = zz[:,0:rank]
    eqforce = hh[:,0:rank]/ss[0:rank]

    ## Bases of self-equilibrium forces
    selfforce = hh[:,rank::]

    ## Bases of first-order infinitesimal mechanism
    flex = zz[:,rank::]

    ## Output load including reaction force corresponding to fixed nodal displacements
    load_full = np.zeros((load.shape[1],nn, node.shape[1]), dtype=np.float64)
    for k in range(load.shape[1]):
        loadk = aa_full@np.ascontiguousarray(eqforce[:,k])
        for i in range(nn):
            load_full[k,i,:] = loadk[3*i:3*i+3]

    ## Output first-order infinitesimal mechanism modes including fixed nodal displacements
    flex_full = np.zeros((flex.shape[1],nn, node.shape[1]), dtype=np.float64)
    for k in range(flex.shape[1]):
        l = 0
        for i in range(nn):
            for j in range(3):
                if ~fix[i,j]:
                    flex_full[k,i,j] = flex[l,k]
                    l += 1

    # ## Display
    # if display:
    #     print(f"Number of nodes:                  {nn:d}")
    #     print(f"Number of members:                {nm:d}")
    #     print(f"Number of boundary conditions:    {fix.sum():d}")
    #     print(f"Size of equilibrium matrix:       {aa.shape[0]:d} x {aa.shape[1]:d}")
    #     print(f"Degree of statical indeterminacy: {aa.shape[1]-rank:d}")
    #     print(f"Degree of kinematic indeterminacy:{aa.shape[0]-rank:d}")
    #     print("Bases of load that can be resolved:")
    #     print(f"{load}\n")
    #     print("Equilibrium internal force without self-equilibrium force:")
    #     print(f"{eqforce}\n")
    #     print("Bases of self-equilibrium force:")
    #     print(f"{selfforce}\n")
    #     print("Bases of first-order infinitesimal mechanism:")
    #     print(f"{flex}\n")
    #     print("Load full:")
    #     for k in range(load_full.shape[0]):
    #         print(f"mode {k+1:d}")
    #         print(load_full[k,:,:])
    #     print("Flex full:")
    #     for k in range(flex_full.shape[0]):
    #         print(f"mode {k+1:d}")
    #         print(flex_full[k,:,:])

    return aa, aa_full, selfforce

def Grad_Self_Equilibrium_Matrix(node,member,fix):
    '''
    (input)
    node[nn,3]<float> : nodal coordinates
    member[nm,2]<int> : connectivity
    fix[nn,3]<bool> : True if fixed, False otherwise

    (output)
    aa_g[3*nn,nfree,nm]: gradient of equilibrium matrix with respect to unconstrained dofs
    aa_full_g[3*nn,3*nn,nm]: gradient of equilibrium matrix with respect to all dofs
    '''
    nn = node.shape[0]
    nm = member.shape[0]

    aa_full_g = np.zeros((3*nn,3*nn,nm),dtype=np.float64)
    xyz = node[member[:,1]] - node[member[:,0]]
    xyz_g = np.zeros((3*nn,nm,3))
    for i in range(nm):
        for j in range(3):
            xyz_g[3*member[i,0]+j,i,j] = -1
            xyz_g[3*member[i,1]+j,i,j] = 1 
    l = np.array([np.linalg.norm(xyz[i]) for i in range(nm)]) # np.linalg.norm(xyz,axis=1)
    l_g = Grad_Length(node,member)
    dd_g = np.zeros((3*nn,nm,3))
    for i in range(3*nn):
        for j in range(nm):
            for k in range(3):
                dd_g[i,j,k] = (xyz_g[i,j,k]*l[j]-xyz[j,k]*l_g[i,j])/l[j]**2
    for k in range(3*nn):
        for i in range(nm):
            aa_full_g[k,3*member[i,0]:3*member[i,0]+3,i] = -dd_g[k,i,:]
            aa_full_g[k,3*member[i,1]:3*member[i,1]+3,i] = dd_g[k,i,:]
    aa_g = aa_full_g[:,~fix.flatten(),:]

    return aa_g, aa_full_g

# @njit(f8[:](f8[:,:],i4[:,:],b1[:,:],f8[:,:]),cache=CACHE,parallel=PARALLEL)
def Axial_Force_Determinate(node,member,fix,load):
    '''
    (input)
    node[nn,3]<float> : nodal coordinates
    member[nm,2]<int> : connectivity
    fix[nn,3]<bool> : True if fixed, False otherwise
    load[nn,3]<float>: nodal loads

    (output)
    internal_force[nm] : axial force of bar members
    '''
    aa, aa_full, selfforce = Self_Equilibrium_Matrix(node,member,fix,False)
    if aa.shape[0] == aa.shape[1]:
        p_free = load.flatten()[~fix.flatten()]
        internal_force = np.linalg.inv(aa)@p_free
    else:
        u,s,vh = np.linalg.svd(aa)
        rank = np.linalg.matrix_rank(aa)
        orthogonality = u.T[rank:]@load[~fix]
        if np.all(np.isclose(orthogonality,0.0)):
            raise Exception("The equilibrium matrix is not square. However, there is an equilibrium state for this load condition.")
        else:
            raise Exception("The equilibrium matrix is not square. Also, there is no equilibirum state for this load condition.")
    
    return internal_force

def Optimize_Strain_Energy(node0,member,fix,load,A,E,is_variable,x_init=None,lb=None,ub=None,tol=1e-5):
    '''
    (input)
    node0[nn,3]<float>: nodal coordinates
    member[nm,2]<int> : connectivity
    fix[nn,3]<bool>   : True if fixed, False otherwise
    load[nn,3]<float> : nodal loads
    A[nm]             : Cross-sectional area
    E[nm]             : Young's modulus
    is_variable[nn,3]<float>: True if the nodal coordinate is variable, False if not variable
    x_init[nvar]<float>: Initial variable (optional). If not provided, the initial nodal location is used.
    lb[nvar]<float>   : Lower bound that each variable can take
    ub[nvar]<float>   : Upper bound that each variable can take
    tol<float>        : Tolerance to evaluate the optimality criterion

    (output)
    node_opt[nn,3]<float> : Optimal nodal coordinates
    res : Information about optimization result
    '''
    from scipy.optimize import minimize, Bounds

    if x_init is None:
        x_init = node0[is_variable]

    def F(x):

        node = np.copy(node0)
        node[is_variable] = x
        se = Strain_Energy(node,member,fix,load,A,E)

        return se

    def dF(x):

        nn = len(node0)
        nm = len(member)

        node = np.copy(node0)
        node[is_variable] = x

        tt, L = Transformation_Matrix(node,member)
        Kl_free, _, kl_el_local = Linear_Stiffness_Matrix(node,member,fix,A,E,L,tt)
        deformation,N,_ = Linear_Structural_Analysis_using_Element_Stiffness_Matrix(node,member,fix,load,kl_el_local)

        free = ~fix.flatten() # DOFs are True, otherwise False
        u_free = deformation.flatten()[free]
        dm = deformation[member].reshape((nm,6))

        dL = Grad_Length(node,member)
        dtt = Grad_Transformation_Matrix(node,member,dL)
        dkl_el_local = Grad_Local_Element_Stiffness_Matrix(E,A,L,dL)
        dKl_free, _ = Grad_Linear_Stiffness_Matrix(member,fix,tt,dtt,kl_el_local,dkl_el_local)
        du_free = Grad_Displacement(Kl_free,dKl_free,u_free)

        d_deformation = np.zeros((nn*3,nn*3),dtype=np.float64)
        d_deformation[:,free] = du_free
        d_deformation = d_deformation.reshape((nn*3,nn,3))
        ddm = d_deformation[:,member].reshape((nn*3,nm,6))
        dN = Grad_Section_Force(tt,dtt,kl_el_local,dkl_el_local,dm,ddm)
        
        dse_a = Grad_Strain_Energy(N,dN,A,E,L,dL)

        return dse_a[is_variable.flatten()]

    if lb is None:
        lb = -np.inf
    if ub is None:
        ub = np.inf
    res = minimize(F,x_init,method='SLSQP',jac=dF,bounds=Bounds(lb,ub),tol=tol)

    node_opt = np.copy(node0)
    node_opt[is_variable] = res.x

    return node_opt, res

def Optimize_Force_Length(node0,member,fix,load,is_variable,x_init=None,lb=None,ub=None,tol=1e-5):
    '''
    (input)
    node0[nn,3]<float>: nodal coordinates
    member[nm,2]<int> : connectivity
    fix[nn,3]<bool>   : True if fixed, False otherwise
    load[nn,3]<float> : nodal loads
    is_variable[nn,3]<float>: True if the nodal coordinate is variable, False if not variable
    x_init[nvar]<float>: Initial variable (optional). If not provided, the initial nodal location is used.
    lb[nvar]<float>   : Lower bound that each variable can take
    ub[nvar]<float>   : Upper bound that each variable can take
    tol<float>        : Tolerance to evaluate the optimality criterion

    (output)
    node_opt[nn,3]<float> : Optimal nodal coordinates
    res : Information about optimization result
    '''
    from scipy.optimize import minimize, Bounds

    if x_init is None:
        x_init = node0[is_variable]

    def F(x):
        node = np.copy(node0)
        node[is_variable] = x
        internal_force = Axial_Force_Determinate(node,member,fix,load)
        length = np.linalg.norm(node[member[:,1]]-node[member[:,0]],axis=1)
        objfun = np.sum(np.abs(internal_force)*length)
        return objfun

    def dF(x):
        node = np.copy(node0)
        node[is_variable] = x
        internal_force = Axial_Force_Determinate(node,member,fix,load)
        aa, aa_full, self_force = Self_Equilibrium_Matrix(node,member,fix,False)
        aa_g, aa_full_g = Grad_Self_Equilibrium_Matrix(node,member,fix)
        length = np.linalg.norm(node[member[:,1]]-node[member[:,0]],axis=1)
        length_g = Grad_Length(node,member)
        internal_force_g = -np.linalg.inv(aa)@aa_g@internal_force
        objfun_g = np.sum(np.sign(internal_force)*internal_force_g*length + np.abs(internal_force)*length_g,axis=1)
        return objfun_g[is_variable.flatten()]

    if lb is None:
        lb = -np.inf
    if ub is None:
        ub = np.inf
    res = minimize(F,x_init,method='SLSQP',jac=dF,bounds=Bounds(lb,ub),tol=tol)

    node_opt = np.copy(node0)
    node_opt[is_variable] = res.x

    return node_opt, res

def Draw2D(node,member,fix,load=None,disp=None,disp_scale=1,line_width=2):

    nn = node.shape[0] # number of nodes
    nm = member.shape[0] # number of members

    ## Set line widths
    if type(line_width) in [float,int]:
        line_width = np.ones(nm)*line_width
    
    ## Draw initial members
    for i in range(member.shape[0]): # repeat for all the members
        # plt.annotate(f'<{i:d}>',(node[member[i,0],:]/2+node[member[i,1],:]/2)[0:2],color='blue') # annotation of members
        plt.plot(node[member[i,:],0],node[member[i,:],1],color='gray',linewidth=1.0,linestyle=':') # draw initial members
    
    ## Draw deformed members
    if disp is not None:
        for i in range(member.shape[0]): # repeat for all the members
            plt.plot(node[member[i,:],0]+disp[member[i,:],0]*disp_scale,node[member[i,:],1]+disp[member[i,:],1]*disp_scale,color='gray',linewidth=line_width[i]) # draw deformed members
    
    ## Draw load vectors
    if load is not None:
        for i in range(node.shape[0]): # repeat for all the nodes
            if not np.all(load[i]==0.0):
                plt.quiver(node[i,0]+disp[i,0]*disp_scale,node[i,1]+disp[i,1]*disp_scale,load[i,0],load[i,1],color='red')
    
    ## Draw deformed nodes
    for i in range(node.shape[0]): # repeat for all the nodes
        plt.annotate(f'({i:d})',node[i,0:2]+disp[i,0:2]*disp_scale+0.05) # annotation of nodes
        if np.all(fix[i,:]):
            plt.plot(node[i,0],node[i,1],color='red',marker='D',markersize=10) # supports
        else:
            plt.plot(node[i,0]+disp[i,0]*disp_scale,node[i,1]+disp[i,1]*disp_scale,color='black',marker='o',markersize=8) # draw a black circle
    
    ## figure setting
    plt.margins(0.2)
    plt.xlabel("$x$") # add x label
    plt.ylabel("$y$") # add y label
    plt.gca().set_aspect('equal', adjustable='box') # display x- and y-axes on the same scale
    plt.show() # show figure


'''
Example 1: Triangle-shaped 3-bar 2D truss
'''

# node0 = np.array([[0,0,0],[8,0,0],[4,4,0]],dtype=np.float64)
# member = np.array([[0,1],[0,2],[1,2]],dtype=np.int32)
# support = np.array([[1,1,1],[0,1,1],[0,0,1]],dtype=bool)
# load = np.array([[0,0,0],[0,0,0],[0,-1,0]],dtype=np.float64)
# A = np.array([1.0,1.0,1.0],dtype=np.float64)
# E = np.array([1.0,1.0,1.0],dtype=np.float64)

# d,s,r = Linear_Structural_Analysis(node0,member,support,load,A,E)

# import time
# t1 = time.perf_counter()
# for i in range(100):
# 	d,s,c = Linear_Structural_Analysis(node0,member,support,load,A,E)
# t2 = time.perf_counter()
# print("d={0}".format(d))
# print("s={0}".format(s))
# print("time={0}".format(t2-t1))

'''
Example 2a: Geometry optimization of a 3x2-grid 2D truss
'''

# node0 = np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0]],dtype=float) # 節点座標 [m]
# member = np.array([[0,1],[1,2],[3,4],[4,5],[1,4],[2,5],[0,4],[1,5]],dtype=int) # どの節点同士を部材でつなげるか
# support = np.zeros((node0.shape[0],3),dtype=bool)
# support[[0,3]] = True
# support[:,2] = True
# load = np.zeros((node0.shape[0],3),dtype=np.float64)
# load[2,1] = -1000
# A = np.ones(member.shape[0])*1e-4
# E = np.ones(member.shape[0])*2.05e11
# disp,N,r = Linear_Structural_Analysis(node0,member,support,load,A,E)
# print(f"The displacement(x) of the bottom-right node:{disp[2,0]*1000}[mm]") # correct value: 0.048780 mm
# print(f"The displacement(y) of the bottom-right node:{disp[2,1]*1000}[mm]") # correct value: -0.666188 mm

# Draw2D(node0,member,support,load,np.zeros_like(node0))

# is_variable = np.zeros_like(node0,dtype=bool)
# var_i = np.array([[1,0],[1,1],[4,0],[4,1],[5,0],[5,1]],dtype=int)
# is_variable[var_i[:,0],var_i[:,1]] = True
# node_opt,res = Optimize_Strain_Energy(node0,member,support,load,A,E,is_variable,lb=node0[is_variable]-0.5,ub=node0[is_variable]+0.5)
# print(f"Optimal strain energy: {res.fun}")

# Draw2D(node_opt,member,support,load,np.zeros_like(node0))

# node0[is_variable] = node0[is_variable] + 1e-4*np.random.rand(is_variable.sum())-5e-5
# node_opt,res = Optimize_Force_Length(node0,member,support,load,is_variable,lb=node0[is_variable]-0.5,ub=node0[is_variable]+0.5)
# print(f"Optimal force x length: {res.fun}")

# Draw2D(node_opt,member,support,load,np.zeros_like(node0))

'''
Example 2b: Obtaining an instability mode using eigenvalue analysis
'''

# node0 = np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0]],dtype=np.float64)
# member = np.array([[0,1],[1,2],[3,4],[4,5],[1,4],[2,5],[0,4],[1,3],[1,5],[2,4]],dtype=np.int32)
# support = np.array([[1,1,1],[0,0,1],[0,0,1],[1,1,1],[0,0,1],[0,0,1]],dtype=bool)
# A = np.array([1,0,1,1,1,1,1,1,0,1],dtype=np.float64)
# E = np.ones(member.shape[0],dtype=np.float64)

# Draw2D(node0,member,support,None,np.zeros_like(node0),line_width=A*2)

# eig_val, eig_mode = Stiffness_Matrix_Eig(node0,member,support,A,E)

# Draw2D(node0,member,support,None,eig_mode[0],line_width=A*2)
