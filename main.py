import numpy as np
from scipy.optimize import minimize
import cmath
import Draw
import DiscreteHolomorphicFunction as DHF
import GraphicStatics
import LinearStructuralAnalysis_Truss as LSAt

N_u = 8 # must be even value, and equal or less than N_v//2
N_v = 16
pts_for_structural_analysis = "pts_invstr" # "pts_in" or "pts_invstr" or "pts_invstr_rot_str"
layout = "partial" # "full" or "partial"
load_condition = "tip" # "tip" or "outer" (This does not affect the results when layout="full")
load_computation = "Christoffel" # "Christoffel" or "shear"
rotation_axis = (0.0,-0.1**0.5,0.9**0.5) # This does not affect the results unless pts_for_structural_analysis = "pts_invstr_rot_str"
rotation_angle = np.pi/8 # This does not affect the results unless pts_for_structural_analysis = "pts_invstr_rot_str"

if layout == "full":
    function_rule = "azimuth"
elif layout == "partial":
    function_rule = "radius"

'''
####################################################
### Part 1: Define discrete exponential function ###
####################################################
'''
'''
Step 1: Definition of a discrete exponential function

F(u,v) = rho*u + i*kappa*v

## Ex.A: specifying the azimuth angle
kappa = (azimuth angle)/N_v
rho = 2*np.arcsinh(np.sin(kappa/2))
F = lambda u,v: cmath.exp(complex(rho*u,kappa*v))

## Ex.B: specifying the external radius
kappa = np.pi/N_v
lR = np.inf
while np.abs(lR-4.81)>1.0e-10:
    rho = 2*np.arcsinh(np.sin(kappa/2))
    F = lambda u,v: cmath.exp(complex(rho*u,kappa*v))
    pts_in, quads = DHF.Quads_in(F,N_u,N_v,param_scale=1,shape_scale=1)
    lR = np.max(pts_in[:,0])
    kappa += (4.81-lR)*1e-2

'''
if function_rule == "azimuth":
    kappa = 2*np.pi/N_v
    rho = 2*np.arcsinh(np.sin(kappa/2))
    F = lambda u,v: cmath.exp(complex(rho*u,kappa*v))
elif function_rule == "radius":
    kappa = np.pi/N_v
    lR = np.inf
    while np.abs(lR-4.81)>1.0e-10:
        rho = 2*np.arcsinh(np.sin(kappa/2))
        F = lambda u,v: cmath.exp(complex(rho*u,kappa*v))
        pts_in, quads = DHF.Quads_in(F,N_u,N_v,param_scale=1,shape_scale=1)
        lR = np.max(pts_in[:,0])
        kappa += (4.81-lR)*1e-2
else:
    raise Exception("Invalid layout parameter")

'''
Step 2: Generate points based on the discrete holomorphic function
'''
pts_in, quads = DHF.Quads_in(F,N_u,N_v,param_scale=1,shape_scale=1)
size = np.linalg.norm(pts_in,axis=1).max()
cross_ratio = DHF.Cross_Ratio(pts_in, quads) # These values should be -1 if isothermic

'''
Step 3: Transformation
The function after Christoffel transformation preserves discrete isothermality (cross-ratio = -1);
i.e., the shapes before and after Christoffel transformation are self-reciprocal
'''
pts_chr = DHF.Christoffel_Transformation(pts_in,N_u,N_v,autoscale=True)
pts_invstr = DHF.Inverse_Stereographic_Projection(pts_in/size)*size
pts_invstr[:,2] *= -1 # flip the z-coordinates of the spherical shape
pts_invstr_chr = DHF.Christoffel_Transformation(pts_invstr,N_u,N_v)
pts_invstr_rot = DHF.Rotation_by_vector(pts_invstr,np.array(rotation_axis),rotation_angle)
pts_invstr_rot_str = DHF.Flip_V(DHF.Stereographic_Projection(pts_invstr_rot/size)*size,N_u,N_v)
pts_invstr_rot_str_chr = GraphicStatics.Dual_Quad(pts_invstr_rot_str,quads,N_u,N_v)
pts_invstr_rot_str_invstr = DHF.Inverse_Stereographic_Projection(pts_invstr_rot_str/size)*size # CAUTION: not isothermic (cross ratio is not -1). 

'''
Step 4: Create Michell's structures by taking diagonal lines of each polygon
'''
if layout == "full":
    diag_connectivity, fix_i, load_i = DHF.Diagonal_Lines_Connectivity(N_u,N_v) # full connectivity
elif layout == "partial":
    diag_connectivity, fix_i, load_i = DHF.Diagonal_Lines_Connectivity_Partial(N_u,N_v,load=load_condition) # partial connectivity

'''
Step 5: Set color and a caption for each discrete surface
'''
pts_all = np.stack((
    pts_in,                   # [0]: discrete exponential function
    pts_chr,                  # [1]: Christoffel transform of [0]
    pts_invstr,               # [2]: inverse stereographic project of [0]
    pts_invstr_chr,           # [3]: Christoffel transform of [2]
    pts_invstr_rot,           # [4]: rotated [2]
    pts_invstr_rot_str,       # [5]: stereographic project of [4]
    pts_invstr_rot_str_chr,   # [6]: Christoffel transform of [5]
    pts_invstr_rot_str_invstr # [7]: inverse stereographic project of [6]
    ))

colors = np.array((
    (0.0,0.8,0.6), # [0]
    (1.0,0.5,0.0), # [1]
    (0.0,0.3,1.0), # [2]
    (1.0,0.0,1.0), # [3]
    (0.0,0.3,1.0), # [4]
    (0.1,0.8,1.0), # [5]
    (0.8,0.2,0.4), # [6]
    (0.5,0.5,0.5)  # [7]
    ))

captions = np.array((
    "(Form 1) Original (discrete isothermic surface)",
    "(Force 1) Christoffel Trans.",
    "(Form 2) Inv. Stereographic Proj.",
    "(Force 2) Inv. Stereographic Proj. + Christoffel Trans.",
    "(Form 3a) Rotated Form 2",
    "(Form 3b) Stereographic Proj. of Form 2",
    "(Force 3) Force",
    "(Form 4) Inv. Stereographic Proj. of Form 3b (not isothermic)"
    ))

'''
Step 6: Draw
'''
Draw.MeshLine2D(pts_all[[0]],diag_connectivity,quads,colors[[0]],0.3,captions[[0]],r"result\plan.pdf")
Draw.MeshLine3D(pts_all[[6]],diag_connectivity,quads,colors[[6]],0.3,captions[[6]],r"result\axo.pdf",angle=[45,60])

'''
(optional)
Mirror 3D object in x-y plane to produce Michell sphere.
To reproduce the Michell sphere, please make sure that
1. Set N_u = 12 and N_v = 24
2. Use Ex. A scheme to specify the azimuth angle 2*pi, and set kappa as kappa = 2*np.pi/N_v in Step 1
3. Implement DHF.Diagonal_Lines_Connectivity() method in Step 4
'''
# pts_mirror = np.vstack([pts_invstr,pts_invstr*[1,1,-1]])
# Draw.MeshLine3D(pts_mirror[np.newaxis,:,:],np.vstack([diag_connectivity,diag_connectivity+len(pts_all[2])]),np.zeros((1,4),dtype=int),colors[[2]],0.3,captions[[2]],r"result\axo.pdf",angle=[0,0])

'''
Please modify the following codes depending on using 2D model or its inverse stereographic project (3D) hereafter.
'''
if pts_for_structural_analysis == "pts_in":
    pts_for_structural_analysis = pts_in
    pts_force = pts_chr
elif pts_for_structural_analysis == "pts_invstr":
    pts_for_structural_analysis = pts_invstr
    pts_force = pts_invstr_chr
elif pts_for_structural_analysis == "pts_invstr_rot_str":
    pts_for_structural_analysis = pts_invstr_rot_str
    pts_force = pts_invstr_rot_str_chr
else:
    raise Exception("Invalid pts_for_structural_analysis parameter")

'''
Step 7: Assign nodal loads
'''
load = np.zeros_like(pts_in)

if load_computation == "Christoffel":
    for i in load_i:
        i1 = i-(N_u+1) if i-(N_u+1) > 0 else load_i[-1]-(N_u+1) # The loaded node with the smallest index must be handled differently
        i2 = i+(N_u+1) if i+(N_u+1) < len(pts_in) else load_i[0]+(N_u+1) # The loaded node with the largest index must be handled differently
        load[i] = pts_force[i1]-pts_force[i2]

    if len(load_i) == 1: # If single load
        load[load_i] /= np.linalg.norm(load[load_i])
    elif len(load_i) > 1: # If multiple loads
        load[load_i] /= np.sum(np.linalg.norm(load[load_i],axis=1))

elif load_computation == "shear":
    total_load_vector_length = 0.0
    for i in load_i:
        m, n = np.divmod(i,N_u+1)
        if n == N_u:
            d = (pts_in[(m+1)*(N_u+1)+n]-pts_in[(m-1)*(N_u+1)+n])
            total_load_vector_length += np.linalg.norm(d)*np.sqrt(2)
        else:
            if i > (N_u+1)*(N_v)/2+N_u:
                d = pts_in[m*(N_u+1)+n+1]+pts_in[(m+1)*(N_u+1)+n]-2*pts_in[m*(N_u+1)+n]
            else:
                d = 2*pts_in[m*(N_u+1)+n]-pts_in[m*(N_u+1)+n+1]-pts_in[(m-1)*(N_u+1)+n]
            total_load_vector_length += np.linalg.norm(d)
        load[i] = d

    if len(load_i) == 1: # If single load
        load[load_i] /= np.linalg.norm(load[load_i])
    elif len(load_i) > 1: # If multiple loads
        load[load_i] /= total_load_vector_length

'''
####################################################
### Part 2: Structural Analysis and Optimization ###
####################################################
'''
'''
Step 1: Prepare inputs for structural analysis
'''
def cull_nodes(nodes,members,node_attrs,node_nums):
    '''
    Cull unused and duplicate nodes.

    ## input ##
    nodes[nn,3]<float>: nodal coordinates
    members[nm,2]<int>: member connectivity
    node_nums[:][:]<int>: list of original node indices

    ## output ##
    nodes[nn_,3]<float>: culled nodal coordinates
    members[nm,2]<int>: member connectivity
    node_nums[:][:_]<int>: list of culled node indices
    shift[nn]<int>: node number shift
    '''
    _, nodes_i, nodes_inv, count = np.unique(np.round(nodes,5),axis=0,return_index=True,return_inverse=True,return_counts=True)
    duplicate = np.ones(len(nodes),dtype=bool)
    duplicate[nodes_i] = False
    keep = np.zeros(len(nodes),dtype=bool)
    keep[members.flatten()] = True
    keep[duplicate] = False
    shift = np.array([np.sum(~keep[:i+1]) for i in range(len(nodes))],dtype=int)
    node_nums = [l[keep[l]] for l in node_nums]
    for i in np.where(count>1)[0]:
        b_to = nodes_i[i]
        b_from = np.setdiff1d(np.where(nodes_inv==i)[0],b_to)
        shift[b_from] = shift[b_to] + b_from - b_to
    return nodes[keep], members-shift[members], [l[keep] for l in node_attrs], [l-shift[l] for l in node_nums], shift

nodes, members, [load], [fix_i,load_i], shift = cull_nodes(pts_for_structural_analysis,diag_connectivity,[load],[fix_i,load_i])

if np.all(np.isclose(nodes[:,2],nodes[:,2].mean())):
    is3D = False
else:
    is3D = True

fix = np.zeros_like(nodes,dtype=bool)
fix[fix_i] = True
fix[:,2] = True # constrain vertical displacements regardless of 2D or 3D

'''
Step 2: Compute outward vector from 2 members connected to the tip loaded node
'''
tip_members = np.array(np.where(members==np.take(load_i,load_i.size//2)))
second_tip_nodes = [members[tip_member[0],(tip_member[1]+1)%2] for tip_member in tip_members.T]
vec = (2*nodes[np.take(load_i,load_i.size//2)]-nodes[second_tip_nodes[0]]-nodes[second_tip_nodes[1]]).flatten()

'''
Step 3: Rotate the shape
'''
nodes = DHF.Rotation_by_vector(nodes,np.array([0,0,1]),-np.arctan(vec[1]/vec[0]))
load = DHF.Rotation_by_vector(load,np.array([0,0,1]),-np.arctan(vec[1]/vec[0]))
# np.savetxt(r"result\node_init.dat",nodes, delimiter=',')

'''
Step 4: Visualize the internal force distribution
'''
internal_force = LSAt.Axial_Force_Determinate(nodes,members,fix,load)
# GraphicStatics.Dual_Isothermic(nodes,members,fix,load) # NOTE: This method properly works only when 2D and DHF.Diagonal_Lines_Connectivity_One() method with load="tip" is used in Step 4 in Part 1.

if is3D:
    Draw.Truss3D(nodes,members,np.any(fix[:,0:2],axis=1),load,internal_force,n_color=["None"],label=None,n_size=[5],l_width=2,name=r"result\truss_opt.png")
else:
    Draw.Truss2D(nodes,members,np.any(fix[:,0:2],axis=1),load,internal_force,n_color=["None"],label=None,n_size=[5],n_marker=["o"],l_width=2,l_color=["black","black"],name=r"result\truss_opt.png")

'''
Appx A: Obtain the self-equilibrium mode (set of internal force distribution);
        the number of self-equilibrium modes is equal to the static indeterminacy set nodal loads.
        The original structure is statically determinate, and constraining 1 dof will increase the
        static indeterminacy from 0 to 1; i.e., one self-equilibrium mode can be obtained. 
'''
# fix2 = np.copy(fix)
# fix2[load_i,1] = True
# _,_,self_force = LSAt.Self_Equilibrium_Matrix(nodes,members,fix2,False)
# self_force = self_force.flatten()
# ratio = internal_force/self_force

'''
Appx B: Verify if internal_force obtained by equilibrium matrix is identical to the internal forces obtained by stiffness method
'''
# size = np.ones(members.shape[0])
# _, force, _ = LSAt.Linear_Structural_Analysis(nodes,members,fix,load,size,np.ones(members.shape[0]))
# internal_force2 = force*size
# print(f"internal forces obtained by equilibirum matrix: {internal_force}")
# print(f"internal forces obtained by stiffness method: {internal_force2}")

'''
Step 5: Compute the objective function to be minimized: sum of axial force x length
'''
force_x_length = np.abs(internal_force)*LSAt.Length(nodes,members) # The values should be the same if isothermic and considering the full connectivity
objfun = np.sum(force_x_length)
print(f"objective function (axial force x length) = {objfun}")

'''
Step 6: Sensitivity of the objective function (axial force x length) with respect to nodal coordinates
'''
is_variable = ~fix
is_variable[load_i] =  False
if is3D:
    is_variable[:,2] = False
_, res = LSAt.Optimize_Force_Length(nodes,members,fix,load,is_variable,tol=1e20) # Obtains the initial solution by setting a huge tolerance
grad_analytic = res.jac # gradients of the initial solution
print("## Analytical gradients ##")
print(f"df_max{np.max(abs(grad_analytic))}")
print(f"df_median{np.median(abs(grad_analytic))}")
print(f"df_min{np.min(abs(grad_analytic))}")
print(f"df_norm{np.linalg.norm(grad_analytic)}")

'''
Appx C: Differential approximation of sensitivity. The analytical sensitivity above should be almost equal to the values below.
        Once it can be verified that these values are almost the same, analytical sensitivity should be used in view of accuracy.
'''
# objfun_d = np.empty(2)
# disturbance = [-5e-6, 5e-6]
# grad_approx = []
# for i in np.delete(np.arange(nodes.shape[0]),np.concatenate([fix_i,load_i])):
#     for j in range(2):
#         for k in range(2):
#             nodes_temp = np.copy(nodes)
#             nodes_temp[i,j] += disturbance[k]
#             objfun_d[k] = np.sum(LSAt.Length(nodes_temp,members)*np.abs(LSAt.Axial_Force_Determinate(nodes_temp,members,fix,load)))
#         grad_approx.append((objfun_d[1]-objfun_d[0])/(disturbance[1]-disturbance[0]))
# print("## Numerical gradients ##")
# print(f"df_max{np.max(np.abs(grad_approx))}")
# print(f"df_median{np.median(np.abs(grad_approx))}")
# print(f"df_min{np.min(np.abs(grad_approx))}")
# print(f"df_norm{np.linalg.norm(grad_approx)}")

'''
Step 7: Solve the optimization problem to obtain the true stationary point of Min.(axial force x length)
'''
nodes_opt, _ = LSAt.Optimize_Force_Length(nodes,members,fix,load,is_variable,lb=nodes[is_variable]-1.1,ub=nodes[is_variable]+1.1,tol=1.0e-7)
objfun = np.sum(LSAt.Length(nodes_opt,members)*np.abs(LSAt.Axial_Force_Determinate(nodes_opt,members,fix,load)))

print(f"objective function (axial force x length) = {objfun}")
print(f"Max. change in nodal position: {np.max(np.linalg.norm(nodes_opt-nodes,axis=1))}")

'''
Step 8: Visualize the optimal shape
'''
if is3D:
    Draw.Truss3D(np.stack((nodes,nodes_opt)),members,np.any(fix[:,0:2],axis=1),load*10,internal_force,n_color=[(0.2,0.8,0.0),(0.2,0.0,0.8)],label=["before","after"],n_size=[5,10],l_width=2,name=r"result\truss_opt.png")
else:
    Draw.Truss2D(np.stack((nodes,nodes_opt)),members,np.any(fix[:,0:2],axis=1),load,internal_force,n_color=[(0.6,0.6,0.6),(0.0,0.0,0.0)],label=["before","after"],n_size=[20,30],n_marker=["s","o"],l_width=2,l_color=["black","black"],name=r"result\truss_opt.png")
# np.savetxt(r"result\node_opt.dat",nodes_opt, delimiter=',')
# GraphicStatics.Dual_Isothermic(nodes_opt,members,fix,load) # This method properly works only when 2D and DHF.Diagonal_Lines_Connectivity_One() method with load="tip" is used in Step 4 in Part 1.
