import numpy as np
from scipyx import ellipj
from scipy.special import ellipk

def Quads_in(F,N_u,N_v,param_scale=1,shape_scale=1):
    '''
    CAUTION: param_scale must be 1 if enforcing isothermality.
    '''

    pts = np.empty(((N_u+1)*(N_v+1),3),dtype=np.float64)
    count = 0
    for m in range(0,N_v+1):
        for n in range(0,N_u+1):
            fnm = F(param_scale*n,param_scale*m)
            pts[count] = [fnm.real,fnm.imag,0.0]
            count += 1
    quads = np.empty((N_u*N_v,4),dtype=int)
    count = 0
    for m in range(N_v):
        for n in range(N_u):
            quads[count] = [(N_u+1)*m+n,(N_u+1)*m+n+1,(N_u+1)*(m+1)+n+1,(N_u+1)*(m+1)+n]
            count += 1
    
    return pts*shape_scale, quads

def Cross_Ratio(pts,quads):

    cr = np.empty(len(quads))
    f = pts[:,0] + 1j*pts[:,1]
    cr = (f[quads[:,0]]-f[quads[:,1]])*(f[quads[:,2]]-f[quads[:,3]])/((f[quads[:,1]]-f[quads[:,2]])*(f[quads[:,3]]-f[quads[:,0]]))
    
    return cr

def Inverse_Stereographic_Projection(pts_in, autoscale=True):

    ds = 1 + pts_in[:,0]**2 + pts_in[:,1]**2
    pts_out = 2*pts_in/ds[:,np.newaxis]
    pts_out[:,2] = (-1 + pts_in[:,0]**2 + pts_in[:,1]**2)/ds
    if autoscale:
        scale = Scale(pts_in,pts_out)
    else:
        scale = 1.0

    return pts_out*scale

def Stereographic_Projection(pts_in, autoscale=True):

    l = np.linalg.norm(pts_in,axis=1)
    if np.any(np.abs(l-1.0)>1e-10):
        raise Exception("Input nodal positions must satisfy x^2+y^2+z^2=1 to implement Stereographic projection.")
    pts_out = np.zeros_like(pts_in)
    pts_out[:,0:2] = pts_in[:,0:2]/np.expand_dims(1-pts_in[:,2],1)
    if autoscale:
        scale = Scale(pts_in,pts_out)
    else:
        scale = 1.0
    return pts_out*scale

def Christoffel_Transformation(pts_in,N_u,N_v,autoscale=True):
    pts_out = np.copy(pts_in)
    for n in range(N_u):
        d = pts_in[n+1]-pts_in[n]
        d /= np.sum(d**2)
        pts_out[n+1] = pts_out[n] - d
    for m in range(N_v):
        for n in range(N_u+1):
            d = pts_in[(N_u+1)*(m+1)+n]-pts_in[(N_u+1)*m+n]
            d /= np.sum(d**2)
            pts_out[(N_u+1)*(m+1)+n] = pts_out[(N_u+1)*m+n] + d
    if autoscale:
        scale = Scale(pts_in,pts_out)
    else:
        scale = 1.0
    return pts_out*scale

def Schwarz_Christoffel_Mapping_Square_to_Disk(pts_in):
    Ke = ellipk(0.5)
    sn,cn,dn,ph = ellipj(Ke*(1+1j)*(pts_in[:,0]+1j*pts_in[:,1])/2-Ke,1/np.sqrt(2))
    uv = (1-1j)*cn/np.sqrt(2)
    if pts_in.shape[1] == 2:
        pts_out = np.stack((uv.real,uv.imag)).T
    elif pts_in.shape[1] == 3:
        pts_out = np.stack((uv.real,uv.imag,np.zeros(len(pts_in)))).T
    return pts_out

def Scale(pts_in,pts_out):
    ub_in = np.max(pts_in,axis=0)
    lb_in = np.min(pts_in,axis=0)
    ub_out = np.max(pts_out[:,0:2],axis=0)
    lb_out = np.min(pts_out[:,0:2],axis=0)
    scale = np.max(ub_in-lb_in)/np.max(ub_out-lb_out)
    return scale

def Rotation_by_vector(node,v,theta):
    '''
    Rotate nodes by theta [rad] around vector v.
    
    (input)
    node[n,3]<float>: nodal coordinates
    v[3]<float>     : vector
    theta<float>    : rotation degree in counter-clockwise direction

    (output)
    node_rotated[n,3]<float>: rotated nodal coordinates
    '''
    c = np.cos(theta)
    s = np.sin(theta)
    v = v/np.linalg.norm(v)
    rot = np.array(
        [[v[0]**2*(1-c)+c,v[0]*v[1]*(1-c)-v[2]*s,v[2]*v[0]*(1-c)+v[1]*s],
        [v[0]*v[1]*(1-c)+v[2]*s,v[1]**2*(1-c)+c,v[1]*v[2]*(1-c)-v[0]*s],
        [v[2]*v[0]*(1-c)-v[1]*s,v[1]*v[2]*(1-c)+v[0]*s,v[2]**2*(1-c)+c]]
        )
    return (rot@node.T).T

def Flip_V(pts_in,N_u,N_v):
    '''
    Flip the nodal indexing in u direction.
    '''
    pts_out = np.empty_like(pts_in)
    for i in range(N_v+1):
        pts_out[(N_u+1)*i:(N_u+1)*(i+1)] = pts_in[(N_u+1)*i:(N_u+1)*(i+1)][::-1]
    return pts_out

def Diagonal_Lines_Connectivity(N_u,N_v,flip=True):
    '''
    Create a Michell-like wheel topology from grid information N_u and N_v.

    (input)
    N_u<int>: number of grids in n direction
    N_v<int>: number of grids in m direction
    flip<bool>: If True, flip the diagonals and take the members next to those in the case "Flip=False".

    (output)
    diag_c[idx,2]<int>: connectivity of bar members comprising a Michell-like structure
    fix_i[:]<int>: indices of the fixed nodes
    load_i[1]<int>: index of the loaded node
    '''

    if flip:
        shift = 1
    else:
        shift = 0

    A = [(N_u+1)*m+n if (m+n)%2==shift else (N_u+1)*m+n+1 for n in range(N_u) for m in range(N_v)]
    B = [(N_u+1)*(m+1)+n+1 if (m+n)%2==shift else (N_u+1)*(m+1)+n for n in range(N_u) for m in range(N_v)]

    diag_c = np.vstack((A,B)).T
    load_i = np.array([(N_u+1)*(i+1)-1 for i in range(shift,N_v+1,2)],dtype=int)
    fix_i = np.array([(N_u+1)*i for i in range(shift,N_v+1,2)],dtype=int)
    
    return diag_c, fix_i, load_i

def Diagonal_Lines_Connectivity_Partial(N_u,N_v,flip=False,load="tip"):
    '''
    Create a partial Michell-like topology from grid information N_u and N_v.

    (input)
    N_u<int>: number of grids in n direction
    N_v<int>: number of grids in m direction
    flip<bool>: If True, flip the diagonals and take the members next to those in the case "Flip=False".
    load<str>: "tip": the load is applied to tip node only
               "outer": the load is applied to outer free nodes

    (output)
    diag_c[idx,2]<int>: connectivity of bar members comprising a Michell-like structure
    fix_i[:]<int>: indices of the fixed nodes
    load_i[1]<int>: index of the loaded node (tip node only)
    '''
    if N_u*2 != N_v:
        raise Exception("If using the Diagonal_Lines_Connectivity_One method, please make sure that N_u*2 == N_v.")
    
    if flip:
        shift = 1
    else:
        shift = 0

    if load == "tip":
        load_i = np.array([(N_u+1)*(N_u+1-shift)-1],dtype=int)
    elif load == "outer":
        li = []
        for i in range(1,N_v//2):
            li.append((N_u+2)*i)
        midload_i = (N_u+2)*N_v//2
        for i in range(N_v//2,N_v):
            li.append(midload_i+N_u*(i-N_v//2))
        load_i = (np.array(li)-shift*(N_u+1))%((N_u+1)*(N_v+1))      
        
    A = [(N_u+1)*m+n if (m+n)%2==shift else (N_u+1)*m+n+1 for m in range(N_v) for n in range(N_u) ]
    B = [(N_u+1)*(m+1)+n+1 if (m+n)%2==shift else (N_u+1)*(m+1)+n for m in range(N_v) for n in range(N_u) ]

    diag_c = np.vstack((A,B)).T
    idx = []
    for i in range(N_v):
        if i<N_v//2:
            rep = i+1
        else:
            rep = N_v-i
        for j in range(rep):
            idx.append(N_u*(i-shift)+j)

    fix_i = np.arange(N_v+1)*(N_u+1)

    return diag_c[idx], fix_i, load_i
