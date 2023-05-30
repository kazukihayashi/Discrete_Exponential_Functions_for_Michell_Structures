import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from operator import itemgetter

def Dual_Quad(pts,quad,nu,nv):

    if pts.shape[1] == 3:
        pts = pts[:,:2]

    pts_out = np.empty_like(pts)*np.nan # initialize with nan
    pts_out[quad[0,[0,1]]] = pts[quad[0,[0,1]]]
    for v in range(nv):
        for u in range(nu):
            index = quad[nu*v+u]
            pts_out[index] = Christoffel_Quad_One(pts[index],pts_out[index])

    pts_out = np.hstack([pts_out,np.zeros((pts.shape[0],1))])

    return pts_out

def Christoffel_Quad_One(pts_four_in,pts_four_out):
    '''
    pts_four_in[4,2]: input four corners of a quad element, aligned in anti-clockwise direction
    pts_four_out[4,2]: known four corner positions of the Christoffel transform, aligned in clockwise direction.
                       NaN if the position is unknown
    '''
    assert np.any(np.isnan(pts_four_out)), "pts_four_out are all unknown."

    if np.all(~np.isnan(pts_four_out[[0,3]])): # fliped line position is known
        if np.any(np.isnan(pts_four_out[1])):
            pts_four_out[1] = Intersection_2D(np.vstack([pts_four_out[0],pts_four_out[0]+pts_four_in[0]-pts_four_in[1]]),np.vstack([pts_four_out[3],pts_four_out[3]+pts_four_in[0]-pts_four_in[2]]))
        if np.any(np.isnan(pts_four_out[2])):
            pts_four_out[2] = Intersection_2D(np.vstack([pts_four_out[0],pts_four_out[0]+pts_four_in[1]-pts_four_in[3]]),np.vstack([pts_four_out[3],pts_four_out[3]+pts_four_in[2]-pts_four_in[3]]))
    elif np.all(~np.isnan(pts_four_out[[0,1]])):
        if np.any(np.isnan(pts_four_out[2])):
            pts_four_out[2] = Intersection_2D(np.vstack([pts_four_out[0],pts_four_out[0]+pts_four_in[1]-pts_four_in[3]]),np.vstack([pts_four_out[1],pts_four_out[1]+pts_four_in[1]-pts_four_in[2]]))
        if np.any(np.isnan(pts_four_out[3])):
            pts_four_out[3] = Intersection_2D(np.vstack([pts_four_out[0],pts_four_out[0]+pts_four_in[0]-pts_four_in[3]]),np.vstack([pts_four_out[1],pts_four_out[1]+pts_four_in[0]-pts_four_in[2]]))
    elif np.all(~np.isnan(pts_four_out[[2,3]])):
        if np.any(np.isnan(pts_four_out[0])):
            pts_four_out[0] = Intersection_2D(np.vstack([pts_four_out[2],pts_four_out[2]+pts_four_in[1]-pts_four_in[3]]),np.vstack([pts_four_out[3],pts_four_out[3]+pts_four_in[0]-pts_four_in[3]]))
        if np.any(np.isnan(pts_four_out[1])):
            pts_four_out[1] = Intersection_2D(np.vstack([pts_four_out[2],pts_four_out[2]+pts_four_in[1]-pts_four_in[2]]),np.vstack([pts_four_out[3],pts_four_out[3]+pts_four_in[0]-pts_four_in[2]]))

    assert np.all(~np.isnan(pts_four_out)), "Invalid computation."

    return pts_four_out

def Intersection_2D(pts_two_1, pts_two_2):
    '''
    pts_two_1[2,2]: Input line defined by the two points that the line pass through.
    pts_two_2[2,2]: Input line defined by the two points that the line pass through.
    '''
    # Calculate the slope of the first line
    slope_1 = (pts_two_1[1,1]-pts_two_1[0,1])/(pts_two_1[1,0]-pts_two_1[0,0]) if (pts_two_1[1,0]-pts_two_1[0,0]) != 0 else float('inf')

    # Calculate the slope of the second line
    slope_2 = (pts_two_2[1,1]-pts_two_2[0,1])/(pts_two_2[1,0]-pts_two_2[0,0]) if (pts_two_2[1,0]-pts_two_2[0,0]) != 0 else float('inf')

    # Check if the lines are parallel
    if slope_1 == slope_2:
        return None  # Lines are parallel, no intersection

    # Calculate the intersection point
    if slope_1 == float('inf'):  # First line is vertical
        x_intersect = pts_two_1[0,0]
        y_intersect = slope_2 * (pts_two_1[0,0] - pts_two_2[0,0]) + pts_two_2[0,1]
    elif slope_2 == float('inf'):  # Second line is vertical
        x_intersect = pts_two_2[0,0]
        y_intersect = slope_1 * (pts_two_2[0,0] - pts_two_1[0,0]) + pts_two_1[0,1]
    else:
        x_intersect = (slope_1 * pts_two_1[0,0] - slope_2 * pts_two_2[0,0] + pts_two_2[0,1] - pts_two_1[0,1]) / (slope_1 - slope_2)
        y_intersect = slope_1 * (x_intersect - pts_two_1[0,0]) + pts_two_1[0,1]

    return np.array([x_intersect,y_intersect])

def Dual_Isothermic(pts,members,fix,load):

    nn = len(pts) # number of nodes
    nm = len(members) # number of bar members

    load_i = np.where(np.any(load!=0.0,axis=1))[0] # indices of loaded nodes
    load_vec = load[load_i] # Pick up one load vector
    next_node_i = set({*load_i}) # Node candidates to be chosen as node_i
    '''
    TODO In general, next_node_i should not include all the loaded nodes.
         This program works only when the number of loaded nodes is one.
    '''
    node_i = load_i[0] # The edges connected to this node is constructed in the while loop

    remain_node_i = set(range(nn)) # The while loop continues until the set becomes empty
    complete_node_i = set() # Indices of nodes that are already handled as node_i

    edge_dir = pts[members[:,1]] - pts[members[:,0]] # Edge direction vectors
    edge_dir = np.vstack((edge_dir,load_vec)) # Load vectors are also handled as edges in the force diagram
    
    connected_member = [[] for i in range(nn)] # indices of members connected to each node
    for i in range(nm):
        for j in range(2):
            connected_member[members[i,j]].append(i)
    for i,v in enumerate(load_i): # Load vectors are also handled as edges in the force diagram
        connected_member[v].append(nm+i)

    edge_pos = [None for _ in range(len(edge_dir))] # The positions of edge ends (None if unknown)
    edge_pos[nm] = np.array([[0,0,0],load_vec[0]]) # The edge position associated with one of the load vectors shall be known
    
    while len(remain_node_i) > 0:
        edges_i = []
        pts_i = []
        for j in connected_member[node_i]:
            if edge_pos[j] is None:
                edges_i.append(j)
            else:
                pts_i.append(edge_pos[j])

        if len(edges_i) == 2:
            pts_i_independent, count = np.unique(np.round(np.vstack(pts_i),5),axis=0,return_counts=True)
            pts_i_independent = pts_i_independent[count==1]
            edge_dir_i = edge_dir[edges_i]
            members_i = members[edges_i]
            for j in range(2):
                if members_i[j,0] == node_i:
                    members_i[j] = members_i[j][::-1]
                    edge_dir_i[j] *= -1
            sol = np.linalg.solve(np.array([[edge_dir_i[0,0],-edge_dir_i[1,0]],[edge_dir_i[0,1],-edge_dir_i[1,1]]]),(pts_i_independent[1]-pts_i_independent[0])[0:2])
            if np.all(sol>0):
                for j in range(2):
                    edge_pos[edges_i[j]] = np.vstack((pts_i_independent[j],pts_i_independent[j]+sol[j]*edge_dir_i[j]))
            elif np.all(sol<0):
                for j in range(2):
                    edge_pos[edges_i[(j+1)%2]] = np.vstack((pts_i_independent[j],pts_i_independent[j]-sol[(j+1)%2]*edge_dir_i[(j+1)%2]))

        complete_node_i.add(node_i)
        next_node_i |= set(members_i[:,0]) # add elements
        feasible = False
        
        for i in next_node_i:
            is_None = itemgetter(*connected_member[i])([type(ep) is type(None) for ep in edge_pos]) # None flag whether the edges connecting to node i has None property
            if type(is_None)==type(()):
                n_None= sum(itemgetter(*connected_member[i])([type(ep) is type(None) for ep in edge_pos])) # Number of edges with None property
            elif type(is_None)==bool:
                n_None = int(is_None) # Number of edges with None property
            if n_None < 2: # This node can be ignored because other nodes will cover the edge connected to this node
                remain_node_i.discard(i)
                complete_node_i.add(i)
            elif n_None == 2: # A node having 2 unknown edge positions will be the next node_i in the while loop
                node_i = i
                feasible = True
        if len(remain_node_i) > 0 and not feasible:
            raise Exception("Unable to construct the force diagram.")

        next_node_i -= set(complete_node_i) # Remove already considered nodes from the candidate nodes
    
    support_pos = []
    for i in np.where(np.all(fix,axis=1))[0]:
        pts_i = [edge_pos[j] for j in connected_member[i]]
        pts_i_independent, count = np.unique(np.round(np.vstack(pts_i),5),axis=0,return_counts=True)
        pts_i_independent = pts_i_independent[count==1]
        support_pos.append(pts_i_independent)
    
    Draw(edge_pos[:nm],support_pos,edge_pos[nm:]) # Draw a force diagram
    nodes = np.unique(np.round(np.vstack(edge_pos),5),axis=0) # Extract nodal positions without duplicate
    np.savetxt(r"result\node_force_diagram.dat",nodes, delimiter=',') # Output the nodal positions
    return edge_pos

def Draw(edge_pos,support_pos,load_pos):
    fig, ax = plt.subplots()
    [ax.add_line(Line2D(edge_pos[i][:,0],edge_pos[i][:,1],ls='-',c=(1.0,0.5,0.0))) for i in range(len(edge_pos)) if type(edge_pos[i])!=type(None)]
    [ax.add_line(Line2D(load_pos[i][:,0],load_pos[i][:,1],ls='-',c='red')) for i in range(len(load_pos))]
    [ax.add_line(Line2D(support_pos[i][:,0],support_pos[i][:,1],ls=':',c='black')) for i in range(len(support_pos))]
    ax.axis('equal')
    plt.show()