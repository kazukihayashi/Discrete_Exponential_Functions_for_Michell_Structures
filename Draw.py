import numpy as np
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.legend_handler import HandlerTuple
cm = plt.cm.get_cmap('coolwarm_r') # "coolwarm_r", "bwr_r", "RdBu_r"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15

def MeshLine2D(pts_all,lines,quads,mesh_color_list,alpha,caption_list,name):
    '''
    2D plot of quad meshes with the same topology

    (input)
    pts_all[n_mesh, n_point, 3]<float>: nodal locations
    lines[n_line, 2]<int>             : line connectivity
    quads[n_quad, 4]<int>             : quad connectivity
    color_list[n_mesh]<float>         : color in RGB(0.0-1.0) format
    alpha<float>                      : alpha in (0.0-1.0)
    caption_list[n_mesh]<str>         : caption of each mesh
    name<str>                         : output file path including directory
    '''

    pts_all = pts_all[:,:,0:2]
    fig = plt.figure()
    ax = plt.gca()

    for i in range(pts_all.shape[0]):
        ax.add_collection(PolyCollection(pts_all[i,quads],color=mesh_color_list[i],facecolor=np.append(mesh_color_list[i],alpha),label=caption_list[i]))
        line_ends = np.array([pts_all[i,lines[j,:]] for j in range(len(lines))])
        ax.add_collection(LineCollection(line_ends,linewidths=2.0,colors=mesh_color_list[i]))
        # ax.scatter(pts_all[fix_i,0],pts_all[fix_i,1],color="red")
        # ax.scatter(pts_all[load_i,0],pts_all[load_i,1],color="blue")
        for j in range(pts_all.shape[1]):
            ax.annotate(f"{j}",pts_all[i,j,0:2],color=mesh_color_list[i])

    ax.axis('equal')
    ax.axis('off')
    # plt.legend(loc=4)
    plt.savefig(name)
    plt.show()
    plt.close()
    return

def MeshLine3D(pts_all,lines,quads,mesh_color_list,alpha,caption_list,name,angle=None):
    '''
    3D plot of quad meshes with the same topology

    (input)
    pts_all[n_mesh, n_point, 3]<float>: nodal locations
    lines[n_line, 2]<int>             : line connectivity
    quads[n_quad, 4]<int>             : quad connectivity
    color_list[n_mesh]<float>         : color in RGB(0.0-1.0) format
    alpha<float>                      : alpha in (0.0-1.0)
    caption_list[n_mesh]<str>         : caption of each mesh
    name<str>                         : output file path including directory
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')

    for i in range(pts_all.shape[0]):
        # ax.scatter(pts_all[i,:,0],pts_all[i,:,1],pts_all[i,:,2],s=1,c=mesh_color_list[i],label=caption_list[i])
        ax.add_collection3d(Poly3DCollection(pts_all[i,quads],color=mesh_color_list[i],facecolor=np.append(mesh_color_list[i],alpha)))
        line_ends = np.array([pts_all[i,lines[j,:]] for j in range(len(lines))])
        ax.add_collection3d(Line3DCollection(line_ends,linewidths=2.0,colors=mesh_color_list[i]))

    lb = np.min(pts_all.reshape(-1,3),axis=0)
    ub = np.max(pts_all.reshape(-1,3),axis=0)
    plot_radius = 0.5*max(ub-lb)
    middle = np.mean(pts_all.reshape(-1,3),axis=0)
    ax.set_xlim3d(middle[0]-plot_radius,middle[0]+plot_radius)
    ax.set_ylim3d(middle[1]-plot_radius,middle[1]+plot_radius)
    ax.set_zlim3d(middle[2]-plot_radius,middle[2]+plot_radius)
    ax.set_box_aspect([1,1,1])
    ax.axis('off')
    # plt.legend(scatterpoints=100)
    if angle is not None:
        ax.view_init(elev=angle[0], azim=angle[1]) # For grid 3D truss
        # ax.view_init(elev=45, azim=60) # For grid 3D truss
    plt.savefig(name)
    plt.show()
    return

def Truss2D(pts_all,lines,fix,load,force,n_color,label,n_size,n_marker,l_width,l_color,name):
    if len(pts_all.shape)==2:
        pts_all = pts_all.reshape(1,*pts_all.shape)
        n_color = [n_color]
    pts_all = pts_all[:,:,0:2]
    
    fig = plt.figure()
    ax = plt.gca()
    p = [None for i in range(pts_all.shape[0])]
    for i in range(pts_all.shape[0]):
        p[i] = ax.scatter(pts_all[i,~fix,0],pts_all[i,~fix,1],color=n_color[i],s=n_size[i],marker=n_marker[i],edgecolors="face",zorder=i)
    ax.scatter(pts_all[0,fix,0],pts_all[0,fix,1],color="black",marker="^",zorder=10)
    loaded = np.any(load!=0,axis=1)

    loaded_2 = np.copy(loaded)
    loaded_2[:len(loaded)//2] = False
    loaded_1 = np.copy(loaded)
    loaded_1[len(loaded)//2:] = False
    ptx = pts_all[0,loaded,0] # ptx = np.concatenate([pts_all[0,loaded_1,0]-load[loaded_1,0]*0.85,pts_all[0,loaded_2,0]])
    pty = pts_all[0,loaded,1] # pty = np.concatenate([pts_all[0,loaded_1,1]-load[loaded_1,1]*0.85,pts_all[0,loaded_2,1]])
    ax.quiver(ptx,pty,load[loaded,0],load[loaded,1],color="red",scale=2,zorder=-20)

    if label is not None:
        line_ends = np.array([pts_all[0,lines[j,:]] for j in range(len(lines))])
        divnorm=colors.TwoSlopeNorm(vcenter=0.0)
        lc1 = LineCollection(line_ends,linestyle=':',linewidth=l_width,cmap=cm,colors=l_color[0],alpha=0.2,norm=divnorm,zorder=-10)
        ax.add_collection(lc1)

    # ax.quiver(pts_all[0,loaded,0],pts_all[0,loaded,1],load[loaded,0],load[loaded,1],color="red")
    line_ends = np.array([pts_all[-1,lines[j,:]] for j in range(len(lines))]) # using only the last set of points
    divnorm=colors.SymLogNorm(linthresh=0.101,vmin=-1.5,vmax=1.5)# colors.TwoSlopeNorm(vcenter=0.0,vmin=-1.5,vmax=1.5)
    lc2 = LineCollection(line_ends,linewidth=l_width,cmap=cm,colors=l_color[1],alpha=0.4,norm=divnorm,zorder=-5)
    # lc2.set_array(force)
    ax.add_collection(lc2)
    # axcb = plt.colorbar(lc2) 
    # axcb.set_label("(compression)   axial force [N]   (tension)")
    ax.axis('equal')
    # ax.axis("off")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if label is not None:
        legend = ax.legend([(p[0],lc1),(p[1],lc2)],label,handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.savefig(name)
    plt.show()
    plt.close()

def Truss3D(pts_all,lines,fix,load,force,n_color,label,n_size,l_width,name,angle=None):
    if len(pts_all.shape)==2:
        pts_all = pts_all.reshape(1,*pts_all.shape)
        n_color = [n_color]
    pts_all = pts_all[:,:,:]
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')
    p = [None for i in range(pts_all.shape[0])]
    for i in range(pts_all.shape[0]):
        p[i] = ax.scatter(pts_all[i,~fix,0],pts_all[i,~fix,1],color=n_color[i],s=n_size[i],marker="o",zorder=i)
    ax.scatter3D(pts_all[0,fix,0],pts_all[0,fix,1],pts_all[0,fix,2],color="black",marker="^",zorder=10)
    loaded = np.any(load!=0,axis=1)

    loaded_2 = np.copy(loaded)
    loaded_2[:len(loaded)//2] = False
    loaded_1 = np.copy(loaded)
    loaded_1[len(loaded)//2:] = False
    ptx = np.concatenate([pts_all[0,loaded_1,0]-load[loaded_1,0]*0.95,pts_all[0,loaded_2,0]])#ptx = pts_all[0,loaded,0] # ptx = np.concatenate([pts_all[0,loaded_1,0]-load[loaded_1,0]*0.85,pts_all[0,loaded_2,0]])
    pty = np.concatenate([pts_all[0,loaded_1,1]-load[loaded_1,1]*0.95,pts_all[0,loaded_2,1]])#pty = pts_all[0,loaded,1] # pty = np.concatenate([pts_all[0,loaded_1,1]-load[loaded_1,1]*0.85,pts_all[0,loaded_2,1]])
    ptz = np.concatenate([pts_all[0,loaded_1,2]-load[loaded_1,2]*0.95,pts_all[0,loaded_2,2]])#ptz = pts_all[0,loaded,2] # ptz = np.concatenate([pts_all[0,loaded_1,2]-load[loaded_1,2]*0.85,pts_all[0,loaded_2,2]])
    ax.quiver3D(ptx,pty,ptz,load[loaded,0],load[loaded,1],load[loaded,2],color="red")

    if label is not None:
        line_ends = np.array([pts_all[0,lines[j,:]] for j in range(len(lines))])
        divnorm=colors.TwoSlopeNorm(vcenter=0.0)
        lc1 = Line3DCollection(line_ends,linestyle=':',linewidth=l_width,cmap=cm,colors=(0.0,0.0,0.0,0.2),norm=divnorm,zorder=-10)
        ax.add_collection3d(lc1)

    line_ends = np.array([pts_all[-1,lines[j,:]] for j in range(len(lines))]) # using only the last set of points
    divnorm=colors.SymLogNorm(linthresh=0.04,vmin=-1,vmax=1)#divnorm=colors.TwoSlopeNorm(vcenter=0.0,vmin=-0.45,vmax=0.45)
    lc2 = Line3DCollection(line_ends,linewidth=l_width,cmap=cm,colors=(0.0,0.0,0.0,0.2),norm=divnorm,zorder=-5)
    lc2.set_array(force)
    ax.add_collection3d(lc2)
    axcb = plt.colorbar(lc2,shrink=0.8) 
    axcb.set_label("(compression)   axial force [N]   (tension)")

    lb = np.min(pts_all.reshape(-1,3),axis=0)
    ub = np.max(pts_all.reshape(-1,3),axis=0)
    plot_radius = 0.5*max(ub-lb)
    middle = np.mean(pts_all.reshape(-1,3),axis=0)
    ax.set_xlim3d(middle[0]-plot_radius,middle[0]+plot_radius)
    ax.set_ylim3d(middle[1]-plot_radius,middle[1]+plot_radius)
    ax.set_zlim3d(0,2*plot_radius)
    # ax.set_zlim3d(middle[2]-plot_radius,middle[2]+plot_radius)
    ax.set_box_aspect([1,1,1])

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(30,150)
    ax.w_xaxis.line.set_color((1.0,1.0,1.0,0.0))
    ax.w_yaxis.line.set_color((1.0,1.0,1.0,0.0))
    ax.w_zaxis.line.set_color((1.0,1.0,1.0,0.0))
    # ax.axis("off")
    

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if label is not None:
        legend = ax.legend([(p[0],lc1),(p[1],lc2)],label,handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.savefig(name)
    plt.show()
    plt.close()