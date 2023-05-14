#Import plotting libraries for later
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import path
import matplotlib.tri as tri

import numpy as np
from numpy import pi, cos, sin, sqrt, log

import os


def gen_mesh_pts(pts, n_samples=1000, mesh_type='random'):
    w=abs(max(pts[0])-min(pts[0]))
    h=abs(max(pts[1])-min(pts[1]))
    tuple_pts=[(x, y) for x, y in zip(pts[0], pts[1])]
    p=path.Path(tuple_pts)

    if mesh_type=='random':
        #random uniform distribution
        np.random.seed()
        dist_unif_x=np.random.uniform(0., w, n_samples)+min(pts[0])
        dist_unif_y=np.random.uniform(0., h, n_samples)+min(pts[1])

        sample_pts=[]
        for x, y in zip(dist_unif_x, dist_unif_y):
            sample_pts.append([x,y])

    elif mesh_type=='grid':
        aspect=w/h
        if aspect>2:
            aspect=2
        n_w=int(np.round(np.sqrt(n_samples)*aspect))
        n_h=int(np.round(np.sqrt(n_samples)*1/aspect))
        x_pts=np.linspace(min(pts[0]), max(pts[0]), n_w)
        y_pts=np.linspace(min(pts[1]), max(pts[1]), n_h)
        xg, yg=np.meshgrid(x_pts, y_pts)
        sample_pts=[]
        for x, y in zip(xg.flatten(), yg.flatten()):
            sample_pts.append([x,y])
        
    
    found_pts=p.contains_points(sample_pts)
    mesh_pts=np.array([s_pts for s_pts, f_pts in zip(sample_pts, found_pts) if f_pts==True])
    return np.transpose(mesh_pts)

def gen_mask(mask_line, x_grid, y_grid, skip=5):
    tuple_pts=[(x, y) for x, y in zip(mask_line[0], mask_line[1])]
    p=path.Path(tuple_pts)
    if x_grid.shape==y_grid.shape:
        mask=np.zeros_like(x_grid[::skip, ::skip], dtype=bool)
        for I, (x_line,y_line) in enumerate(zip(x_grid[::skip, ::skip], y_grid[::skip, ::skip])): 
            for J, (x,y) in enumerate(zip(x_line, y_line)):
                mask[I, J]=not p.contains_point((x,y))

        return mask
    else:
        print('Data grid must be square')

def save_mesh(mesh_pts, fname='test_mesh.pts', plane='yz'):
    x=mesh_pts[0]
    y=mesh_pts[1]
    with open(fname, 'w') as f:
        for x_pts,y_pts in zip(x, y):
            if plane=='yz':
                f.write('{}\t{}\t{}\n'.format(0,x_pts,y_pts))
            elif plane=='xy':
                f.write('{}\t{}\t{}\n'.format(x_pts,y_pts,0))
            elif plane=='xz':
                f.write('{}\t{}\t{}\n'.format(x_pts,0,y_pts))

    return fname


def import_field(field_file=os.getcwd()+'\\'+'mag_E_test', plane='yz'):
    if plane=='yz':
        inds=[1,2,3]
    elif plane=='xy':
        inds=[0,1,3]
    elif plane=='xz':
        inds=[0,2,3]
    d=np.loadtxt(field_file, skiprows=1)
    x=[]
    y=[]
    z=[]
    for rows in d:
        x.append(rows[inds[0]])
        y.append(rows[inds[1]])
        z.append(rows[inds[2]])
    return np.array(x),np.array(y),np.array(z)

def mirror_x(pts, xline, concatenate=True):
    if concatenate==True:
        pts_x=np.concatenate((pts[0], xline-pts[0][::-1]))
        pts_y=np.concatenate((pts[1], pts[1][::-1]))
        return np.array([pts_x, pts_y])
    else:
        return np.array([xline-pts[0], pts[1]])

def interp_mesh(field_file, plane, boundary=None, mask=False, skip=1, bound_value=0):
    x,y,z=import_field(field_file=field_file, plane=plane)
    if type(boundary)!=type(None) and type(bound_value)!=type(None):
        x=np.concatenate((x, boundary[0]))
        y=np.concatenate((y, boundary[1]))
        z=np.concatenate((z, np.full(len(boundary[0]), bound_value)))

    xi=np.linspace(min(x), max(x), len(x))
    yi=np.linspace(min(y), max(y), len(y))
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    if mask==True and type(boundary)!=type(None):
        mask=gen_mask(boundary, Xi, Yi, skip=skip)

        zi=zi[::skip, ::skip]
        xi=xi[::skip]
        yi=yi[::skip]

        zi=np.ma.array(zi, mask=mask)
    
    elif mask==True and type(boundary)==type(None):
        raise Exception("A boundary must be defined for a mask to be used")

    return xi, yi, zi