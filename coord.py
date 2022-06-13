
from hdf5_to_dict import load_tracer, get_tracer_fnams
from dataio import get_tracer_num
import os,sys
import numpy as np
import math
import h5py
import time
import numba

def thJ_of_X(Xharm):
    poly_alpha = 14
    poly_xt = 0.82
    poly_norm = 0.5 * math.pi * 1. / (1. + 1. / (poly_alpha + 1.) * 1. / poly_xt ** poly_alpha)
    y = 2 * Xharm[1] - 1
    thJ = poly_norm * y * (1 + (y / poly_xt) ** poly_alpha / (poly_alpha + 1.)) + 0.5 * math.pi
    return thJ

def thG_of_X(Xharm:np.array, hslope):
    return math.pi * Xharm[1] + ((1-hslope)/2) * math.sin(2 * math.pi * Xharm[1])

def harm2bl(Xharm:np.array, hslope=0.3):
    Xbl = np.zeros_like(Xharm)
    Xbl[2] = Xharm[2]
    Xbl[0] = math.exp(Xharm[0])

    thg = thG_of_X(Xharm,hslope)
    thj = thJ_of_X(Xharm)
    mks_smooth = 0.5
    Rout = 20.
    N1TOT = 256
    Reh = 0.9
    Rin = math.exp((N1TOT * math.log(Reh) / 5.5 - math.log(Rout)) / (-1. + N1TOT / 5.5));
    startx = [math.log(Rin)]
    th = thg + math.exp(mks_smooth * (startx[0] - Xharm[0])) * (thj-thg)
    Xbl[1] = th

    return Xbl

def bl2cart(Xbl:np.array):
    Xcart = np.zeros_like(Xbl)
    r = Xbl[0]
    th = Xbl[1]
    ph = Xbl[2]
    Xcart[0] = r * math.sin(th) * math.cos(ph)
    Xcart[1] = r * math.sin(th) * math.sin(ph)
    Xcart[2] = r * math.cos(th)
    return Xcart


def harm2cart(Xharm:np.array):
    Xbl = harm2bl(Xharm)
    Xcart = bl2cart(Xbl)
    return Xcart

def get_raw_coords(f):
    with h5py.File(f,'r') as f:
        if 'Step#0' in f.keys():
            Xharm = f['Step#0/Xharm'][()]
            Xcart = f['Step#0/Xcart'][()]
            id = f['Step#0/id'][()]
        else:
            Xharm = f['Xharm'][()]
            Xcart = f['Xcart'][()]
            id = f['id'][()]
    return np.concatenate([Xharm,Xcart],axis=-1), id

def helper_fun(tra, c):
    idx = tra['len']
    tra['coord'][idx] = c
    tra['len'] += 1
    return tra

def load_trajectory(tracer_dir):
    files = get_tracer_fnams(tracer_dir)
    if not files:
        raise ValueError("This directory is empty!")
    last_id = None
    for i,f in enumerate(files):
        print(i)
        coord, particle_index = get_raw_coords(f)
        if last_id is None:
            p_to_k = {p:k for k,p in enumerate(particle_index)}
            tra = np.zeros((coord.shape[0], len(files),coord.shape[1]),dtype=np.float32)
            length = np.ones((coord.shape[0]),dtype= np.int32)
            tra[:,0,:] = coord
        else:
            particle_index, _, array_id2 = np.intersect1d(last_id, particle_index, assume_unique=True,return_indices=True)
            index = [p_to_k[p] for p in particle_index]
            tra[index,length[index]] = coord[array_id2]
            length[index] += 1
        last_id = particle_index
        # n += tracer_num[i]
    return tra, length

if __name__ == '__main__':
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    fname = os.path.join(data_path,'tracer/torus_gw170817_traces_pruned_r250/')
    tra, length = load_trajectory(fname)
    np.save('trajectories',tra)
    np.save('tra_len',length)
    

