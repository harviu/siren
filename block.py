
from typing import Hashable
from hdf5_to_dict import load_tracer
import os
import numpy as np
import math

def thJ_of_X(Xharm):
    poly_alpha = 14
    poly_xt = 0.82
    poly_norm = 0.5 * math.pi * 1. / (1. + 1. / (poly_alpha + 1.) * 1. / math.pow(poly_xt, poly_alpha))
    y = 2 * Xharm[2] - 1
    thJ = poly_norm * y * (1. + (y / poly_xt) ** poly_alpha) / (poly_alpha + 1.) + 0.5 * math.pi
    return thJ
}

def thG_of_X(Xharm:np.array, hslope):
    return math.pi * Xharm[1] + ((1-hslope)/2) * math.sin(2 * math.pi * Xharm[1])

def harm2bl(Xharm:np.array, hslope=0.3):
    Xbl = np.zeros_like(Xharm)
    Xbl[2] = Xharm[2]
    Xbl[0] = math.exp(Xharm[0])
    thg = thG_of_X(Xharm,hslope)
    thj = thJ_of_X(Xharm)
    mks_smooth = 0.5
    
    th = thg + math.exp()

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

if __name__ == '__main__':
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    data = load_tracer(os.path.join(data_path,'tracer/torus_gw170817_traces_pruned_r250/tracers_00000050.h5part'))

    n = data[0]['ntracers']
    T = data[2]['T']
    rho = data[2]['rho']

    Xcart = data[2]['Xcart']
    Xharm = data[2]['Xharm']
    test_Xcart = Xcart[1]
    test_Xharm = Xharm[1]

    recon_Xcart = harm2cart(test_Xharm)
    print(recon_Xcart, test_Xcart)
