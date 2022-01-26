import numpy as np 
from hdf5_to_dict import load_tracer
import os

class Stat:
    def __init__(self, dims:list, filename:str, by_harm:bool=False) -> None:
        self.by_harm = by_harm
        self.dims = dims 
        #Load data
        self.data = load_tracer(filename,None)
        self.attr_type = ['T', 'rho', 'Ye']
        self.n = self.data[0]['ntracers']
        self.attr = {}
        for a in self.attr_type:
            self.attr[a] = self.data[2][a]
        self.coord = self.data[2]['Xharm'] if self.by_harm else self.data[2]['Xcart']
        #Calcuate the statistics
        self.uni = {}
        self.copular = np.zeros((len(self.attr_type),len(self.attr_type)))
        




if __name__ == '__main__':
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    fname = os.path.join(data_path,'tracer/torus_gw170817_traces_pruned_r250/tracers_00000050.h5part')
    stat = Stat([100,100,100],fname,False)