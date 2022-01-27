import numpy as np
from sympy import false, true 
from hdf5_to_dict import load_tracer
import os

class Stat:
    def __init__(self, dims:list, filename:str, by_harm:bool=False) -> None:
        self.by_harm = by_harm
        self.dims = np.array(dims)
        #Load data
        self.data = load_tracer(filename,None)
        self.attr_type = ['T', 'rho', 'Ye']
        self.n = self.data[0]['ntracers']
        self.attr = {}
        for a in self.attr_type:
            self.attr[a] = self.data[2][a]
        self.coord = self.data[2]['Xharm'] if self.by_harm else self.data[2]['Xcart']
        self.extent = np.stack([np.min(self.coord,axis=0),np.max(self.coord,axis=0)],axis=0)
        self.interval = 1/(self.dims - 1) * (self.extent[1] - self.extent[0])
        #Calcuate the statistics
        blocks = []
        xx, yy, zz = dims
        for x in range(xx):
            blocks.append([])
            for y in range(yy):
                blocks[x].append([])
                for z in range(zz):
                    blocks[x][y].append({})
                    stats = blocks[x][y][z] 
                    stats['uni'] = {a:None for a in self.attr_type}
                    stats['uni']['x'] = None
                    stats['uni']['y'] = None
                    stats['uni']['z'] = None
                    stats['cor'] = None
        self.blocks = blocks

    def grid_to_xyz(self, grid:list):
        interval = self.interval
        xyz = interval * grid + self.extent[0]
        return xyz

    def check_cell(self,grid:list):
        for i,d in enumerate(grid):
            if d >=0 and d <= self.dims[i] - 2:
                return true
        return false

    def pad_grid(self,grid_id):
        #replicant padding
        grid_id = np.where(grid_id >= 0, grid_id, 0)
        grid_id = np.where(grid_id <= self.dims-1, grid_id, self.dims -1 )
        return grid_id


    def get_grid_of_cell(self, cell:list):
        if self.check_cell(cell):
            idx = [[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]]
            grid_dict = {}
            for id in idx:
                grid_dict[''.join(map(str,id))] = np.array(cell) + np.array(id)
            return grid_dict
        else:
            raise ValueError('The cell is not in a legal range.')

    def get_particle_in_range(self, range:dict):
        cond = np.array(range['000'])[None,:] <= self.coord
        cond1 = np.all(cond, axis = 1)
        
        cond = np.array(range['111'])[None,:] >= self.coord
        cond2 = np.all(cond, axis = 1)

        cond = cond1 & cond2
        return cond


    def cal_block(self, cell:list):
        grids = self.get_grid_of_cell(cell)
        coords_range = {k: self.grid_to_xyz(grid) for k,grid in grids.items()}
        block_mask = self.get_particle_in_range(coords_range)
        if block_mask.sum() > 0:
            attr = np.stack([self.attr[a][block_mask] for a in self.attr_type],axis=-1)
            coord = self.coord[block_mask]

            cor = np.corrcoef(np.concatenate([attr,coord],axis=1),rowvar=False)
            self.blocks[cell[0]][cell[1]][cell[2]]['cor'] = cor
            


        

        
        



if __name__ == '__main__':
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    fname = os.path.join(data_path,'tracer/torus_gw170817_traces_pruned_r250/tracers_00000050.h5part')
    dims = [11,11,5]
    stat = Stat(dims,fname,False)
    for i in range(dims[0]-1):
        for j in range(dims[1]-1):
            for k in range(dims[2]-1):
                stat.cal_block([i,j,k])