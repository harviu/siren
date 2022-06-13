import numpy as np
from sympy import false, true 
from hdf5_to_dict import load_tracer
import os, math
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import UniformUnivariate, GaussianKDE, GaussianUnivariate, BetaUnivariate

class Stat:
    def __init__(self, dims:list, filename:str, by_harm:bool=False) -> None:
        self.by_harm = by_harm
        self.dims = np.array(dims)
        #Load data
        self.data = load_tracer(filename,None)
        # self.attr_type = ['T', 'rho', 'Ye']
        self.attr_type = ['Ye']
        self.n = self.data[0]['ntracers']
        self.attr = {}
        for a in self.attr_type:
            self.attr[a] = self.data[2][a]
        self.coord = self.data[2]['Xharm'] if self.by_harm else self.data[2]['Xcart']
        self.extent = np.stack([np.min(self.coord,axis=0),np.max(self.coord,axis=0)],axis=0)
        self.interval = 1/(self.dims - 1) * (self.extent[1] - self.extent[0])
        self._cal_all_blocks()

    def grid_to_xyz(self, grid:list):
        return self.interval * np.array(grid) + self.extent[0]

    def xyz_to_grid(self, xyz:list):
        return ((np.array(xyz) - self.extent[0]) // self.interval).astype(int)

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

    def _helper_sample(self,c):
        grid = self.xyz_to_grid(c)
        if not self.check_cell(grid):
            grid = np.where(grid == self.dims-1, self.dims -2, grid)
        copula = self.blocks[tuple(grid)]['copula']
        sample = copula.sample(1, conditions={
                'x':c[0],
                'y':c[1],
                'z':c[2],
            }).to_numpy()
        return sample[0]

    def sample_at_particles(self):
        attr_array = np.stack([self.attr[a] for a in self.attr_type],axis=-1)
        data = np.concatenate([self.coord,attr_array],axis = 1)
        recon = np.zeros((self.n, self.coord.shape[1]+len(self.attr_type),))
        # from multiprocessing import Pool
        # pool = Pool(8)
        # print(pool.map(self._helper_sample, self.coord))
        for i in range(0,len(self.coord)):
            """
            bug1: when using GausianKDE, sample error,
            bug2: fall in empty block.
            """
            c = self.coord[i]
            print(i,end='\r')
            try:
                grid = self.xyz_to_grid(c)
                if not self.check_cell(grid):
                    grid = np.where(grid == self.dims-1, self.dims -2, grid)
                    print(grid)
                copula = self.blocks[tuple(grid)]['copula']
                sample = copula.sample(1, conditions={
                        'x':c[0],
                        'y':c[1],
                        'z':c[2],
                    }).to_numpy()
                recon[i] = sample[0]
            except:
                recon[i] = data[i]
        error = data - recon
        self.error = error[:,3]
        print(error)
        print(error.min(), error.max())
        mse = (error ** 2).mean()
        print(mse)
        psnr = 10 * math.log10(0.07457909845/mse)
        print(psnr)

            

    def _cal_block(self, cell:list):
        grids = self.get_grid_of_cell(cell)
        coords_range = {k: self.grid_to_xyz(grid) for k,grid in grids.items()}
        block_mask = self.get_particle_in_range(coords_range)
        if block_mask.sum() > 0:
            coord = self.coord[block_mask]
            attr = {k:self.attr[k][block_mask] for k in self.attr_type}
            df_dict = {
                'x': coord[:,0],
                'y': coord[:,1],
                'z': coord[:,2],
            }
            df_dict.update(attr)
            df = pd.DataFrame.from_dict(df_dict)
            copula = GaussianMultivariate(
                {
                    'x':UniformUnivariate,
                    'y':UniformUnivariate,
                    'z':UniformUnivariate,
                    # 'T':GaussianKDE,
                    # 'rho':GaussianKDE,
                    'Ye': GaussianUnivariate,
                })
            copula.fit(df)
            return {'copula': copula}

            # manually calcuate copular
            # attr_array = np.stack([self.attr[a][block_mask] for a in self.attr_type],axis=-1)
            # cor = np.corrcoef(np.concatenate([attr_array,coord],axis=1),rowvar=False)
            # self.blocks[cell[0]][cell[1]][cell[2]]['cor'] = cor
            # uni = {}
            # uni['x'] = np.histogram(coord[:,0])
            # uni['y'] = np.histogram(coord[:,1])
            # uni['z'] = np.histogram(coord[:,2])
            # for a in self.attr_type:
            #     uni[a] = np.histogram(attr[a])
            # self.blocks[cell[0]][cell[1]][cell[2]]['uni'] = uni

    def _cal_all_blocks(self):
        #Calcuate the statistics
        xx, yy, zz = self.dims
        blocks = np.empty((xx,yy,zz),dtype=dict)
        for x in range(xx-1):
            for y in range(yy-1):
                for z in range(zz-1):
                    blocks[x,y,z] = self._cal_block([x,y,z])
        self.blocks = blocks



