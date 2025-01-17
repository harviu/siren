'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import modules, utils
from dataio import get_attr, get_volume, write_vts, Tracer
from hdf5_to_dict import load_tracer
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=128)

opt = p.parse_args()


class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        if opt.mode == 'mlp':
            self.model = modules.SingleBVPNet(type=opt.model_type, out_features=1, in_features=3)
        elif opt.mode == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', out_features=1, in_features=3)
        self.model.load_state_dict(torch.load(opt.checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']


sdf_decoder = SDFDecoder()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

try:
    data_path = os.environ['data']
except KeyError:
    data_path = './data/'
# save resampled volume
N = opt.resolution
volume = get_volume(sdf_decoder, N, max_batch=opt.batch_size)
volume = volume.numpy().transpose(2,1,0).flatten()
path_to_tracer = os.path.join(data_path,'tracer/torus_gw170817_traces_pruned_r250/tracers_00000050.h5part')
tracer = load_tracer(path_to_tracer)
coords = tracer[2]['Xcart']
attr = tracer[2]['Ye']
write_vts(coords,N,N,N,{'Ye':volume},'test.vts')

tracer_norm = Tracer(path_to_tracer,opt.batch_size)
pred_attr = get_attr(sdf_decoder,tracer_norm.coords,opt.batch_size)
psnr = 10*np.log10(4 /  ((tracer_norm.attr.squeeze() - pred_attr.numpy()) ** 2).mean())
print(psnr)