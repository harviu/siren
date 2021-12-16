'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
import torch
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
import numpy as np
from functools import partial

try:
    data_path = os.environ['data']
except KeyError:
    data_path = './data/'

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in steps until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--tracer_path', type=str, default=os.path.join(data_path,'tracer/torus_gw170817_traces_pruned_r250'),
               help='Path to the tracer file')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()


tracer_dataset = dataio.Tracers(opt.tracer_path,opt.batch_size)
dataloader = DataLoader(tracer_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=4)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=4)
model.cuda()

# Define the loss
loss_fn = loss_functions.particle

root_path = os.path.join(opt.logging_root, opt.experiment_name)

#write the ground true slices to tb first
# summaries_dir = os.path.join(root_path, 'summaries')
# utils.cond_mkdir(summaries_dir)

# slice_coords_2d = dataio.get_mgrid(128)
# gt_img = {}

# yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
# yz_slice_gt = dataio.resample(tracer_dataset.coords,tracer_dataset.attr,yz_slice_coords.numpy(),0.04)
# yz_slice_gt = (yz_slice_gt +1) /2
# gt_img['yz'] = dataio.lin2img(yz_slice_gt[:,:,-1:])

# xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
#                                 torch.zeros_like(slice_coords_2d[:, :1]),
#                                 slice_coords_2d[:,-1:]), dim=-1)
# xz_slice_gt = dataio.resample(tracer_dataset.coords,tracer_dataset.attr,xz_slice_coords.numpy(),0.04)
# xz_slice_gt = (xz_slice_gt +1) /2
# gt_img['xz'] = dataio.lin2img(xz_slice_gt[:,:,-1:])

# xy_slice_coords = torch.cat((slice_coords_2d[:,:2],0.25*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
# xy_slice_gt = dataio.resample(tracer_dataset.coords,tracer_dataset.attr,xy_slice_coords.numpy(),0.04)
# xy_slice_gt = (xy_slice_gt +1) /2
# gt_img['xy']  = dataio.lin2img(xy_slice_gt[:,:,-1:])

# summary_fn = partial(utils.write_particle_summary, gt_img = gt_img)
summary_fn = utils.write_particle_series_summary

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)
