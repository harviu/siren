# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, modules

from torch.utils.data import DataLoader
import configargparse
import torch
from matplotlib import pyplot as plt


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')
p.add_argument('-m','--checkpoint_path', required=True, help='Checkpoint to trained model.')
p.add_argument('--dataset', type=str, default='isabel',
               help='Volume dataset')

opt = p.parse_args()

cuda = False

if opt.dataset == 'isabel':
    data_path = '/fs/project/PAS0027/Isabel_data_pressure/Pf35.bin.gz'
    vol_dataset = dataio.Isabel(data_path)
elif opt.dataset == 'tornado' :
    data_path = '../summer_intern/vorts_grid.bin'
    vol_dataset = dataio.Tornado(data_path)

coord_dataset = dataio.Implicit3DWrapper(vol_dataset, sidelength=vol_dataset.shape, sample_fraction=1)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model and load in checkpoint path
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, out_features=vol_dataset.channels,
                                 mode='mlp', hidden_features=1024, num_hidden_layers=3)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', in_features=3, out_features=vol_dataset.channels, mode=opt.model_type)
else:
    raise NotImplementedError
model.load_state_dict(torch.load(opt.checkpoint_path,map_location=torch.device('cpu') if not cuda else torch.device('cuda')))
if cuda:
    model.cuda()



# Evaluate the trained model

resolution = vol_dataset.shape
frames = [50]
# frames = np.arange(100)
Nslice = 5

with torch.no_grad():
    if cuda:
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...].cuda() for f in frames]
    else:
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...] for f in frames]
    for idx, f in enumerate(frames):
        coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
    coords = torch.cat(coords, dim=0)
    output = torch.zeros((len(frames), resolution[1] * resolution[2], 1))
    split = int(coords.shape[1] / Nslice)
    for i in range(Nslice):
        pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
        output[:, i*split:(i+1)*split, :] =  pred.cpu()
    # time_start = time.time()
    # for f in frames:
    #     coords = dataio.get_mgrid((1, resolution[1], resolution[2]),dim=3).cuda()
    #     coords[..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
    #     split = int(coords.shape[1] / Nslice)
    #     for i in range(Nslice):
    #         pred = model({'coords':coords[None, i*split:(i+1)*split, :]})['model_out'].cpu()
    # time_end = time.time()
    # print("test_time:", time_end - time_start)

pred_vid = output.view(len(frames), resolution[1], resolution[2], 1) / 2 + 0.5
pred_vid = torch.clamp(pred_vid, 0, 1)
gt_vid = torch.from_numpy(vol_dataset.vid[frames, :, :, :])
psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))

pred_vid = pred_vid.permute(0, 3, 1, 2)
gt_vid = gt_vid.permute(0, 3, 1, 2)
output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
plt.imshow(pred_vid.numpy().squeeze())
plt.savefig('pred_50.png')
plt.imshow(gt_vid.numpy().squeeze())
plt.savefig('gt_50.png')
