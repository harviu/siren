from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import numpy as np

class Spline(Dataset):
    def __init__(self, fname):
        self.spline = np.load(fname)
        self.n, self.t, self.d = self.spline.shape
        self.t = self.t - 1
        self.start_min = self.spline[:,0,:].min()
        self.start_max = self.spline[:,0,:].max()
        # self.start_mean = self.spline[:,0,:].mean()
        self.spline[:,0,:] = (self.spline[:,0,:] - self.start_min) / (self.start_max - self.start_min)
        # from matplotlib import pyplot as plt 
        # plt.hist(self.spline[:,0,:].flatten())
        # plt.show()
        # exit()

    def __getitem__(self, index):
        tra_idx = index // self.t
        t_idx = index - (self.t * tra_idx)
        start = self.spline[tra_idx][0] 
        time = np.array([t_idx / self.t],dtype=np.float32)
        end = self.spline[tra_idx][t_idx+1]
        return start, time, end
    def __len__(self):
        return self.n * self.t

class Basic(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Linear(cin,cout),
            nn.LayerNorm(cout),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.basic(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.start_mlp = nn.Sequential(
            Basic(3,64),
            Basic(64,128),
            Basic(128,256),
            Basic(256,512),
            nn.Linear(512,512),
            nn.ReLU(),
        )
        self.time_mlp = nn.Sequential(
            Basic(1,16),
            Basic(16,32),
            Basic(32,64),
            Basic(64,128),
            Basic(128,256),
            Basic(256,512),
            nn.Linear(512,512),
            nn.ReLU(),
        )
        self.end_mlp = nn.Sequential(
            Basic(1024,1024),
            Basic(1024,512),
            Basic(512,256),
            Basic(256,128),
            Basic(128,64),
            nn.Linear(64,3),
            nn.Sigmoid(),
        )

    def forward(self,start, time):
        z1 = self.start_mlp(start)
        z2 = self.time_mlp(time)
        z = torch.cat([z1,z2],dim=-1)
        y = self.end_mlp(z)
        return y



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (start, time, end) in enumerate(train_loader):
        start, time, end = start.to(device), time.to(device), end.to(device)
        # print(start[0],time[0])
        optimizer.zero_grad()
        output = model(start, time)
        # print(output[0])
        # print(end[0])
        loss = F.mse_loss(output, end, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(start), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    # n = test_loader.dataset.n
    # t = test_loader.dataset.t
    # all_output = np.zeros([n * t, 3],dtype=np.float32)
    # idx = 0
    with torch.no_grad():
        for start, time, end in test_loader:
            start, time, end = start.to(device), time.to(device), end.to(device)
            output = model(start, time)
            test_loss += F.mse_loss(output, end, reduction='mean').item() * len(start)  # sum up batch loss
            # all_output[idx:idx + len(start)] = output.cpu().numpy()
            # idx += len(start)
    # np.save('output_spline',all_output)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10240, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=30000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size,'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size,'shuffle': False }
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                    }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset = Spline('spline_training.npy')
    train_loader = torch.utils.data.DataLoader(dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    last_loss = -1
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        if last_loss != -1:
            if  abs(last_loss - test_loss) < 1e-8:
                break
        last_loss = test_loss
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "spline_model.pth")


if __name__ == '__main__':
    main()