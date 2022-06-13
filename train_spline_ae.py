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

    def __getitem__(self, index):
        return self.spline[index][0],self.spline[index][1:]
    def __len__(self):
        return len(self.spline)

class Net(nn.Module):
    def __init__(self,latent_size,num_control_point):
        super(Net, self).__init__()
        self.mlp1 = nn.Linear(num_control_point * 3,512)
        self.mlp2 = nn.Linear(512,256)
        self.mlp3 = nn.Linear(256,latent_size)
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_control_point * 3)

    def encoder(self, x):
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.relu(x)
        x = self.mlp3(x)
        return x

    def decoder(self, x):
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        y = F.sigmoid(x)
        return y

    def forward(self,x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).view(-1,args.num_control_points * 3)
        optimizer.zero_grad()
        output = model(target)
        loss = F.mse_loss(output, target, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    all_output = np.zeros((len(test_loader.dataset),args.num_control_points,3),dtype=np.float32)
    idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).view(-1,args.num_control_points * 3)
            output = model(target)
            test_loss += F.mse_loss(output, target, reduction='mean').item() * len(data)  # sum up batch loss
            all_output[idx:idx+len(data)] = output.view(-1,args.num_control_points,3).cpu().numpy()
            idx += len(data)
    np.save('output_spline',all_output)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
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
    parser.add_argument('--latent-size', type=int, default=20, metavar='N',
                        help='Latent dimensionality')
    parser.add_argument('--num-control-points', type=int, default=96, metavar='N',
                        help='Number of control points')
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

    dataset = Spline('spline.npy')
    train_loader = torch.utils.data.DataLoader(dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    model = Net(args.latent_size,args.num_control_points).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    last_loss = -1
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(args, model, device, test_loader)
        if last_loss != -1:
            if  abs(last_loss - test_loss) < 1e-8:
                break
        last_loss = test_loss
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "spline.pth")


if __name__ == '__main__':
    main()