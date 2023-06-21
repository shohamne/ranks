import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import uci_datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from kernels import LinearKernel, GaussianKernel
from utils import hermitian_logdet_and_inverse

torch.set_default_dtype(torch.float64)

class DeepNetwork(nn.Module):
    def __init__(self, input_size):
        super(DeepNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_size, 250, bias=False), nn.ReLU())] +
            [nn.Sequential(nn.Linear(250, 250, bias=False), nn.ReLU()) for _ in range(3 - 2)] +
            [nn.Sequential(nn.Linear(250, 1, bias=False))]
        )
        # self.layers = nn.ModuleList(
        #     [nn.Sequential(nn.Linear(input_size, 200, bias=False), nn.ReLU())] +
        #     [nn.Sequential(nn.Linear(200, 200, bias=False), nn.ReLU())] +
        #     [nn.Sequential(nn.Linear(200, 50, bias=False), nn.ReLU())] +
        #     [nn.Sequential(nn.Linear(50, 2, bias=False), nn.ReLU())] +
        #     [nn.Sequential(nn.Linear(2, 1, bias=False))]
        # )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        activations = []
        for layer in self.layers:
            activations.append(x)
            x = layer(x)
        return x, activations

    @property
    def w(self):
        return self.layers[-1][0].weight

    @property
    def weigths():
        weights = []
        for layer in self.layers:
            weights.append(layer[0].weight)    
        return weights

        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

# Add the argument parser for command line arguments
parser = argparse.ArgumentParser(description='GP Regression')
parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
parser.add_argument('--dataset', type=str, default='energy', help='Dataset to use')
parser.add_argument("--split", type=int, default=0, help="Split of the dataset to use for training")
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs')
parser.add_argument('--weight_decay_rate', type=float, default=0.0, help='Weight decay rate for optimizer')
parser.add_argument('--sigma', type=float, default=0.1, help='Sigma value for kernel')
parser.add_argument('--csv_name', type=str, default='clean_results.csv', help='CSV filename for the results')
parser.add_argument('--optimize_logdet', type=str2bool, default=True, help='CSV filename for the results')
parser.add_argument('--optimize_logdet2', type=str2bool, default=False, help='CSV filename for the results')
parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'rbf'], help='Kernel to use')
parser.add_argument('--standard_loss', type=str2bool, default=False, help='Run standard training')
parser.add_argument('--l2_reg', type=float, default=0.0, help='Lambda coefficient for l2 regularization')




args = parser.parse_args()

# Open a new CSV file for writing
with open(args.csv_name, mode='w') as csv_file:
    result_writer = csv.writer(csv_file)
    result_writer.writerow(['Epoch', 'Train loss', 'Learning rate', 'Train MSE', 'Test MSE', 'logdet', 'sigma', 'Train NN MSE', 'Test NN MSE', 'Train loss2'])

    # Rest of your code with all the constants replaced with args.variable_name
    sigma = nn.Parameter(torch.tensor(args.sigma), requires_grad=args.optimize_logdet)

    data = uci_datasets.Dataset(args.dataset)
    x_train, y_train, x_test, y_test = data.get_split(split=0)
    x_train = torch.tensor(x_train).to(args.device)
    y_train = torch.tensor(y_train).to(args.device)
    x_test = torch.tensor(x_test).to(args.device)
    y_test = torch.tensor(y_test).to(args.device)
    mean = torch.mean(x_train, dim=0)
    variance = torch.var(x_train, dim=0)
    variance[variance==0.0]=1

    x_train = (x_train - mean) / torch.sqrt(variance)
    x_test = (x_test - mean) / torch.sqrt(variance)

    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    input_size = x_train.shape[1]

    model = DeepNetwork(input_size)
    model = model.to(args.device)

    # Select the kernel based on the argument
    if args.kernel == 'linear':
        kernel = LinearKernel().to(args.device)
    elif args.kernel == 'rbf':
        kernel = GaussianKernel(1.0, 1.0).to(args.device)

    linear_kernel = LinearKernel().to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + [sigma], lr=args.lr, weight_decay=args.weight_decay_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',min_lr=1e-5, factor=0.1, patience=50, verbose=True)

    running_loss = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            x_batch, y_batch = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()
            outputs, activations = model(x_batch)
            if args.standard_loss:
                loss = criterion(outputs, y_batch) + args.l2_reg*model.w.pow(2).sum() 
            else:
                z = activations[-1]
                K = kernel(z) + sigma**2*torch.eye(z.shape[0], device=args.device)
                Klogdet, Kinv = hermitian_logdet_and_inverse(K) 
                loss = y_batch.T @ Kinv @ y_batch 
                if args.optimize_logdet:
                    loss += Klogdet
                if args.optimize_logdet2:
                    assert args.optimize_logdet,  "args.optimize_logdet is 0"
                    z2 = activations[-2]
                    K = linear_kernel(z2) + sigma**2*torch.eye(z2.shape[0], device=args.device)
                    # todo check _ is improve?
                    Klogdet2, _ = hermitian_logdet_and_inverse(K)
                    loss += Klogdet2
                loss = loss / len(y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*len(y_batch)
            sigma.data.clamp_(min=1e-1)

        running_loss /= len(y_train)

        with torch.no_grad():
            o_train, a_train = model(x_train)
            o_test, a_test = model(x_test)
            z_train = a_train[-1]
            z_test = a_test[-1]
            
            K = kernel(z_train) + sigma**2*torch.eye(z_train.shape[0], device=args.device) 
            Klogdet, Kinv = hermitian_logdet_and_inverse(K)
            loss = y_train.T @ Kinv @ y_train + Klogdet

            K = kernel(a_train[-2]) + sigma**2*torch.eye(a_train[-2].shape[0], device=args.device)
            Klogdet2, _ = hermitian_logdet_and_inverse(K)
            loss2 = loss + Klogdet2

            loss /= len(y_train)
            loss2 /= len(y_train)
            alpha =  z_train.T @ Kinv @ y_train
            train_mse = criterion(kernel(z_train, z_train) @ Kinv @ y_train, y_train)
            test_mse = criterion(kernel(z_test, z_train) @ Kinv @ y_train, y_test)  
            nn_train_mse = criterion(o_train, y_train)
            nn_test_mse = criterion(o_test, y_test)
        
        print(f"Epoch OL{args.optimize_logdet:<7} {epoch+1}/{args.epochs}.. Train loss: {loss.item():.6f} lr: {optimizer.param_groups[0]['lr']:.6f} Train MSE: {train_mse.item():.6f}"
              f" Test MSE: {test_mse.item():.6f} logdet: {Klogdet.item()/len(y_train):.6f} sigma: {sigma.item():.6f}"
              f" running loss: {running_loss:.6f} {nn_train_mse.item():.6f} {nn_test_mse.item():.6f} loss2: {loss2.item():.6f}")
        # Added Klogdet.item() and sigma.item() to the row
        result_writer.writerow([epoch+1, loss.item(), optimizer.param_groups[0]['lr'], train_mse.item(), test_mse.item(), 
                                Klogdet.item()/len(y_train), sigma.item(), nn_train_mse.item(), nn_test_mse.item(), loss2.item()])
        csv_file.flush()
        if args.standard_loss:
            scheduler.step(nn_train_mse)
        else:
            scheduler.step(train_mse)
        
print('Finished Training')
