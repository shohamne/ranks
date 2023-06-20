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

class DeepNetwork(nn.Module):
    def __init__(self, input_size, output_size, depth, hidden_size):
        super(DeepNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.ReLU())] +
            [nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.ReLU()) for _ in range(depth - 2)] +
            [nn.Sequential(nn.Linear(hidden_size, output_size, bias=False))]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        for layer in self.layers:
            z = x
            x = layer(x)
        return x, z


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
parser.add_argument('--dataset', type=str, default='gas', help='Dataset to use')
parser.add_argument("--split", type=int, default=0, help="Split of the dataset to use for training")
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs')
parser.add_argument('--depth', type=int, default=3, help='Depth of network')
parser.add_argument('--hidden_size', type=int, default=256, help='Size of hidden layer')
parser.add_argument('--output_size', type=int, default=1, help='Size of output layer')
parser.add_argument('--weight_decay_rate', type=float, default=0.0, help='Weight decay rate for optimizer')
parser.add_argument('--sigma', type=float, default=1.0, help='Sigma value for kernel')
parser.add_argument('--csv_name', type=str, default='clean_results.csv', help='CSV filename for the results')
parser.add_argument('--optimise_logdet', type=str2bool, default=True, help='CSV filename for the results')
parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'rbf'], help='Kernel to use')



args = parser.parse_args()

def compute_logdet(K):
    return torch.linalg.eigvalsh(K).log().sum()
    #return torch.slogdet(K)[1]

# Open a new CSV file for writing
with open(args.csv_name, mode='w') as csv_file:
    result_writer = csv.writer(csv_file)
    result_writer.writerow(['Epoch', 'Train loss', 'Learning rate', 'Train MSE', 'Test MSE'])

    # Rest of your code with all the constants replaced with args.variable_name
    sigma = nn.Parameter(torch.tensor(args.sigma))

    data = uci_datasets.Dataset(args.dataset)
    x_train, y_train, x_test, y_test = data.get_split(split=0)
    x_train = torch.tensor(x_train, dtype=torch.float32).to(args.device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(args.device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(args.device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(args.device)
    mean = torch.mean(torch.tensor(x_train, dtype=torch.float32), dim=0)
    variance = torch.var(torch.tensor(x_train, dtype=torch.float32), dim=0)

    x_train = (x_train - mean) / torch.sqrt(variance)
    x_test = (x_test - mean) / torch.sqrt(variance)

    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    input_size = x_train.shape[1]

    model = DeepNetwork(input_size, args.output_size, args.depth, hidden_size=args.hidden_size)
    model = model.to(args.device)

    # Select the kernel based on the argument
    if args.kernel == 'linear':
        kernel = LinearKernel().to(args.device)
    elif args.kernel == 'rbf':
        kernel = GaussianKernel().to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',min_lr=1e-5, factor=0.1, patience=25, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            x_batch, y_batch = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()
            outputs, z = model(x_batch)
            K = kernel(z) + sigma**2*torch.eye(z.shape[0], device=args.device)
            loss = y_batch.T @ K.inverse() @ y_batch 
            if args.optimise_logdet:
                loss += compute_logdet(K)
            loss = loss / len(y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, z_train = model(x_train)
            _, z_test = model(x_test)
            K = kernel(z_train) + sigma**2*torch.eye(z_train.shape[0], device=args.device) 
            Kinv = K.inverse()
            loss = (y_train.T @ Kinv @ y_train + compute_logdet(K)) / len(y_train)
            alpha =  z_train.T @ Kinv @ y_train
            train_mse = criterion(kernel(z_train, z_train) @ Kinv @ y_train, y_train)
            test_mse = criterion(kernel(z_test, z_train) @ Kinv @ y_train, y_test)  
            
        
        print(f"Epoch {epoch+1}/{args.epochs}.. Train loss: {loss.item():.6f} lr: {optimizer.param_groups[0]['lr']:.6f} Train MSE: {train_mse.item():.6f} Test MSE: {test_mse.item():.6f}")
        result_writer.writerow([epoch+1, loss.item(), optimizer.param_groups[0]['lr'], train_mse.item(), test_mse.item()])

print('Finished Training')
