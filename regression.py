import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import uci_datasets
import time
import csv

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train deep networks on a UCI regression dataset.")
    parser.add_argument("--dataset", type=str, default="challenger", help="UCI dataset")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of units in hidden layers")
    parser.add_argument("--depth", type=int, default=4, help="Depth of the neural network")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 1.0, 10.0], help="Lambda values for smooth rank")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay rate for optimizer")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on - 'cpu' or 'cuda'")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--evaluation_batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--csv_name", type=str, default="metrics.csv", help="Name of the CSV file to save metrics")
    parser.add_argument("--alpha", type=float, default=0.9, help="Decay rate for running covariance matrix")
    parser.add_argument("--beta", type=float, default=0.1, help="Coefficient for smooth rank regularization in the loss")
    return parser.parse_args()

def main(args):
    device = torch.device(args.device)

    # Load UCI regression dataset
    data = uci_datasets.Dataset(args.dataset)
    x_train, y_train, x_test, y_test = data.get_split(split=0)

    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_data, batch_size=args.evaluation_batch_size, shuffle=False)

    train_eval_loader = DataLoader(train_data, batch_size=args.evaluation_batch_size, shuffle=True)

    class DeepNetwork(nn.Module):
        def __init__(self, input_size, output_size, depth, hidden_size):
            super(DeepNetwork, self).__init__()
            self.layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())] +
                [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(depth - 2)] +
                [nn.Linear(hidden_size, output_size)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    input_size = x_train.shape[1]
    output_size = 1

    model = DeepNetwork(input_size, output_size, args.depth, hidden_size=args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def compute_F_batch_per_layer(x_batch):
        F_batch_per_layer = [x_batch.T @ x_batch for x_batch in x_batch]
        return F_batch_per_layer

    def compute_M_per_layer(model, loader):
        M_per_layer = [0] * len(model.layers)
        n = 0
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                activations = x_batch
                for l, layer in enumerate(model.layers):
                    M_per_layer[l] += activations.T @ activations
                    activations = layer(activations)
                n += len(x_batch)
            for l in range(len(model.layers)):
                M_per_layer[l] /= n
        return M_per_layer

    def compute_metrics(M, lambdas):
        eigs = torch.linalg.eigvalsh(M)
        p = eigs / eigs.sum()
        er = (-p * torch.log(p)).sum().exp().cpu().item()
        smooth_ranks = [(eigs / (eigs + lam)).sum().cpu().item() for lam in lambdas]
        return er, smooth_ranks

    def compute_smooth_rank(M, lam):
        eigs = torch.linalg.eigvalsh(M)
        return (eigs / (eigs + lam)).sum()

    def evaluate(loader):
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                total_loss += loss.item() * x_batch.size(0)
                total_samples += x_batch.size(0)
        avg_loss = total_loss / total_samples
        return avg_loss

    # Create CSV file
    csv_file = open(args.csv_name, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["Epoch", "Layer", "Depth", "Effective Rank"]
    header.extend([f"Smooth Rank (λ={lam})" for lam in args.lambdas])
    header.extend(["Train Loss", "Test Loss"])
    csv_writer.writerow(header)

    F_per_layer = [torch.zeros(1, device=device) for _ in range(args.depth)]

    for epoch in range(args.epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            loss = 0
            activations = x_batch
            for l, layer in enumerate(model.layers):
                if 1 <= l <= args.depth - 2:
                    F_per_layer[l] = args.alpha * F_per_layer[l].detach() + (1 - args.alpha) * activations.T @ activations
                    loss += compute_smooth_rank(F_per_layer[l], 1.0)
                activations = layer(activations)
            output = activations
            loss += criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        train_loss = evaluate(train_eval_loader)
        test_loss = evaluate(test_loader)
        M_per_layer = compute_M_per_layer(model, train_eval_loader)  # Compute M_per_layer for the whole dataset
        metrics_per_layer = [compute_metrics(M, args.lambdas) for M in M_per_layer]
        effective_ranks_per_layer, smooth_ranks_per_layer = zip(*metrics_per_layer)

        for l in range(args.depth):
            row = [epoch, l, args.depth, effective_ranks_per_layer[l]]
            row.extend(smooth_ranks_per_layer[l])
            row.extend([train_loss, test_loss])
            csv_writer.writerow(row)

        print(f"Epoch {epoch}: Train Loss = {train_loss}, Test Loss = {test_loss}")

    csv_file.close()

if __name__ == "__main__":
    main(parse_arguments())
