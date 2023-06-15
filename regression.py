import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import uci_datasets 


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train deep networks on a UCI regression dataset.")
    parser.add_argument("--dataset", type=str, default="housing", help="UCI dataset")
    parser.add_argument("--hidden_size", type=int, default=256, help="Number of units in hidden layers")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the neural network")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--sigma0", type=float, default=1.0, help="Initial sigma")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on - 'cpu' or 'cuda'")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--evaluation_batch_size", type=int, default=2000, help="Batch size for evaluation")
    parser.add_argument("--csv_name", type=str, default="metrics.csv", help="Name of the CSV file to save metrics")
    parser.add_argument("--alpha", type=float, default=0.1, help="Decay rate for running covariance matrix")
    parser.add_argument("--start_layer", type=int, default=3, help="First layer to regulize it's input")
    parser.add_argument("--stop_rank_reg", type=int, default=1e32, help="Epoch to stop rank regularization")
    parser.add_argument("--split", type=int, default=0, help="Split of the dataset to use for training")
    parser.add_argument("--dimension_metric", type=str, default="log_det", choices=["smooth_rank", "log_det"], help="Dimension metric to compute")
    parser.add_argument('--optimize_activations', type=str2bool, default=True, help='Optimize activations if this argument is set to yes')
    parser.add_argument('--optimize_rank', type=str2bool, default=True, help='Optimize rank if this argument is set to yes')
    parser.add_argument('--optimize_sigma', type=str2bool, default=False, help='Optimize sigma if this argument is set to yes')
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use for training")
    return parser.parse_args()



def main(args):
    device = torch.device(args.device)

    # Create a summary writer
    summary_writer = SummaryWriter("logs")

    # Load UCI regression dataset
    data = uci_datasets.Dataset(args.dataset)
    x_train, y_train, x_test, y_test = data.get_split(split=args.split)

    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_data, batch_size=args.evaluation_batch_size, shuffle=True)

    train_eval_loader = DataLoader(train_data, batch_size=args.evaluation_batch_size, shuffle=False)

    class DeepNetwork(nn.Module):
        def __init__(self, input_size, output_size, depth, hidden_size):
            super(DeepNetwork, self).__init__()
            self.layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.Tanh())] +
                [nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.Tanh()) for _ in range(depth - 2)] +
                [nn.Sequential(nn.Linear(hidden_size, output_size, bias=False))]
            )
            self.sigma = torch.nn.Parameter(torch.tensor(args.sigma0), requires_grad=args.optimize_sigma)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    input_size = x_train.shape[1]
    output_size = 1

    model = DeepNetwork(input_size, output_size, args.depth, hidden_size=args.hidden_size).to(device)
    criterion = nn.MSELoss(reduction='sum')

    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    def compute_M_and_evaluate(loader):
        M_per_layer = [0] * len(model.layers)
        mse = 0.0
        total_samples = 0
        smooth_rank_per_layer = []
        l2_norm = 0.0
        sigma_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                activations = x_batch
                for l, layer in enumerate(model.layers):
                    M = activations.T @ activations
                    M_per_layer[l] += M
                    activations = layer(activations)
                mse += criterion(activations, y_batch) 
                total_samples += x_batch.size(0)
            for l, layer in enumerate(model.layers):
                smooth_rank = compute_smooth_rank(M_per_layer[l], model.sigma) if args.dimension_metric == "smooth_rank" \
                                else compute_log_det(M_per_layer[l], model.sigma)
                #print('$', l, smooth_rank, M_per_layer[l][0][0], x_batch[0][0], len(loader.dataset))
                smooth_rank_per_layer.append(smooth_rank)
                l2_norm += layer[0].weight.pow(2).sum()
                if args.dimension_metric == "log_det":
                    sigma_loss += (total_samples-len(M_per_layer[l]))*torch.log(model.sigma)
            avg_l2_norm = l2_norm / total_samples
            avg_sigma_loss = sigma_loss / total_samples
            avg_mse = mse / total_samples
            avg_smooth_rank_per_layer = torch.tensor(smooth_rank_per_layer)/total_samples
            avg_smooth_rank = avg_smooth_rank_per_layer[2:].sum()
            avg_loss = avg_mse/model.sigma + avg_l2_norm + avg_sigma_loss + avg_smooth_rank
            
        return [m.cpu().item() for m in avg_smooth_rank_per_layer], avg_smooth_rank.cpu().item(), avg_loss.cpu().item(), avg_mse.cpu().item(), \
            avg_l2_norm.cpu().item(), avg_sigma_loss.cpu().item()

    def compute_smooth_rank(M, lam):
        eigs = torch.linalg.eigvalsh(M)
        return (eigs / (eigs + lam)).sum()

    def compute_log_det(M, lam):
        return torch.linalg.eigvalsh(M + lam*torch.eye(M.shape[0], device=device)).log().sum()
        
    # Create CSV file
    csv_file = open(args.csv_name, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["Epoch", "Model Sigma", "Train Loss", "Train MSE", "Test MSE", "L2 Norm", "Sigma Loss"]
    for l in range(args.depth):
        header.extend([f"Layer {l} Smooth Rank"])
    csv_writer.writerow(header)

    F_per_layer = [torch.zeros(1, device=device) for _ in range(args.depth)]

    
    for epoch in range(args.epochs):
        acc_loss = None
        acc_mse = None
        acc_smooth_rank = None
        acc_sigma_loss = None
        acc_l2_norm = None 
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            l2_norm = torch.tensor(0.0, device=device)
            sigma_loss = torch.tensor(0.0, device=device)
            smooth_rank = torch.tensor(0.0, device=device)

            activations = x_batch
            for l, layer in enumerate(model.layers):
                with torch.enable_grad() if l >= args.start_layer else torch.no_grad():
                    F = activations.T @ activations
                F_per_layer[l] = args.alpha * F_per_layer[l].detach() + (1 - args.alpha) * F
                with torch.enable_grad() if args.optimize_rank else torch.no_grad():                    
                    layer_smooth_rank = compute_smooth_rank(F_per_layer[l], model.sigma) if args.dimension_metric == "smooth_rank" \
                                else compute_log_det(F_per_layer[l], model.sigma)
                    sigma_loss  += (len(x_batch)-len(F_per_layer[l]))*torch.log(model.sigma)
                smooth_rank += layer_smooth_rank
                activations = layer(activations)
                l2_norm += layer[0].weight.pow(2).sum() 
                
            output = activations
            mse = criterion(output, y_batch)
            loss = mse/model.sigma + l2_norm + sigma_loss + smooth_rank
            loss.backward()

            acc_loss = loss.item()/len(x_batch)*0.1 + acc_loss*0.9 if acc_loss is not None else loss.item()/len(x_batch)
            acc_mse = mse.item()/len(x_batch)*0.1 + acc_mse*0.9 if acc_mse is not None else mse.item()/len(x_batch)
            acc_smooth_rank = smooth_rank.item()/len(x_batch)*0.1 + acc_smooth_rank*0.9 if acc_smooth_rank is not None else smooth_rank.item()/len(x_batch)
            acc_sigma_loss = sigma_loss.item()/len(x_batch)*0.1 + acc_sigma_loss*0.9 if acc_sigma_loss is not None else sigma_loss.item()/len(x_batch)
            acc_l2_norm = l2_norm.item()/len(x_batch)*0.1 + acc_l2_norm*0.9 if acc_l2_norm is not None else l2_norm.item()/len(x_batch)
            
            optimizer.step()

        #acc_mse /= len(train_loader)
        smooth_rank_per_layer, smooth_rank, train_loss, train_mse, l2_norm, sigma_loss = compute_M_and_evaluate(train_loader)
        _, _, _, test_mse, _, _ = compute_M_and_evaluate(test_loader)

        row = [epoch, model.sigma.item(), train_loss, train_mse, test_mse, l2_norm, sigma_loss]
        for l in range(args.depth):
            row.extend([smooth_rank_per_layer[l]])
        csv_writer.writerow(row)

        print(f"L{args.start_layer}, S{args.split}, Epoch {epoch}: Train Loss={train_loss:.4f}, Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}, L2 Norm={l2_norm:.4f}, "
            f"Sigma Loss={sigma_loss:.4f}, Smooth Rank={smooth_rank:.4f}, MSE diff={test_mse-train_mse:.4f}, MSE div={test_mse/train_mse:.4f}, "
            #Acc Loss={acc_loss:.4f}, Acc MSE={acc_mse:.4f}, "
            #f"Acc Smooth Rank={acc_smooth_rank:.4f}, Acc Sigma Loss={acc_sigma_loss:.4f}, Acc L2 Norm={acc_l2_norm:.4f}, "
            f"Sigma={model.sigma:.4f}")
        #with summary_writer.as_default():
        summary_writer.add_scalar("Train Loss", train_loss, epoch)
        summary_writer.add_scalar("Train MSE", train_mse, epoch)
        summary_writer.add_scalar("Test MSE", test_mse, epoch)
        summary_writer.add_scalar("L2 Norm", l2_norm, epoch)
        summary_writer.add_scalar("Sigma Loss", sigma_loss, epoch)
        summary_writer.add_scalar("Smooth Rank", smooth_rank, epoch)
        summary_writer.add_scalar("Acc Loss", acc_loss, epoch)
        summary_writer.add_scalar("Acc MSE", acc_mse, epoch)
        summary_writer.add_scalar("Acc Smooth Rank", acc_smooth_rank, epoch)
        summary_writer.add_scalar("Acc Sigma Loss", acc_sigma_loss, epoch)
        summary_writer.add_scalar("Acc L2 Norm", acc_l2_norm, epoch)
        summary_writer.add_scalar("Sigma", model.sigma, epoch)

    csv_file.close()

if __name__ == "__main__":
    main(parse_arguments())
