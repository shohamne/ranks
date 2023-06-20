import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import uci_datasets 
from kernels import GaussianKernel, LinearKernel


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
    parser.add_argument("--epochs", type=int, default=1500, help="Number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--sigma0", type=float, default=1.0, help="Initial sigma")
    parser.add_argument("--logdet_coeff", type=float, default=-1, help="DEBUG Coefficient for log det regular")
    parser.add_argument("--l2_coeff", type=float, default=0.0, help="DEBUG Coefficient for l2 regular")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on - 'cpu' or 'cuda'")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--evaluation_batch_size", type=int, default=2000, help="Batch size for evaluation")
    parser.add_argument("--csv_name", type=str, default="metrics.csv", help="Name of the CSV file to save metrics")
    parser.add_argument("--alpha", type=float, default=0.1, help="Decay rate for running covariance matrix")
    parser.add_argument("--start_layer", type=int, default=5, help="First layer to regulize it's input")
    parser.add_argument("--stop_rank_reg", type=int, default=1e32, help="Epoch to stop rank regularization")
    parser.add_argument("--kernel", type=str, default="linear", choices=["rbf", "linear", "none"], help="Kernel to use for training")
    parser.add_argument("--split", type=int, default=0, help="Split of the dataset to use for training")
    parser.add_argument("--dimension_metric", type=str, default="log_det", choices=["smooth_rank", "log_det"], help="Dimension metric to compute")
    parser.add_argument('--optimize_activations', type=str2bool, default=True, help='Optimize activations if this argument is set to yes')
    parser.add_argument('--optimize_rank', type=str2bool, default=True, help='Optimize rank if this argument is set to yes')
    parser.add_argument('--optimize_sigma', type=str2bool, default=False, help='Optimize sigma if this argument is set to yes')
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="Optimizer to use for training")
    return parser.parse_args()



def main(args):
    device = torch.device(args.device)

    # Create a summary writer
    summary_writer = SummaryWriter("logs")

    # Load UCI regression dataset
    data = uci_datasets.Dataset(args.dataset)
    x_train, y_train, x_test, y_test = data.get_split(split=args.split)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Compute mean and variance of x_train
    mean = torch.mean(torch.tensor(x_train, dtype=torch.float32), dim=0)
    variance = torch.var(torch.tensor(x_train, dtype=torch.float32), dim=0)

    # Normalize x_train and x_test
    x_train = (x_train - mean) / torch.sqrt(variance)
    x_test = (x_test - mean) / torch.sqrt(variance)

    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_data, batch_size=args.evaluation_batch_size, shuffle=False)

    train_eval_loader = DataLoader(train_data, batch_size=args.evaluation_batch_size, shuffle=False)

    class DeepNetwork(nn.Module):
        def __init__(self, input_size, output_size, depth, hidden_size):
            super(DeepNetwork, self).__init__()
            self.layers = nn.ModuleList(
                    [nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.ReLU())] +
                    [nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.ReLU()) for _ in range(depth - 2)] +
                    [nn.Sequential(nn.Linear(hidden_size, output_size, bias=False))]
            )
            self.sigma = torch.nn.Parameter(torch.tensor(args.sigma0), requires_grad=args.optimize_sigma)


            self.apply(self._init_weights)


        @property
        def sigma_square(self):
            return self.sigma.pow(2)
        

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    input_size = x_train.shape[1]
    output_size = 1

    model = DeepNetwork(input_size, output_size, args.depth, hidden_size=args.hidden_size).to(device)
    criterion = nn.MSELoss(reduction='sum')

    kernel = GaussianKernel(torch.tensor(1.0), torch.tensor(1.0)) if args.kernel == "rbf" else LinearKernel()

    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    def compute_M_and_evaluate(loader, train_features=None):
        M_per_layer = [0] * len(model.layers)
        mse = 0.0
        total_samples = len(loader.dataset)
        smooth_rank_per_layer = []
        l2_norm = 0.0
        sigma_loss = 0.0
        features_list = []
        y_list = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                activations = x_batch
                for l, layer in enumerate(model.layers):
                    M = activations.T @ activations
                    M_per_layer[l] += M
                    features = activations
                    activations = layer(activations)
                y_list.append(y_batch)
                features_list.append(features)
                mse += criterion(activations, y_batch) 
            for l, layer in enumerate(model.layers):
                smooth_rank = compute_smooth_rank(M_per_layer[l], model.sigma_square) if args.dimension_metric == "smooth_rank" \
                                else compute_log_det(M_per_layer[l], model.sigma_square)
                #print('$', l, smooth_rank, M_per_layer[l][0][0], x_batch[0][0], len(loader.dataset))

                # contains_nan = torch.isnan(smooth_rank).any()
                # assert not contains_nan, "NaN in the expirement aborting"
                smooth_rank_per_layer.append(smooth_rank)
                l2_norm += layer[0].weight.pow(2).sum()
                if args.dimension_metric == "log_det":
                    sigma_loss += (total_samples-len(M_per_layer[l]))*torch.log(model.sigma_square)
            avg_l2_norm = l2_norm / total_samples
            avg_sigma_loss = sigma_loss / total_samples
            avg_mse = mse / total_samples
            smooth_rank = sum(smooth_rank_per_layer[4:])
            avg_smooth_rank = torch.tensor(smooth_rank)/total_samples
            avg_smooth_rank_per_layer = torch.tensor(smooth_rank_per_layer)/total_samples
            if args.logdet_coeff <0:
                avg_loss = avg_mse/model.sigma_square + avg_l2_norm + avg_sigma_loss + avg_smooth_rank
            else:
                avg_loss = avg_mse + args.l2_coeff*l2_norm + args.logdet_coeff*avg_smooth_rank
            
            features = torch.cat(features_list, dim=0)
            y = torch.cat(y_list, dim=0)
            if train_features is None:
                train_features = features
            K = kernel(train_features) + model.sigma_square*torch.eye(len(train_features), device=device)
            Kinv = K.inverse()
            gp_loss = (y.T @ Kinv @ y + compute_log_det(K, 0))/total_samples
            y_hat = kernel(features, train_features) @ Kinv @ y
            gp_mse = criterion(y_hat, y)/total_samples

        return [m.cpu().item() for m in avg_smooth_rank_per_layer], avg_smooth_rank.cpu().item(), avg_loss.cpu().item(), avg_mse.cpu().item(), \
            avg_l2_norm.cpu().item(), avg_sigma_loss.cpu().item(), gp_loss.cpu().item(), gp_mse.cpu().item(), features
    
    def compute_smooth_rank(M, lam):
        eigs = torch.linalg.eigvalsh(M)
        return (eigs / (eigs + lam)).sum()

    def compute_log_det(M, sigma_square):
        #return torch.logdet(M + lam*torch.eye(M.shape[0], device=device))
        return torch.linalg.eigvalsh(M + sigma_square*torch.eye(M.shape[0], device=device)).log().sum()
        
    # Create CSV file
    csv_file = open(args.csv_name, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["Epoch", "Model Sigma", "Train Loss", "Train GP Loss", "Train MSE", "Test MSE", "Train GP MSE", "Test GP MSE", "L2 Norm", "Sigma Loss"]
    for l in range(args.depth):
        header.extend([f"Layer {l} Smooth Rank"])
    csv_writer.writerow(header)

    F_per_layer = [torch.zeros(1, device=device) for _ in range(args.depth)]
    K_per_layer = [torch.zeros(1, device=device) for _ in range(args.depth)]

    wl = 0.0
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
            dataset_length = len(x_train)

            for l, layer in enumerate(model.layers):
                batch_size = len(x_batch)
                if l >= args.start_layer:   
                    with torch.enable_grad() if l >= args.start_layer else torch.no_grad():
                        F = (activations.T @ activations)*(dataset_length/batch_size)
                    # (1-bt)Ft+ (bt*n/|S|)*sum(Fi)    
                    F_per_layer[l] = (1-args.alpha) * F_per_layer[l].detach() + args.alpha * F
                    with torch.enable_grad() if args.optimize_rank else torch.no_grad():                    
                        # layer_smooth_rank = compute_smooth_rank(F_per_layer[l], model.sigma_square) if args.dimension_metric == "smooth_rank" \
                        #             else compute_log_det(F_per_layer[l], model.sigma_square)
                        layer_smooth_rank = compute_log_det(F_per_layer[l], model.sigma_square)
                        sigma_loss  += (batch_size-len(F_per_layer[l]))*torch.log(model.sigma_square)

                else:
                    layer_smooth_rank = torch.tensor(0.0, device=device)
            
                smooth_rank += layer_smooth_rank
                features = activations
                activations = layer(activations)
                l2_norm += layer[0].weight.pow(2).sum() 

                if l>= args.start_layer:
                    wl = layer[0].weight.pow(2).sum() 
                else:
                    l2_norm += layer[0].weight.pow(2).sum() 

                   
                #l2_norm += layer[0].weight.pow(2).sum() * (args.l2_coeff if l < args.start_layer else model.sigma_square)
                
            output = activations
            mse = criterion(output, y_batch)

            # case 1 - baseline witout witout wight decay
            # case 2 - neta algorithm

            s_div_n = (batch_size/dataset_length) 

            if args.kernel != "none":
                K = kernel(features) + model.sigma_square*torch.eye(len(x_batch), device=device)
                loss = y_batch.T @ K.inverse() @ y_batch + compute_log_det(K, 0)                
                loss /= len(x_batch)
            else:
                if args.logdet_coeff < 0:
                    #loss =(ZW(L)-y)/sigma2  + logdet(Z*Z+sigma2I) + |W(L)| + (n-d)log(sigma^2)
                    loss = mse/model.sigma_square  + smooth_rank        + wl*s_div_n    + sigma_loss* s_div_n
                    # loss = mse/model.sigma_square  + smooth_rank        + wl     + sigma_loss +  args.l2_coeff*l2_norm 
                    #loss = mse/model.sigma_square + args.l2_coeff*l2_norm  
                    #loss = mse/model.sigma_square + wl + sigma_loss + smooth_rank
                    loss /= len(x_batch)
                    # print(f"loss={loss}")
                else:
                    assert args.start_layer >= args.depth, "start layer must be greater than depth because neta has hiddin auumption"
                    loss = mse/len(x_batch) + args.l2_coeff*l2_norm + args.logdet_coeff*smooth_rank/len(x_batch)
            
            contains_nan = torch.isnan(loss)
            assert not contains_nan, "NaN in the loss expirement aborting"
            # assert(args.optimize_sigma == False)
            loss.backward()

            acc_loss = loss.item()/len(x_batch)*0.1 + acc_loss*0.9 if acc_loss is not None else loss.item()/len(x_batch)
            acc_mse = mse.item()/len(x_batch)*0.1 + acc_mse*0.9 if acc_mse is not None else mse.item()/len(x_batch)
            acc_smooth_rank = smooth_rank.item()/len(x_batch)*0.1 + acc_smooth_rank*0.9 if acc_smooth_rank is not None else smooth_rank.item()/len(x_batch)
            acc_sigma_loss = sigma_loss.item()/len(x_batch)*0.1 + acc_sigma_loss*0.9 if acc_sigma_loss is not None else sigma_loss.item()/len(x_batch)
            acc_l2_norm = l2_norm.item()/len(x_batch)*0.1 + acc_l2_norm*0.9 if acc_l2_norm is not None else l2_norm.item()/len(x_batch)
            
            optimizer.step()

        #acc_mse /= len(train_loader)
        smooth_rank_per_layer, smooth_rank, train_loss, train_mse, l2_norm, sigma_loss, \
            train_gp_loss, train_gp_mse, train_features = compute_M_and_evaluate(train_loader)
        _, _, _, test_mse, _, _, _, test_gp_mse, _ = compute_M_and_evaluate(test_loader, train_features)

        row = [epoch, model.sigma_square.item(), train_loss, train_gp_loss, train_mse, train_gp_mse, test_mse, train_gp_mse, test_gp_mse, l2_norm, sigma_loss]
        for l in range(args.depth):
            row.extend([smooth_rank_per_layer[l]])
        csv_writer.writerow(row)

        print(f"{args.dataset:<20} L{args.start_layer} S{args.split} Epoch {epoch}: Train Loss={train_loss:.4f}, Train GP MSE={train_gp_mse:.4f}, Test GP MSE={test_gp_mse:.4f}, Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}, L2 Norm={l2_norm:.4f}, "
            f"Sigma Loss={sigma_loss:.4f}, Smooth Rank={smooth_rank:.4f}, MSE diff={test_mse-train_mse:.4f}, MSE div={test_mse/train_mse:.4f}, "
            #Acc Loss={acc_loss:.4f}, Acc MSE={acc_mse:.4f}, "
            #f"Acc Smooth Rank={acc_smooth_rank:.4f}, Acc Sigma Loss={acc_sigma_loss:.4f}, Acc L2 Norm={acc_l2_norm:.4f}, "
            f"Sigma={model.sigma_square:.4f}")


    csv_file.close()

if __name__ == "__main__":
    main(parse_arguments())
