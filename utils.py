
import torch

def hermitian_logdet_and_inverse(matrix):
    # Perform Cholesky decomposition
    L = torch.linalg.cholesky(matrix)

    # Compute logdet using the property of Cholesky decomposition
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))

    # Compute the inverse using the property of Cholesky decomposition
    L_inv = torch.inverse(L)
    inverse = L_inv.t().conj() @ L_inv

    return logdet, inverse