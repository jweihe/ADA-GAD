import torch
import torch_geometric.utils as pyg_utils

def compute_E_high(adj_matrix, feat_matrix):
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    feat_tensor = feat_matrix.clone().detach().to(dtype=torch.float32)

    deg_tensor = torch.sum(adj_tensor, dim=1)
    deg_matrix = torch.diag(deg_tensor)

    laplacian_tensor = deg_matrix - adj_tensor
    numerator = torch.matmul(torch.matmul(feat_tensor.T, laplacian_tensor), feat_tensor)
    denominator = torch.matmul(feat_tensor.T, feat_tensor)
    
    S_high = torch.sum(numerator) / torch.sum(denominator)

    return S_high.item()

def compute_G_ano(adj_matrix, feat_matrix):
    a_high = compute_E_high(adj_matrix, feat_matrix)
    deg_matrix = torch.diag(torch.sum(torch.tensor(adj_matrix, dtype=torch.float32), dim=1))
    s_high = compute_E_high(adj_matrix, deg_matrix)

    return a_high, s_high
