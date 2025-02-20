from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal, Uniform
import torch
from tqdm import trange

def generate_CAR_dataset(n_samples, D, W, device="cpu"):
    n = W.shape[0]
    Z = torch.randn(n_samples, n).to(device)
    tbar = trange(n_samples)
    for fct_id in tbar:
        phi = Uniform(0, 0.99).sample().to(device)
        prec = D - phi * W
        CAR_cho, _ = torch.linalg.cholesky_ex(prec)
        Z[fct_id] = torch.linalg.solve_triangular(
            CAR_cho, Z[fct_id].reshape(-1, 1), upper=False).reshape(-1)
        tbar.set_description(f'Draw training data')
        pass
    return Z

class CARDataset(Dataset):
    def __init__(self, data):
        'Initialization'
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

    def __getitem__(self, i):
        'Generates one sample of data'
        data_i = self.data[i, ]
        return data_i
