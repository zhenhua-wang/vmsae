import torch
from torch.utils.data import DataLoader
from vae import VAE
from car_dataset import CARDataset, generate_CAR_dataset

def train_vae(W, save_path,
              n_samples, batch_size,
              epoch, lr_init, lr_min, verbose=True, use_gpu=True):
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    if verbose:
        if use_gpu and device == 'cpu':
            print("GPU is not available. VAE is trained on CPU instead.")
        else:
            print(f"VAE is trained on {device}.")
            pass
        pass
    lr_gamma = pow(lr_min * (1/lr_init), 1/epoch)
    in_locations = W.shape[0]
    hidden_dim = in_locations
    latent_dim = in_locations
    W = torch.from_numpy(W.copy())
    d = W.sum(axis=0)
    D = torch.diag(d)
    n_samples = int(n_samples)
    batch_size = int(batch_size)
    epoch = int(epoch)
    # training data
    dataset_CAR = generate_CAR_dataset(
        n_samples, D.to(device), W.to(device), device, verbose).cpu()
    train_dataset = CARDataset(dataset_CAR)
    train_dl = DataLoader(train_dataset,
                          batch_size=batch_size,
                          drop_last=True,
                          shuffle=True)
    # vae
    model = VAE(input_dim=in_locations,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim).to(device)
    loss, RCL, KLD = model.fit(dataloader=train_dl,
                               lr=lr_init,
                               lr_gamma=lr_gamma,
                               epoch=epoch,
                               beta=1/latent_dim,
                               clip_value=1,
                               device=device,
                               verbose=verbose)
    torch.save(model.state_dict(), save_path)
    return loss, RCL, KLD
