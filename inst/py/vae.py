import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import trange


class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # self.enc_input = MaskedArealLayer(NB)
        self.enc_input = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec_input = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)
        # self.dec_out = MaskedArealLayer(NB)
        self.activate = nn.ELU()
        pass

    def encoder(self, x):
        h = self.activate(self.enc_input(x))
        z_mu = self.enc_mu(h)
        z_logvar = self.enc_logvar(h)
        return z_mu, z_logvar

    def decoder(self, z):
        h = self.activate(self.dec_input(z))
        x = self.dec_out(h)
        return x

    def forward(self, x):
        z_mu, z_logvar = self.encoder(x)
        x_sample = self.reparameterize(z_mu, torch.exp(0.5 * z_logvar))
        generated_x = self.decoder(x_sample)
        return generated_x, z_mu, z_logvar

    def reparameterize(self, z_mu, z_sd):
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        if self.training:
            # sample from the distribution having latent parameters z_mu, z_sd
            # reparameterize
            eps = torch.randn_like(z_sd)
            z = z_mu + z_sd * eps
            return z
        return z_mu

    def calculate_loss(self, x, reconstructed_x, z_mu, z_logvar, beta):
        # reconstruction loss
        RCL = torch.mean(torch.sum(F.mse_loss(reconstructed_x, x, reduction='none'), dim=1), dim=0)
        # kl divergence loss
        KLD = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0)
        return RCL + beta * KLD, RCL, KLD

    def fit_step(self, dataset_i, optimizer, beta, clip_value, device):
        target = dataset_i.float().to(device)
        optimizer.zero_grad()
        y_hat, z_mu, z_logvar = self.forward(target)
        loss, RCL, KLD = self.calculate_loss(target, y_hat, z_mu, z_logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
        optimizer.step()
        return loss, RCL, KLD

    def fit(self, dataloader, lr, lr_gamma, beta, epoch,
            clip_value=1.0, device='cpu', verbose=True):
        # train
        loss_list = []
        loss_RCL_list = []
        loss_KLD_list = []
        n_batch = len(dataloader)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        t = trange(epoch, dynamic_ncols=True, disable=not verbose)
        for e in t:
            beta = beta
            # set training mode
            self.train()
            total_loss = 0
            total_loss_RCL = 0
            total_loss_KLD = 0
            for i, dataset_i in enumerate(dataloader):
                loss, RCL, KLD = self.fit_step(dataset_i, optimizer, beta, clip_value, device)
                total_loss += loss.item()
                total_loss_RCL += RCL.item()
                total_loss_KLD += KLD.item()
                t.set_description(f'Train RCL: {RCL.item():.5}, Train KLD: {KLD.item():.5}, LR: {scheduler.get_last_lr()[0]:.2E}')
                pass
            scheduler.step()
            t.set_description(f'Train RCL: {total_loss_RCL/n_batch:.5}, Train KLD: {total_loss_KLD/n_batch:.5}')
            loss_list.append(total_loss/n_batch)
            loss_RCL_list.append(total_loss_RCL/n_batch)
            loss_KLD_list.append(total_loss_KLD/n_batch)
            pass
        return loss_list, loss_RCL_list, loss_KLD_list
