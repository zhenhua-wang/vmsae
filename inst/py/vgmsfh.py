import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS
from jax.extend import backend

def decoder_learned(z, W_in, B_in, W_out, B_out):
    h = jax.nn.elu(jnp.matmul(z, W_in) + B_in)
    x = jnp.matmul(h, W_out) + B_out
    return x

def regression(N, p_y, p_x):
    # mean
    mu = numpyro.sample('mu', dist.Normal(0, 100), sample_shape=(p_y, ))
    # covariate
    beta = numpyro.sample('beta', dist.Normal(0, 100), sample_shape=(p_x, p_y))
    return mu, beta

def fine_scale_variation(N, p_y):
    # normal random effect
    delta_sigma2 = numpyro.sample('delta_sigma2',
                                  dist.InverseGamma(0.001, 0.001),
                                  sample_shape=(p_y, ))
    delta = numpyro.sample('delta', dist.Normal(jnp.zeros(p_y),
                                                jnp.sqrt(delta_sigma2)),
                           sample_shape=(N, ))
    return delta

def spatial_effect(n_latent, N, p_y, W_in, B_in, W_out, B_out, W):
    # marginal
    z1 = numpyro.sample('z1', dist.Normal(jnp.zeros(n_latent),
                                          jnp.ones(n_latent)))
    phi1_latent = decoder_learned(z1, W_in, B_in, W_out, B_out)
    phi1_sigma2 = numpyro.sample('phi1_sigma2',
                                 dist.InverseGamma(0.001, 0.001))
    phi1 = jnp.sqrt(phi1_sigma2) * phi1_latent
    phi = phi1
    # conditional
    phii1 = phi1
    for i in range(2, p_y + 1):
        zi = numpyro.sample(f'z{i}', dist.Normal(jnp.zeros(n_latent),
                                                 jnp.ones(n_latent)))
        phii_latent = decoder_learned(zi, W_in, B_in, W_out, B_out)
        phii_sigma2 = numpyro.sample(f'phi{i}_sigma2',
                                     dist.InverseGamma(0.001, 0.001))
        phii_eta0 = numpyro.sample(f'phi{i}_eta0', dist.Normal(0, 100))
        phii_eta1 = numpyro.sample(f'phi{i}_eta1', dist.Normal(0, 100))
        A = phii_eta0*jnp.eye(N) + phii_eta1*W
        phii_given_phii1 = A @ phii1 + jnp.sqrt(phii_sigma2) * phii_latent
        phi = jnp.column_stack((phii_given_phii1, phi))
        phii1 = phii_given_phii1
        pass
    phi = numpyro.deterministic('phi', phi)
    return phi

def interpolation(N, Y, sigma_y, yhat):
    # incidence matrix
    miss_Y = np.isnan(Y)
    miss_sigma_y = np.isnan(sigma_y)
    miss = np.logical_or(miss_Y, miss_sigma_y)
    if np.isnan(miss).any():
        I = jnp.eye(N)
        H1 = I[~miss[:, 0]]
        H2 = I[~miss[:, 1]]
        # data model
        Y_vec = jnp.concatenate((Y[:, 0][~miss[:, 0]], Y[:, 1][~miss[:, 1]]))
        sigma_y_vec = jnp.concatenate((sigma_y[:, 0][~miss[:, 0]], sigma_y[:, 1][~miss[:, 1]]))
        yhat_vec = jnp.concatenate((H1 @ yhat[:, 0], H2 @ yhat[:, 1]))
    else:
        Y_vec = Y
        sigma_y_vec = sigma_y
        yhat_vec = yhat
        pass
    return Y_vec, sigma_y_vec, yhat_vec

def vgmsfh(args):
    # parameters
    n_latent = args['n_latent']
    Y = args['Y']
    sigma_y = args['sigma_y']
    W_in = args['W_in']
    B_in = args['B_in']
    W_out = args['W_out']
    B_out = args['B_out']
    W = args['W']
    X = args['X']
    N = Y.shape[0]
    p_y = Y.shape[1]
    p_x = X.shape[1]
    # spatial random effect
    phi = spatial_effect(n_latent, N, p_y, W_in, B_in, W_out, B_out, W)
    # fixed effect regression
    mu, beta = regression(N, p_y, p_x)
    fixed = jnp.matmul(X, beta)
    # fine scale variation
    delta = fine_scale_variation(N, p_y)
    # process model
    # reshape for univariate case
    if p_y == 1:
        mu = np.repeat(mu, N).reshape(-1, 1)
        phi = phi.reshape(-1, 1)
        delta = delta.reshape(-1, 1)
        pass
    yhat = numpyro.deterministic('y_hat', mu + fixed + phi + delta)
    # imputation
    Y_vec, sigma_y_vec, yhat_vec = interpolation(N, Y, sigma_y, yhat)
    # data model
    Y_vec = numpyro.sample("Y", dist.Normal(yhat_vec, sigma_y_vec), obs=Y_vec)
    pass

def run_vgmsfh(p_y, y, y_sigma, X, W,
               W1, B1, W2, B2,
               num_samples, num_warmup, verbose=True, use_gpu=False):
    device = 'cpu' if not use_gpu else backend.get_backend().platform
    numpyro.set_platform(device)
    if verbose:
        if use_gpu and device == 'cpu':
            print("GPU is not available. VGMSFH is trained on CPU instead.")
        else:
            print(f"VGMSFH is trained on {device}.")
            pass
        print("To use a different device, please restart R to reload the Python environment")
        pass
    num_samples, num_warmup = int(num_samples), int(num_warmup)
    latent_dim = W1.shape[0]
    # check univariate case
    y = np.reshape(y, (-1, p_y))
    y_sigma = np.reshape(y_sigma, (-1, p_y))
    mcmc_data = {
        "n_latent": latent_dim,
        "Y": y,
        "sigma_y": y_sigma,
        'W_in': W1,
        'B_in': jnp.array(B1),
        'W_out': W2,
        'B_out': jnp.array(B2),
        'W': W,
        'X': X
    }
    kernel = NUTS(vgmsfh, find_heuristic_step_size=True)
    mcmc_vgmsfh = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc_vgmsfh.run(jax.random.PRNGKey(0), mcmc_data)
    priorvae_samples = {key: np.array(value) for key, value in mcmc_vgmsfh.get_samples().items()}
    return priorvae_samples
