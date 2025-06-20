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
    car1_latent = decoder_learned(z1, W_in, B_in, W_out, B_out)
    car1_sigma2 = numpyro.sample('car1_sigma2',
                                 dist.InverseGamma(0.001, 0.001))
    car1 = jnp.sqrt(car1_sigma2) * car1_latent
    car = car1
    # conditional
    cari1 = car1
    for i in range(2, p_y + 1):
        zi = numpyro.sample(f'z{i}', dist.Normal(jnp.zeros(n_latent),
                                                 jnp.ones(n_latent)))
        cari_latent = decoder_learned(zi, W_in, B_in, W_out, B_out)
        cari_sigma2 = numpyro.sample(f'car{i}_sigma2',
                                     dist.InverseGamma(0.001, 0.001))
        cari_eta0 = numpyro.sample(f'car{i}_eta0', dist.Normal(0, 100))
        cari_eta1 = numpyro.sample(f'car{i}_eta1', dist.Normal(0, 100))
        A = cari_eta0*jnp.eye(N) + cari_eta1*W
        cari_given_cari1 = A @ cari1 + jnp.sqrt(cari_sigma2) * cari_latent
        car = jnp.column_stack((cari_given_cari1, car))
        cari1 = cari_given_cari1
        pass
    car = numpyro.deterministic('car', car)
    return car

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

def vgmcar(args):
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
    car = spatial_effect(n_latent, N, p_y, W_in, B_in, W_out, B_out, W)
    # fixed effect regression
    mu, beta = regression(N, p_y, p_x)
    fixed = jnp.matmul(X, beta)
    # fine scale variation
    delta = fine_scale_variation(N, p_y)
    # process model
    # reshape for univariate case
    if p_y == 1:
        mu = np.repeat(mu, N).reshape(-1, 1)
        car = car.reshape(-1, 1)
        delta = delta.reshape(-1, 1)
        pass
    yhat = numpyro.deterministic('y_hat', mu + fixed + car + delta)
    # imputation
    Y_vec, sigma_y_vec, yhat_vec = interpolation(N, Y, sigma_y, yhat)
    # data model
    Y_vec = numpyro.sample("Y", dist.Normal(yhat_vec, sigma_y_vec), obs=Y_vec)
    pass

def run_vgmcar(p_y, y, y_sigma, X, W,
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
    kernel = NUTS(vgmcar, find_heuristic_step_size=True)
    mcmc_car = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc_car.run(jax.random.PRNGKey(0), mcmc_data)
    priorvae_samples = {key: np.array(value) for key, value in mcmc_car.get_samples().items()}
    return priorvae_samples
