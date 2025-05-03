#' VGMSFH S4 Class
#'
#' An S4 class to store results from the Variational Gaussian Markov Small Area Estimation with Fay-Herriot model (VGMSFH). This class holds the posterior samples for various model components as well as the original direct estimates.
#'
#' @slot model_name Character. The name of the trained VAE model.
#' @slot direct_estimate Array. Direct estimates of parameters.
#' @slot yhat_samples Array. Posterior samples of the estimated parameters.
#' @slot spatial_samples Array. Posterior samples of the estimated spatial random effects.
#' @slot beta_samples Array. Posterior samples of the fixed effect coefficients.
#' @slot all_samples List. Posterior samples of all parameters in the VGMSFH model.
#'
#' @export
setClass("VGMSFH",
  slots = c(
    model_name = "character",
    direct_estimate = "array",
    yhat_samples = "array",
    spatial_samples = "array",
    beta_samples = "array",
    all_samples = "list"
  )
)

#' Run VGMSFH Using NumPyro
#'
#' This function runs the Variational Generalized Multivariate Spatil Fay-Herriot model (VGMSFH) using NumPyro as the inference backend. It loads pretrained VAE decoder weights, prepares the data, and performs posterior sampling.
#'
#' @param y Matrix. Response variables (direct estimates).
#' @param y_sigma Matrix. Reported standard deviations of the responses.
#' @param X Matrix. Covariate matrix.
#' @param W Matrix. Proximity or adjacency matrix defining spatial structure.
#' @param GEOID Character vector. FIPS codes or other region identifiers used to match with the pretrained VAE model.
#' @param model_name Character. The name of the pretrained VAE model.
#' @param save_dir Character. The directory where the VAE model is stored. If \code{NULL}, a default pretrained model directory is used.
#' @param num_samples Integer. Number of posterior samples to draw. Default is 1000.
#' @param num_warmup Integer. Number of warmup (burn-in) iterations. Default is 1000.
#'
#' @return An object of class \code{VGMSFH}, which contains:
#' \itemize{
#'   \item \code{direct_estimate}: the observed response data,
#'   \item \code{yhat_samples}: posterior samples of the latent population process,
#'   \item \code{spatial_samples}: posterior samples of spatial random effects (CAR),
#'   \item \code{beta_samples}: posterior samples of fixed effect coefficients,
#'   \item \code{all_samples}: a list containing all sampled parameters, including \code{mu}, \code{delta}, and other intermediate quantities.
#' }
#'
#' @details
#' This function uses a pretrained VAE decoder to parameterize the CAR prior and enables scalable inference through NumPyro. It is suitable for both univariate and multivariate response modeling in spatial domains.
#'
#' @references
#' Wang, Z., Parker, P. A., & Holan, S. H. (2025). Variational Autoencoded Multivariate Spatial Fay-Herriot Models. arXiv:2503.14710. \url{https://arxiv.org/abs/2503.14710}
#'
#' @examples
#' \donttest{
#' library(sf)
#' library(vmsae)
#' install_environment()
#' load_environment()
#'
#' acs_data <- read_sf(system.file("example", "mo_county.shp", package = "vmsae"))
#' y <- readRDS(system.file("example", "y.Rds", package = "vmsae"))
#' y_sigma <- readRDS(system.file("example", "y_sigma.Rds", package = "vmsae"))
#' X <- readRDS(system.file("example", "X.Rds", package = "vmsae"))
#' W <- readRDS(system.file("example", "W.Rds", package = "vmsae"))
#'
#' num_samples <- 1000 # set to larger values in practice, e.g. 10000.
#' model <- vgmsfh_numpyro(y, y_sigma, X, W,
#'   GEOID = acs_data$GEOID,
#'   model_name = "mo_county", save_dir = NULL,
#'   num_samples = num_samples, num_warmup = num_samples)
#' y_hat_np <- model@yhat_samples
#' y_hat_mean_np <- apply(y_hat_np, c(2, 3), mean)
#' y_hat_lower_np <- apply(y_hat_np, c(2, 3), quantile, 0.025)
#' y_hat_upper_np <- apply(y_hat_np, c(2, 3), quantile, 0.975)
#'
#' plot(model, shp = acs_data, type = "compare", var_idx = 2)
#' }
#'
#' @importFrom reticulate py
#' @importFrom methods new
#'
#' @export
vgmsfh_numpyro <- function(y, y_sigma, X, W, GEOID,
                           model_name, save_dir = NULL,
                           num_warmup = 1000, num_samples = 1000) {
  vae_weights <- load_vae(model_name, save_dir)
  W_in <- vae_weights@W_in
  B_in <- vae_weights@B_in
  W_out <- vae_weights@W_out
  B_out <- vae_weights@B_out
  GEOID_vae <- vae_weights@GEOID
  ## check univariate case
  if (is.null(dim(y)) && is.null(dim(y_sigma))) {
    y <- matrix(y, ncol = 1)
    y_sigma <- matrix(y_sigma, ncol = 1)
  }
  p_y <- dim(y)[2]
  data <- sort_data(y, y_sigma, X, W, GEOID, GEOID_vae)
  samples <- py$run_vgmcar(
    p_y, data$y, data$y_sigma, data$X, data$W, W_in, B_in, W_out, B_out,
    num_samples, num_warmup)
  vgmsfh <- new("VGMSFH",
    model_name = model_name,
    direct_estimate = y,
    yhat_samples = samples$y_hat,
    spatial_samples = samples$car,
    beta_samples = samples$beta,
    all_samples = samples)
  return(vgmsfh)
}

sort_data <- function(y, y_sigma, X, W, GEOID, GEOID_vae) {
  idx <- match(GEOID, GEOID_vae)
  if (any(is.na(idx))) {
    stop("There are mismatches between GEOID and GEOID_vae.")
  }
  y <- y[idx, ]
  y_sigma <- y_sigma[idx, ]
  X <- X[idx, ]
  W <- W[idx, idx]
  return(list(y = y, y_sigma = y_sigma, X = X, W = W))
}
