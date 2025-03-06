#' Class contains the results of VGMSFH.
#'
#' @slot direct_estimate array, Direct estimate of the population parameters.
#' @slot yhat_samples array, The posterior samples of estimated population parameters.
#' @slot spatial_samples array, The posterior samples of estimated spatial random effects.
#' @slot beta_samples array, The posterior samples of coefficients of fixed effects.
#' @slot all_samples array, The posterior samples of all parameters in VGMSFH.
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

#' Run vgmsfh using numpyro.
#'
#' Run vgmsfh using numpyro as the backend.
#'
#' @param y Matrix, responses.
#' @param y_sigma Matrix, reported standard deviations.
#' @param X Matrix, covariance matrix.
#' @param W Matrix, proximity matrix
#' @param GEOID Vector, FIPS codes or other equivalent GEOIDs.
#' @param model_name String, vae model name.
#' @param save_dir String, vae model saving directory. Default to use pretrained models.
#' @param num_samples Int, Number of posterior samples. Default to 1000.
#' @param num_warmup Int, Number of burning-in. Default to 1000.
#' @return VGMSFH s4 object, which contains the direct estimate and the posterior samples of yhat (population process), mu (intercept of population process), beta (coefficient of covariates), delta (fine scale variations of population process), car (spatial random effects of population process). In addition, all other posteriors (bridging variables, variations of random effects) including all meantioned before are provided in "all_samples".
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
