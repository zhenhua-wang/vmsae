setClass("VAE",
  slots = c(
    GEOID = "character",
    W_in = "array",
    B_in = "array",
    W_out = "array",
    B_out = "array"
  )
)

setClass("VGMSFH",
  slots = c(
    yhat_samples = "array",
    mu_samples = "array",
    beta_samples = "array",
    delta_samples = "array",
    car_samples = "array",
    all_samples = "list"
  )
)

#' Run vgmsfh using numpyro.
#'
#' This function runs vgmsfh using numpyro as the backend.
#'
#' @param y Matrix, responses.
#' @param y_sigma Matrix, reported standard deviations.
#' @param X Matrix, covariance matrix.
#' @param W Matrix, proximity matrix
#' @param GEOID Vector, FIPS codes or other equivalent GEOIDs.
#' @param vae_model_name String, vae model name.
#' @param vae_save_dir String, vae model saving directory.
#' @param num_samples Int, Number of posterior samples.
#' @param num_warmup Int, Number of burning-in.
#' @return Matrix, the true process.
#' @export
vgmsfh_numpyro <- function(y, y_sigma, X, W, GEOID,
                           vae_model_name, vae_save_dir = NULL,
                           num_warmup = 1000, num_samples = 1000) {
  vae_weights <- load_vae(vae_model_name, vae_save_dir)
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
    yhat_samples = samples$y_hat,
    mu_samples = samples$mu,
    beta_samples = samples$beta,
    delta_samples = samples$delta,
    car_samples = samples$car,
    all_samples = samples)
  return(vgmsfh)
}

load_vae <- function(vae_model_name, vae_save_dir = NULL) {
  vae_model <- NULL
  GEOID <- NULL
  GEOID_name <- paste0(tolower(vae_model_name), ".GEOID")
  vae_full_name <- paste0(tolower(vae_model_name), ".model")
  if (is.null(vae_save_dir)) {
    vae_model <- py$torch$load(
      system.file("model", vae_full_name, package = "vmsae"),
      weights_only = TRUE)
    GEOID <- read.table(
      system.file("model", GEOID_name, package = "vmsae"),
      header = FALSE)
  } else {
    vae_model <- py$torch$load(
      file.path(vae_save_dir, vae_full_name),
      weights_only = TRUE)
    GEOID <- read.table(
      file.path(vae_save_dir, GEOID_name),
      header = FALSE)
  }
  W_in <- vae_model$dec_input.weight$numpy()
  B_in <- vae_model$dec_input.bias$numpy()
  W_out <- vae_model$dec_out.weight$numpy()
  B_out <- vae_model$dec_out.bias$numpy()
  vae_weights <- new("VAE",
    GEOID = as.character(GEOID$V1),
    W_in = W_in, B_in = B_in,
    W_out = W_out, B_out = B_out)
  return(vae_weights)
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

#' Summarize VGMSFH result.
#'
#' This function display the summary of VGMSFH result.
#'
#' @return data frame, summary of VGMSFH result.
#' @export
summary.VGMSFH <- function(object, var_idx = 1, field = "beta") {
  var_mean <- apply(slot(object, field)[, , var_idx], 2, mean)
  var_summary <- confint(object, var_idx, field)
  var_summary["mean"] <- var_mean
  var_summary <- var_summary[, c("mean", "lower", "upper")]
  return(var_summary)
}

#' Plot VGMSFH result.
#'
#' This function plot the spatial map of VGMSFH result.
#'
#' @param object An object containing the VGMSFH results.
#' @param var_idx Index of the variable of interest.
#' @param field The field from the object to plot.
#' @export
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 geom_sf
#' @importFrom ggplot2 theme_minimal
#' @importFrom ggplot2 scale_fill_viridis_c
#' @importFrom gridExtra grid.arrange
plot.VGMSFH <- function(object, shp, var_idx = 1, field = "yhat_samples", ...) {
  var <- slot(object, field)[, , var_idx]
  shp["mean"] <- apply(var, 2, mean)
  shp["std"] <- apply(var, 2, sd)
  shp["lower"] <- apply(var, 2, quantile, 0.025)
  shp["upper"] <- apply(var, 2, quantile, 0.975)
  range_min <- min(shp$mean, shp$lower, shp$upper, na.rm = TRUE)
  range_max <- max(shp$mean, shp$lower, shp$upper, na.rm = TRUE)
  breaks_mean <- round(
    seq(floor(min(shp$lower)), ceiling(max(shp$upper)),
      length.out = 8), 2)
  ## plot(shp[c("mean", "std", "lower", "upper")], key.pos = 4, ...)
  p1 <- ggplot() +
    geom_sf(data = shp, aes(fill = mean)) +
    scale_fill_viridis_c(name = "Mean",
      breaks = breaks_mean,
      limits = c(range_min, range_max),
      option = "D") +
    theme_minimal()
  p2 <- ggplot() +
    geom_sf(data = shp, aes(fill = std)) +
    scale_fill_viridis_c(name = "Std. Dev.",
      option = "C") +
    theme_minimal()
  p3 <- ggplot() +
    geom_sf(data = shp, aes(fill = lower)) +
    scale_fill_viridis_c(name = "Lower Bound (2.5%)",
      breaks = breaks_mean,
      limits = c(range_min, range_max),
      option = "D") +
    theme_minimal()
  p4 <- ggplot() +
    geom_sf(data = shp, aes(fill = upper)) +
    scale_fill_viridis_c(name = "Upper Bound (97.5%)",
      breaks = breaks_mean,
      limits = c(range_min, range_max),
      option = "D") +
    theme_minimal()
  grid.arrange(p1, p2, p3, p4, nrow = 2)
}

#' @export
coef.VGMSFH <- function(object, var_idx = 1, type = "fixed") {
  if (type == "fixed") {
    var <- object@beta[, , var_idx]
  } else if (type == "spatial") {
    var <- object@car[, , var_idx]
  } else if (type == "fine-scale") {
    var <- object@delta[, , var_idx]
  }
  var_mean <- apply(var, 2, mean)
  return(var_mean)
}

#' @export
confint.VGMSFH <- function(object, var_idx = 1, field = "yhat_samples") {
  var_lower <- apply(slot(object, field)[, , var_idx], 2, quantile, 0.025)
  var_upper <- apply(slot(object, field)[, , var_idx], 2, quantile, 0.975)
  var_confint <- data.frame(lower = var_lower, upper = var_upper)
  return(var_confint)
}
