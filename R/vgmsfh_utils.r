#' Plot VGMSFH Result
#'
#' This method plots spatial summaries of results from a \code{VGMSFH} object, including model estimates and comparisons with direct estimates.
#'
#' @param x An object of class \code{VGMSFH}, containing posterior samples and direct estimates from the model.
#' @param shp An \code{sf} object representing the spatial shapefile. If \code{NULL}, the function will automatically download a shapefile associated with the pretrained model.
#' @param var_idx Integer. The index of the variable of interest (for multivariate models).
#' @param type Character. The type of plot to generate. Options are:
#' \itemize{
#'   \item \code{"compare"} – compare direct estimates and model-based estimates.
#'   \item \code{"estimate"} – show the posterior mean and standard deviation of the model estimate.
#' }
#' @param verbose Logical; if \code{TRUE} (default), prints error messages.
#'
#' @details
#' The function provides spatial visualization of model results. It supports both univariate and multivariate response settings. When \code{type = "compare"}, it generates side-by-side choropleth maps for the direct and model-based estimates. When \code{type = "estimate"}, it plots the posterior mean and standard deviation of the VGMSFH model output.
#'
#' If no shapefile is provided, a default geometry is loaded from the pretrained repository.
#'
#' @return A \code{ggplot} object. The plot is rendered to the active device.
#'
#' @examples
#' library(vmsae)
#' library(sf)
#' example_model <- readRDS(system.file("example", "example_model.Rds", package = "vmsae"))
#' example_shp <- read_sf(system.file("example", "mo_county.shp", package = "vmsae"))
#' plot(example_model, shp = example_shp, type = "compare")
#' plot(example_model, shp = example_shp, type = "estimate", var_idx = 2)
#'
#' @export
setMethod("plot", "VGMSFH",
  function(x, shp = NULL, var_idx = 1, type = "compare", verbose = TRUE) {
    if (is.null(shp)) {
      ## download shapefile from pretrained model
      shp <- load_pretrained_shapefile(x@model_name)
    }
    if (type == "estimate") {
      plot_estimate(x, shp, var_idx)
    } else if(type == "compare") {
      plot_compare(x, shp, var_idx)
    } else {
      if(verbose){
        warning("Plot type not supported.")
      }
    }
  })

#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 geom_sf
#' @importFrom ggplot2 scale_fill_viridis_c
#' @importFrom tidyr pivot_longer
#' @importFrom gridExtra grid.arrange
#' @importFrom methods slot
#' @importFrom stats sd
#' @importFrom rlang .data
plot_estimate <- function(object, shp, var_idx) {
  var <- ith_data(slot(object, "yhat_samples"), var_idx)
  shp["mean"] <- apply(var, 2, mean)
  shp["std"] <- apply(var, 2, sd)
  p1 <- ggplot() +
    geom_sf(data = shp, aes(fill = .data[["mean"]])) +
    scale_fill_viridis_c(name = "Mean",
      option = "D")
  p2 <- ggplot() +
    geom_sf(data = shp, aes(fill = .data[["std"]])) +
    scale_fill_viridis_c(name = "Std. Dev.",
      option = "C")
  grid.arrange(p1, p2, nrow = 1)
}

#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 geom_sf
#' @importFrom ggplot2 facet_wrap
#' @importFrom ggplot2 scale_fill_viridis_c
#' @importFrom dplyr %>%
#' @importFrom tidyr pivot_longer
#' @importFrom methods slot
#' @importFrom stats quantile
#' @importFrom rlang .data
plot_compare <- function(object, shp, var_idx) {
  var <- ith_data(slot(object, "yhat_samples"), var_idx)
  shp["direct estimate"] <- ith_data(slot(object, "direct_estimate"), var_idx)
  shp["vgmsfh mean"] <- apply(var, 2, mean)
  shp["vgmsfh lower"] <- apply(var, 2, quantile, 0.025)
  shp["vgmsfh upper"] <- apply(var, 2, quantile, 0.975)
  cols <- c("direct estimate", "vgmsfh mean", "vgmsfh lower", "vgmsfh upper")
  shp <- shp %>%
    pivot_longer(
      cols = cols,
      names_to = "type",
      values_to = "value")
  shp$type <- factor(shp$type, levels = cols)
  ggplot(shp) +
    geom_sf(aes(fill = .data[["value"]])) +
    scale_fill_viridis_c(option = "D") +
    facet_wrap(~ .data[["type"]], nrow = 2, ncol = 2)
}

#' @importFrom sf st_read
#' @importFrom utils download.file
load_pretrained_shapefile <- function(model_name, verbose = TRUE) {
  base_url <-  "https://raw.githubusercontent.com/zhenhua-wang/vmsae_resources/refs/heads/main/shp_processed/"
  ## download shapefiles
  files <- paste0(model_name, c(".cpg", ".dbf", ".prj", ".shp", ".shx"))
  urls <- paste0(base_url, files)
  save_dir = tempdir()
  file_paths <- file.path(save_dir, files)
  tryCatch({
    for (i in seq_along(urls)) {
      download.file(urls[i], file_paths[i], mode = "wb", quiet = !verbose)
    }
  }, error = function(e) {
    if(verbose){
      cat("Error:", model_name, "could not be found. Please provide the necessary shapefiles in plot function.\n")
    }
  })
  ## load into sf
  shp_path <- file.path(save_dir, grep("shp$", files, value = TRUE))
  shapefile <- st_read(shp_path)
  return(shapefile)
}

#' Summarize VGMSFH Result
#'
#' This method provides a summary of posterior samples from a \code{VGMSFH} object, including posterior means and credible intervals for a specified parameter field.
#'
#' @param object An object of class \code{VGMSFH}, containing posterior samples from the model.
#' @param var_idx Integer. The index of the variable of interest (for multivariate models). Default is \code{1}.
#' @param field Character. The name of the slot in the \code{VGMSFH} object to summarize (e.g., \code{"beta_samples"}, \code{"spatial_samples"}, \code{"yhat_samples"}). Default is \code{"beta_samples"}.
#'
#' @return A data frame with columns:
#' \itemize{
#'   \item \code{mean}: Posterior mean,
#'   \item \code{lower}: Lower bound of the credible interval,
#'   \item \code{upper}: Upper bound of the credible interval.
#' }
#'
#' @details
#' This function extracts the posterior samples for the specified variable index, and combines it with \code{confint()} to compute credible intervals. The result is a compact summary table of central tendency and uncertainty.
#'
#' @examples
#' library(vmsae)
#' example_model <- readRDS(system.file("example", "example_model.Rds", package = "vmsae"))
#' summary(example_model)  # Summary of beta_samples for variable 1
#' summary(example_model, var_idx = 2, field = "yhat_samples")
#'
#' @importFrom methods slot
#'
#' @export
setMethod("summary", "VGMSFH", function(object, var_idx = 1, field = "beta_samples") {
  var_mean <- apply(ith_data(slot(object, field), var_idx), 2, mean)
  var_summary <- confint(object, var_idx = var_idx, field = field)
  var_summary["mean"] <- var_mean
  var_summary <- var_summary[, c("mean", "lower", "upper")]
  return(var_summary)
})

#' Extract Coefficients from a VGMSFH Object
#'
#' This method extracts posterior mean estimates of model coefficients from a \code{VGMSFH} object. It can return either fixed effect coefficients or spatial random effects.
#'
#' @param object An object of class \code{VGMSFH}.
#' @param var_idx Integer. The index of the variable of interest (for multivariate models). Default is \code{1}.
#' @param type Character. The type of coefficient to extract. Options are:
#' \itemize{
#'   \item \code{"fixed"} – extract the posterior mean of fixed effect coefficients (default).
#'   \item \code{"spatial"} – extract the posterior mean of spatial random effects.
#' }
#'
#' @return A numeric vector of posterior means for the selected coefficient type.
#'
#' @examples
#' library(vmsae)
#' example_model <- readRDS(system.file("example", "example_model.Rds", package = "vmsae"))
#' coef(example_model)  # Get fixed effect coefficients
#' coef(example_model, type = "spatial")  # Get spatial random effects
#'
#' @export
setMethod("coef", "VGMSFH", function(object, var_idx = 1, type = "fixed") {
  if (type == "fixed") {
    var <- ith_data(object@beta_samples, var_idx)
  } else if (type == "spatial") {
    var <- ith_data(object@spatial_samples, var_idx)
  }
  var_mean <- apply(var, 2, mean)
  return(var_mean)
})

#' Compute Credible Intervals for VGMSFH Parameters
#'
#' This method computes 95\% Bayesian credible intervals for the posterior samples of a selected parameter field in a \code{VGMSFH} object.
#'
#' @param object An object of class \code{VGMSFH}.
#' @param var_idx Integer. The index of the variable of interest (for multivariate models). Default is \code{1}.
#' @param field Character. The name of the slot to summarize (e.g., \code{"yhat_samples"}, \code{"beta_samples"}, \code{"spatial_samples"}). Default is \code{"yhat_samples"}.
#'
#' @return A data frame with two columns:
#' \itemize{
#'   \item \code{lower}: the 2.5\% quantile of the posterior distribution.
#'   \item \code{upper}: the 97.5\% quantile of the posterior distribution.
#' }
#'
#' @details
#' The function extracts posterior samples for the specified variable and then computes quantiles to form 95\% credible intervals. This is useful for uncertainty quantification in model predictions or parameter estimates.
#'
#' @examples
#' library(vmsae)
#' example_model <- readRDS(system.file("example", "example_model.Rds", package = "vmsae"))
#' confint(example_model)  # Get credible intervals for predicted values
#' confint(example_model, field = "beta_samples")  # For fixed effects
#'
#' @importFrom methods slot
#' @importFrom stats quantile
#'
#' @export
setMethod("confint", "VGMSFH", function(object, var_idx = 1, field = "yhat_samples") {
  var_lower <- apply(
    ith_data(slot(object, field), var_idx), 2, quantile, 0.025)
  var_upper <- apply(
    ith_data(slot(object, field), var_idx), 2, quantile, 0.975)
  var_confint <- data.frame(lower = var_lower, upper = var_upper)
  return(var_confint)
})

ith_data <- function(data, var_idx, verbose = TRUE) {
  if (is.null(dim(data))) {
    if(verbose){
      warning("Univariate data, returning the only response.")
    }
    return(data)
  } else if (length(dim(data)) == 2) {
    return(data[, var_idx])
  } else if (length(dim(data)) == 3) {
    return(data[, , var_idx])
  }
}
