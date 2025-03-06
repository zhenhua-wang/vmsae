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
plot.VGMSFH <- function(object, shp = NULL, var_idx = 1, field = "yhat_samples", ...) {
  if (is.null(shp)) {
    ## download shapefile from pretrained model
    shp <- load_shapefile_from_url(object@model_name,
      "https://github.com/zhenhua-wang/VMSAE_resources/tree/main/shp_processed")
  }
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

#' @importFrom sf st_read
load_shapefile_from_url <- function(name, base_url) {
  ## download shapefiles
  files <- paste0(name, c(".cpg", ".dbf", ".prj", ".shp", ".shx"))
  urls <- paste0(base_url, files)
  temp_dir <- tempdir()
  file_paths <- file.path(temp_dir, files)
  for (i in seq_along(urls)) {
    download.file(urls[i], file_paths[i], mode = "wb")
  }
  ## load into sf
  shp_path <- file.path(temp_dir, grep("shp$", files, value = TRUE))
  shapefile <- st_read(shp_path)
  return(shapefile)
}

#' Summarize VGMSFH result.
#'
#' This function display the summary of VGMSFH result.
#'
#' @return data frame, summary of VGMSFH result.
#' @export
setMethod("summary", "VGMSFH", function(object, var_idx = 1, field = "beta_samples") {
  var_mean <- apply(slot(object, field)[, , var_idx], 2, mean)
  var_summary <- confint(object, var_idx, field)
  var_summary["mean"] <- var_mean
  var_summary <- var_summary[, c("mean", "lower", "upper")]
  return(var_summary)
})

#' @export
setMethod("coef", "VGMSFH", function(object, var_idx = 1, type = "fixed") {
  if (type == "fixed") {
    var <- object@beta_samples[, , var_idx]
  } else if (type == "spatial") {
    var <- object@spatial_samples[, , var_idx]
  }
  var_mean <- apply(var, 2, mean)
  return(var_mean)
})

#' @export
setMethod("confint", "VGMSFH", function(object, var_idx = 1, field = "yhat_samples") {
  var_lower <- apply(slot(object, field)[, , var_idx], 2, quantile, 0.025)
  var_upper <- apply(slot(object, field)[, , var_idx], 2, quantile, 0.975)
  var_confint <- data.frame(lower = var_lower, upper = var_upper)
  return(var_confint)
})
