#' Plot VGMSFH result.
#'
#' This function plot the spatial map of VGMSFH result.
#'
#' @param object An object containing the VGMSFH results.
#' @param shp The shapefile that contains the geometry. If null, this is downloaded from the pretrain model.
#' @param var_idx Index of the variable of interest.
#' @param type Type of plots. "compare" is the plot comparing direct estimate and model estimate. "estimate" is the plot that shows the mean and standard deviation of the model estimate.
#' @export
setMethod("plot", "VGMSFH",
  function(x, shp = NULL, var_idx = 1, type = "compare") {
    if (is.null(shp)) {
      ## download shapefile from pretrained model
      shp <- load_pretrained_shapefile(x@model_name)
    }
    if (type == "estimate") {
      plot_estimate(x, shp, var_idx)
    } else if(type == "compare") {
      plot_compare(x, shp, var_idx)
    } else {
      warning("Plot type not supported.")
    }
  })

#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 geom_sf
#' @importFrom ggplot2 scale_fill_viridis_c
#' @importFrom tidyr pivot_longer
#' @importFrom gridExtra grid.arrange
plot_estimate <- function(object, shp, var_idx) {
  var <- ith_data(slot(object, "yhat_samples"))
  shp["mean"] <- apply(var, 2, mean)
  shp["std"] <- apply(var, 2, sd)
  p1 <- ggplot() +
    geom_sf(data = shp, aes(fill = mean)) +
    scale_fill_viridis_c(name = "Mean",
      option = "D")
  p2 <- ggplot() +
    geom_sf(data = shp, aes(fill = std)) +
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
plot_compare <- function(object, shp, var_idx) {
  var <- ith_data(slot(object, "yhat_samples"))
  shp["direct estimate"] <- ith_data(slot(object, "direct_estimate"))
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
    geom_sf(aes(fill = value)) +
    scale_fill_viridis_c(option = "D") +
    facet_wrap(~ type, nrow = 2, ncol = 2)
}

#' @importFrom sf st_read
load_pretrained_shapefile <- function(model_name) {
  base_url <-  "https://raw.githubusercontent.com/zhenhua-wang/vmsae_resources/refs/heads/main/shp_processed/"
  ## download shapefiles
  files <- paste0(model_name, c(".cpg", ".dbf", ".prj", ".shp", ".shx"))
  urls <- paste0(base_url, files)
  save_dir = tempdir()
  file_paths <- file.path(save_dir, files)
  tryCatch({
    for (i in seq_along(urls)) {
      download.file(urls[i], file_paths[i], mode = "wb")
    }
  }, error = function(e) {
    cat("Error:", model_name, "could not be found. Please provide the necessary shapefiles in plot function.\n")
  })
  ## load into sf
  shp_path <- file.path(save_dir, grep("shp$", files, value = TRUE))
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
  var_mean <- apply(ith_data(slot(object, field)), 2, mean)
  var_summary <- confint(object, var_idx, field)
  var_summary["mean"] <- var_mean
  var_summary <- var_summary[, c("mean", "lower", "upper")]
  return(var_summary)
})

#' @export
setMethod("coef", "VGMSFH", function(object, var_idx = 1, type = "fixed") {
  if (type == "fixed") {
    var <- ith_data(object@beta_samples)
  } else if (type == "spatial") {
    var <- ith_data(object@spatial_samples)
  }
  var_mean <- apply(var, 2, mean)
  return(var_mean)
})

#' @export
setMethod("confint", "VGMSFH", function(object, var_idx = 1, field = "yhat_samples") {
  var_lower <- apply(ith_data(slot(object, field)), 2, quantile, 0.025)
  var_upper <- apply(ith_data(slot(object, field)), 2, quantile, 0.975)
  var_confint <- data.frame(lower = var_lower, upper = var_upper)
  return(var_confint)
})

ith_data <- function(data, var_idx) {
  if (is.null(dim(data))) {
    warning("Univariate data, returning the only response.")
    return(data)
  } else {
    return(data[, , var_idx])
  }
}
