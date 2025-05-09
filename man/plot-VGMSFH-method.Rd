% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/vgmsfh_utils.r
\name{plot,VGMSFH-method}
\alias{plot,VGMSFH-method}
\title{Plot VGMSFH Result}
\usage{
\S4method{plot}{VGMSFH}(x, shp = NULL, var_idx = 1, type = "compare", verbose = TRUE)
}
\arguments{
\item{x}{An object of class \code{VGMSFH}, containing posterior samples and direct estimates from the model.}

\item{shp}{An \code{sf} object representing the spatial shapefile. If \code{NULL}, the function will automatically download a shapefile associated with the pretrained model.}

\item{var_idx}{Integer. The index of the variable of interest (for multivariate models).}

\item{type}{Character. The type of plot to generate. Options are:
\itemize{
\item \code{"compare"} – compare direct estimates and model-based estimates.
\item \code{"estimate"} – show the posterior mean and standard deviation of the model estimate.
}}

\item{verbose}{Logical; if \code{TRUE} (default), prints error messages.}
}
\value{
A \code{ggplot} object. The plot is rendered to the active device.
}
\description{
This method plots spatial summaries of results from a \code{VGMSFH} object, including model estimates and comparisons with direct estimates.
}
\details{
The function provides spatial visualization of model results. It supports both univariate and multivariate response settings. When \code{type = "compare"}, it generates side-by-side choropleth maps for the direct and model-based estimates. When \code{type = "estimate"}, it plots the posterior mean and standard deviation of the VGMSFH model output.

If no shapefile is provided, a default geometry is loaded from the pretrained repository.
}
\examples{
library(vmsae)
library(sf)
example_model <- readRDS(system.file("example", "example_model.Rds", package = "vmsae"))
example_shp <- read_sf(system.file("example", "mo_county.shp", package = "vmsae"))
plot(example_model, shp = example_shp, type = "compare")
plot(example_model, shp = example_shp, type = "estimate", var_idx = 2)

}
