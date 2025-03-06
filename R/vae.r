#' Class representing a VAE decoder.
#'
#' @slot GEOID character vector, The FIPS codes or other equivalent GEOIDs.
#' @slot W_in array, The input weights.
#' @slot B_in array, The input bias.
#' @slot W_out array, The output weights.
#' @slot B_in array, The output bias.
setClass("Decoder",
  slots = c(
    GEOID = "character",
    W_in = "array",
    B_in = "array",
    W_out = "array",
    B_out = "array"
  )
)

#' Train VAE for CAR prior.
#'
#' Train a Variational Autoencoder to learn the spatial structure of the Conditional Autoregressive prior and save the parameters in each layer, which can later be used as the generator in the Hamiltonian Monte Carlo step.
#'
#' @param W Matrix, proximity matrix.
#' @param model_name, String, trained VAE model name in lower case.
#' @param save_dir, String, directory to save the trained VAE model. Default to the current directory.
#' @param n_samples, Int, number of training samples drawn from the prior. Default to 10,000.
#' @param batch_size, Int, batch size of VAE. Default to 256.
#' @param epoch, Int epoch of VAE. Default to 10,000.
#' @param lr_init, Float, initial learning rate. Default to 0.001.
#' @param lr_min, Float, reduced learning rate at the last epoch. Default to 1e-7.
#' @export
#'
#' @return A list that contains the total loss, reconstructed error and KL divergence of VAE
#'
#' @examples
#' library(sf)
#' library(tidyverse)
#' library(spdep)
#' library(vmsae)
#' install_environment()
#' load_environment()
#'
#' acs_data <-
#'   read_sf(system.file("data", "mo_county.shp", package = "vmsae")) %>%
#'   na.omit()
#' W <- nb2mat(poly2nb(acs_data), style = "B", zero.policy = TRUE)
#'
#' loss <- train_vae(W = W,
#'   GEOID = acs_data$GEOID,
#'   model_name = "test",
#'   save_dir = ".",
#'   n_samples = 10000,
#'   batch_size = 256,
#'   epoch = 10000)
train_vae <- function(W, GEOID, model_name, save_dir = ".",
                      n_samples = 10000, batch_size = 256, epoch = 10000,
                      lr_init = 0.001, lr_min = 1e-7) {
  save_path <- get_save_path(model_name, save_dir)
  vae_path <- save_path$vae_path
  GEOID_path <- save_path$GEOID_path
  loss <- py$train_vae(W, vae_path,
    n_samples, batch_size, epoch,
    lr_init, lr_min)
  write.table(GEOID, file = GEOID_path,
    row.names = FALSE, col.names = FALSE)
  return(list(loss = loss[[1]], RCL = loss[[2]], KLD = loss[[3]]))
}


#' Load pretrained VAE decoder.
#'
#' @param model_name, String, trained VAE model name in lower case.
#' @param save_dir, String, directory to save the trained VAE model. Default to the current directory.
#' @export
#'
#' @return An object of the Decoder class.
load_vae <- function(model_name, save_dir = NULL) {
  save_path <- get_save_path(model_name, save_dir)
  vae_path <- save_path$vae_path
  GEOID_path <- save_path$GEOID_path
  vae_model <- py$torch$load(vae_path, weights_only = TRUE)
  GEOID <- read.table(GEOID_path, header = FALSE)
  W_in <- vae_model$dec_input.weight$numpy()
  B_in <- vae_model$dec_input.bias$numpy()
  W_out <- vae_model$dec_out.weight$numpy()
  B_out <- vae_model$dec_out.bias$numpy()
  vae_weights <- new("Decoder",
    GEOID = as.character(GEOID$V1),
    W_in = W_in, B_in = B_in,
    W_out = W_out, B_out = B_out)
  return(vae_weights)
}

get_save_path <- function(model_name, save_dir) {
  model_name <- tolower(model_name)
  GEOID_name <- paste0(tolower(model_name), ".GEOID")
  vae_name <- paste0(tolower(model_name), ".model")
  if (is.null(save_dir)) {
    GEOID_path <- system.file("model", GEOID_name, package = "vmsae")
    vae_path <- system.file("model", vae_name, package = "vmsae")
  } else {
    GEOID_path <- file.path(save_dir, GEOID_name)
    vae_path <- file.path(save_dir, vae_name)
  }
  return(list(vae_path = vae_path, GEOID_path = GEOID_path))
}
