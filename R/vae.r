#' Decoder S4 Class
#'
#' An S4 class to represent a neural network decoder, used for emulating spatial priors.
#' The class includes parameters for input and output weight matrices and biases, as well as region identifiers.
#'
#' @slot GEOID A character vector of region or area identifiers.
#' @slot W_in An array representing the input weight matrix of the decoder.
#' @slot B_in An array representing the input bias vector of the decoder.
#' @slot W_out An array representing the output weight matrix of the decoder.
#' @slot B_out An array representing the output bias vector of the decoder.
#'
#' @export
setClass("Decoder",
  slots = c(
    GEOID = "character",
    W_in = "array",
    B_in = "array",
    W_out = "array",
    B_out = "array"
  )
)

#' Train VAE for CAR Prior
#'
#' Trains a Variational Autoencoder (VAE) to learn the spatial structure implied by the
#' Conditional Autoregressive (CAR) prior. The trained VAE parameters are saved and can
#' later be used as a generator within Hamiltonian Monte Carlo (HMC) sampling.
#'
#' @param W Matrix. A proximity or adjacency matrix representing spatial relationships.
#' @param GEOID Character vector. Identifiers for spatial units (e.g., region or area codes).
#' @param model_name Character. The name of the trained VAE model.
#' @param save_dir Character. Directory to save the trained VAE model and associated metadata. Defaults to the current working directory.
#' @param n_samples Integer. Number of samples to draw from the prior for training. Default is \code{10000}.
#' @param batch_size Integer. Batch size for VAE training. Default is \code{256}.
#' @param epoch Integer. Number of training epochs. Default is \code{10000}.
#' @param lr_init Numeric. Initial learning rate. Default is \code{0.001}.
#' @param lr_min Numeric. Minimum learning rate at the final epoch. Default is \code{1e-7}.
#' @param verbose Logical; if \code{TRUE} (default), prints progress.
#' @param use_gpu Boolean. Use GPU if available.
#'        Default is `TRUE`.
#'
#' @return A named list containing:
#' \item{loss}{Total training loss}
#' \item{RCL}{Reconstruction error}
#' \item{KLD}{Kullback–Leibler divergence}
#'
#' @details
#' The function requires a configured Python environment via the \pkg{reticulate} interface,
#' with VAE training implemented in Python. It uses \code{py$train_vae()} defined in the
#' sourced Python modules (see \code{\link{load_environment}}).
#'
#' @examples
#' \dontrun{
#' library(vmsae)
#' library(sf)
#' # this function is time consuming for the first run
#' install_environment()
#' load_environment()
#'
#' acs_data <- read_sf(system.file("example", "mo_county.shp", package = "vmsae"))
#' W <- readRDS(system.file("example", "W.Rds", package = "vmsae"))
#'
#' loss <- train_vae(W = W,
#'   GEOID = acs_data$GEOID,
#'   model_name = "test",
#'   save_dir = tempdir(),
#'   n_samples = 1000, # set to larger values in practice, e.g. 10000.
#'   batch_size = 256,
#'   epoch = 1000)     # set to larger values in practice, e.g. 10000.
#' }
#'
#' @importFrom reticulate py
#' @importFrom utils write.table
#'
#' @export
train_vae <- function(W, GEOID, model_name, save_dir,
                      n_samples = 10000, batch_size = 256, epoch = 10000,
                      lr_init = 0.001, lr_min = 1e-7,
                      verbose = TRUE, use_gpu = TRUE) {
  save_path <- get_save_path(model_name, save_dir)
  vae_path <- save_path$vae_path
  GEOID_path <- save_path$GEOID_path
  loss <- py$train_vae(W, vae_path,
    n_samples, batch_size, epoch,
    lr_init, lr_min, verbose, use_gpu)
  write.table(GEOID, file = GEOID_path,
    row.names = FALSE, col.names = FALSE)
  return(list(loss = loss[[1]], RCL = loss[[2]], KLD = loss[[3]]))
}

#' Load Pretrained VAE Decoder
#'
#' Load a pretrained Variational Autoencoder (VAE) decoder from disk. This function reads the saved PyTorch model weights and corresponding GEOID list, and constructs a `Decoder` S4 object with the loaded parameters.
#'
#' @param model_name Character. The name of the trained VAE model (without `.zip` extensions).
#' @param save_dir Character. The directory where the trained VAE model is saved. Defaults to the current directory if `NULL`.
#'
#' @return An object of class \code{Decoder}, containing the decoder weights and region identifiers.
#'
#' @details
#' This function assumes the model was trained and saved using `train_vae()`, and that the decoder weights are stored in a file compatible with `torch::load()` (via reticulate). It extracts the decoder input/output weights and biases, along with region GEOIDs, and returns them as an S4 object of class `Decoder`.
#'
#' @examples
#' \dontrun{
#' library(vmsae)
#' # this function is time consuming for the first run
#' install_environment()
#' load_environment()
#' decoder <- load_vae(model_name = "mo_county")
#' }
#'
#' @importFrom reticulate py
#' @importFrom methods new
#' @importFrom utils read.table
#'
#' @export
load_vae <- function(model_name, save_dir = NULL) {
  save_path <- get_save_path(model_name, save_dir)
  vae_path <- save_path$vae_path
  GEOID_path <- save_path$GEOID_path
  vae_model <- py$torch$load(vae_path,
    map_location = 'cpu',
    weights_only = TRUE)
  GEOID <- read.table(GEOID_path, header = FALSE, colClasses = "character")
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
