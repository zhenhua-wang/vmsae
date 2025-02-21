#' Train VAE for spatial random effect.
#'
#' This function trains a VAE that learn the conditional autoregressive prior.
#'
#' @param W Matrix, proximity matrix.
#' @param vae_model_name, String, trained VAE model name.
#' @param vae_save_dir, String, directory to save the trained VAE model.
#' @param n_samples, Int, number of training samples drawn from the prior.
#' @param batch_size, Int, batch size of VAE.
#' @param epoch, Int epoch of VAE.
#' @param lr_init, Float, initial learning rate.
#' @param lr_min, Float, reduced learning rate at the last epoch.
#' @export
train_vae <- function(W, GEOID, vae_model_name, vae_save_dir = ".",
                      n_samples = 10000, batch_size = 256, epoch = 10000,
                      lr_init = 0.001, lr_min = 1e-7) {
  vae_model_name <- tolower(vae_model_name)
  vae_save_path <- file.path(
    vae_save_dir, paste0(vae_model_name, ".model"))
  GEOID_save_path <- file.path(
    vae_save_dir, paste0(vae_model_name, ".GEOID"))
  loss <- py$train_vae(W, vae_save_path,
    n_samples, batch_size, epoch,
    lr_init, lr_min)
  write.table(GEOID, file = GEOID_save_path,
    row.names = FALSE, col.names = FALSE)
  return(list(loss = loss[[1]], RCL = loss[[2]], KLD = loss[[3]]))
}
