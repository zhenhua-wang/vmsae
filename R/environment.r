#' create the python environment.
#'
#' This function loads the vgmsfh numpyro module
#'
#' @param envname String, string path to python virtual environment.
#' @export
install_environment <- function(envname = "vmsae") {
  reticulate::install_python()
  reticulate::py_install("torch", envname = envname)
  reticulate::py_install("numpyro", envname = envname)
}

#' Load the vgmsfh numpyro module.
#'
#' This function loads the vgmsfh numpyro module
#'
#' @param envname String, string path to python virtual environment.
#' @export
load_environment <- function(envname = "vmsae") {
  vgmcar_module <- system.file("py", "vgmcar.py", package = "vmsae")
  vae_module <- system.file("py", "vae.py", package = "vmsae")
  train_vae_module <- system.file("py", "train_vae.py", package = "vmsae")
  car_dataset_module <- system.file("py", "car_dataset.py", package = "vmsae")
  reticulate::use_virtualenv(envname, required = TRUE)
  reticulate::py_config()
  reticulate::source_python(vgmcar_module)
  reticulate::source_python(vae_module)
  reticulate::source_python(train_vae_module)
  reticulate::source_python(car_dataset_module)
}

#' Download pretrained VAE model.
#'
#' This function downloads pretrained VAE model and the corresponding GEOID.
#'
#' @param model_name String, vae model name. e.g. "mo_county".
#' @export
download_pretrained_vae <- function(model_name, save_dir) {
  url <- "https://github.com/zhenhua-wang/vmsae_resources/raw/refs/heads/main/model/"
  vae_name <- paste0(model_name, ".model")
  GEOID_name <- paste0(model_name, ".GEOID")
  model_save_path <- file.path(save_dir, vae_name)
  GEOID_save_path <- file.path(save_dir, GEOID_name)
  tryCatch({
    download.file(
      url = paste0(url, vae_name),
      destfile = model_save_path)
    download.file(
      url = paste0(url, GEOID_name),
      destfile = GEOID_save_path)
    cat(vae_name, "downloaded successfully\n")
  }, error = function(e) {
    cat("Error:", vae_name, "could not be found.\n")
  })
}
