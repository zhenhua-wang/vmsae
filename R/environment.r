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
  url <- "https://zenodo.org/records/14993110/files/%s?download=1"
  zip_name <- paste0(model_name, ".zip")
  zip_save_path <- file.path(save_dir, zip_name)
  tryCatch({
    download.file(
      url = sprintf(url, zip_name),
      destfile = zip_save_path)
    unzip(zipfile = zip_save_path, exdir = save_dir)
    file.remove(zip_save_path)
    cat(model_name, "downloaded successfully\n")
  }, error = function(e) {
    cat("Error:", model_name, "could not be found.\n")
  })
}
