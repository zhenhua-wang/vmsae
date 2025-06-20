#' Install python environment.
#'
#' This function creates the vmsae python environment and installs required packages.
#'
#' @param envname Character. The name of the Python environment to create or update.
#'        Default is `"vmsae"`.
#'
#' @return No return value, called for side effects
#'
#' @examples
#' \dontrun{
#' library(vmsae)
#' # this function is time consuming for the first run
#' install_environment()          # Install into default "vmsae" environment
#'
#'
#' # this step is time consuming for the first run
#' install_environment("custom")  # Install into a custom-named environment
#' }
#'
#' @importFrom reticulate install_python
#' @importFrom reticulate py_install
#'
#' @export
install_environment <- function(envname = "vmsae") {
  install_python()
  py_install("torch", envname = envname)
  py_install("numpyro", envname = envname)
}

#' Load Python Environment and Source Model Modules
#'
#' This function activates a specified Python virtual environment and sources Python modules
#' used by the \pkg{vmsae} package, including models and python scripts.
#'
#' @param envname Character. The name of the Python environment to create or update.
#'        Default is `"vmsae"`.
#' @param is_conda Boolean. The indicator for whether the loaded environment is a conda environment.
#'        Default is `"FALSE"`.
#'
#' @details
#' The function loads four Python scripts located in the package's `py/` directory:
#' \itemize{
#'   \item \code{vgmcar.py}
#'   \item \code{vae.py}
#'   \item \code{train_vae.py}
#'   \item \code{car_dataset.py}
#' }
#'
#' The environment must be created beforehand (e.g., using `install_environment()`),
#' and must include all Python dependencies required by these modules.
#'
#' @return No return value, called for side effects
#'
#' @examples
#' \dontrun{
#' library(vmsae)
#'
#' # this function is time consuming for the first run
#' install_environment()
#' load_environment()          # Load default "vmsae" environment
#'
#' # this function is time consuming for the first run
#' install_environment("custom")
#' load_environment("custom") # Load custom virtual environment
#'
#' load_environment("custom", is_conda = TRUE) # Load custom conda environment
#' }
#'
#' @importFrom reticulate use_condaenv
#' @importFrom reticulate use_virtualenv
#' @importFrom reticulate py_config
#' @importFrom reticulate source_python
#'
#' @export
load_environment <- function(envname = "vmsae", is_conda = FALSE) {
  vgmcar_module <- system.file("py", "vgmcar.py", package = "vmsae")
  vae_module <- system.file("py", "vae.py", package = "vmsae")
  train_vae_module <- system.file("py", "train_vae.py", package = "vmsae")
  car_dataset_module <- system.file("py", "car_dataset.py", package = "vmsae")
  if (is_conda) {
    use_condaenv(envname, required = TRUE)
  } else {
    use_virtualenv(envname, required = TRUE)
  }
  py_config()
  source_python(vgmcar_module)
  source_python(vae_module)
  source_python(train_vae_module)
  source_python(car_dataset_module)
}

#' Download and Extract a Pretrained VAE Model
#'
#' This function downloads a pretrained VAE model archive from Zenodo, extracts its contents
#' into a specified directory, and removes the downloaded ZIP file after extraction.
#'
#' @param model_name Character. The name of the model file (without extension) to download.
#'        This should correspond to a `*model_name*.zip` file hosted on Zenodo (e.g., `"ca_county"`).
#' @param save_dir Character. The local directory where the model should be saved and extracted.
#' @param verbose Logical; if \code{TRUE} (default), prints progress and error messages.
#'
#' @return No return value, called for side effects
#'
#' @examples
#' \dontrun{
#' library(vmsae)
#' # this function is time consuming for the first run
#' install_environment()
#' load_environment()
#' download_pretrained_vae("mo_county", tempdir())
#' }
#'
#' @importFrom utils download.file
#' @importFrom utils unzip
#'
#' @export
download_pretrained_vae <- function(model_name, save_dir, verbose = TRUE) {
  url <- "https://zenodo.org/records/14993110/files/%s?download=1"
  zip_name <- paste0(model_name, ".zip")
  zip_save_path <- file.path(save_dir, zip_name)
  tryCatch({
    download.file(
      url = sprintf(url, zip_name),
      destfile = zip_save_path,
      quiet = !verbose)
    unzip(zipfile = zip_save_path, exdir = save_dir)
    file.remove(zip_save_path)
    if(verbose){
      cat(model_name, "downloaded successfully\n")
    }
  }, error = function(e) {
    if(verbose){
      cat("Error:", model_name, "could not be found.\n")
    }
  })
}
