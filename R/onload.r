.onLoad <- function(libname, pkgname) {
  # Check if reticulate is loaded, load it if not
  if (!("reticulate" %in% loadedNamespaces())) {
    message("Loading reticulate...")
    library(reticulate)
  }
}
