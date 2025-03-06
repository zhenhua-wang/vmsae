library(sf)
library(tidyverse)
library(spdep)
library(vmsae)
install_environment()
load_environment()

acs_data <-
  read_sf(system.file("data", "mo_county.shp", package = "vmsae")) %>%
  na.omit()
W <- nb2mat(poly2nb(acs_data), style = "B", zero.policy = TRUE)

loss <- train_vae(W = W,
  GEOID = acs_data$GEOID,
  model_name = "test",
  save_dir = ".",
  n_samples = 10000,
  batch_size = 256,
  epoch = 10000)
