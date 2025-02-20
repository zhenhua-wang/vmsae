library(sf)
library(tidyverse)
library(spdep)
library(reticulate)
library(vmsae)
install_environment()
load_environment()


state <- "mo"
geography <- "county"
acs_data <-
  read_sf(system.file("data", sprintf("%s_%s.shp",
    tolower(state), tolower(geography)),
    package = "vmsae")) %>%
  mutate(
    var = (moe / 1.645)^2,
    estimate_log = log(estimate),
    var_log = (1 / estimate)^2 * var,
    var2 = (moe2 / 1.645)^2,
    estimate2_log = log(estimate2),
    var2_log = (1 / estimate2)^2 * var2) %>%
  na.omit()
W <- nb2mat(poly2nb(acs_data), style = "B", zero.policy = TRUE)

loss <- train_vae(W = W,
  GEOID = acs_data$GEOID,
  vae_model_name = "test",
  vae_save_dir = ".",
  n_samples = 10000,
  batch_size = 256,
  epoch = 3)
