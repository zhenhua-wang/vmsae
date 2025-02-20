library(sf)
library(tidyverse)
library(cmdstanr)
library(spdep)
library(reticulate)
library(vmsae)
install_environment()
load_environment()

state <- "mo"
geography <- "county"
shp_name <- sprintf("%s_%s.shp", tolower(state), tolower(geography))
vae_name <- sprintf("%s_%s", state, geography)
acs_data <-
  read_sf(system.file("data", shp_name, package = "vmsae")) %>%
  mutate(
    var = (moe / 1.645)^2,
    estimate_log = log(estimate),
    var_log = (1 / estimate)^2 * var,
    var2 = (moe2 / 1.645)^2,
    estimate2_log = log(estimate2),
    var2_log = (1 / estimate2)^2 * var2) %>%
  na.omit()

y <- acs_data %>%
  select(estimate_log, estimate2_log) %>%
  st_drop_geometry() %>%
  as.matrix()
y_sigma <- acs_data %>%
  select(var_log, var2_log) %>%
  st_drop_geometry() %>%
  as.matrix() %>%
  sqrt()
X <- acs_data %>%
  select(poverty, black, india, asian) %>%
  st_drop_geometry() %>%
  as.matrix()
W <- nb2mat(poly2nb(acs_data), style = "B", zero.policy = TRUE)

num_samples <- 10000
pos_samples <- vgmsfh_numpyro(y, y_sigma, X, W,
  GEOID = acs_data$GEOID,
  vae_model_name = vae_name, vae_save_dir = NULL,
  num_samples = num_samples, num_warmup = num_samples)
y_hat_np <- pos_samples@y_hat
y_hat_mean_np <- apply(y_hat_np, c(2, 3), mean)
y_hat_lower_np <- apply(y_hat_np, c(2, 3), quantile, 0.025)
y_hat_upper_np <- apply(y_hat_np, c(2, 3), quantile, 0.975)

## remove.packages("vmsae")
## devtools::document();devtools::install(".")
plot(pos_samples, acs_data)
