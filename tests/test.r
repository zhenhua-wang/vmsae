library(sf)
library(tidyverse)
library(spdep)
library(vmsae)
install_environment()
load_environment()

acs_data <-
  read_sf(system.file("data", "mo_county.shp", package = "vmsae")) %>%
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
model <- vgmsfh_numpyro(y, y_sigma, X, W,
  GEOID = acs_data$GEOID,
  model_name = "mo_county", save_dir = NULL,
  num_samples = num_samples, num_warmup = num_samples)
y_hat_np <- model@yhat_samples
y_hat_mean_np <- apply(y_hat_np, c(2, 3), mean)
y_hat_lower_np <- apply(y_hat_np, c(2, 3), quantile, 0.025)
y_hat_upper_np <- apply(y_hat_np, c(2, 3), quantile, 0.975)

plot(model, shp = acs_data, type = "compare", var_idx = 2)
## remove.packages("vmsae")
## devtools::document();devtools::install(".")
