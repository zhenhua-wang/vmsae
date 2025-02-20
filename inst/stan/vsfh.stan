functions {
  vector elu(vector x) {
    int N = num_elements(x);
    vector[N] result;
    for (n in 1:N) {
      result[n] = (x[n] > 0) ? x[n] : 1.0 * (exp(x[n]) - 1);
    }
    return result;
  }
  vector layer(vector x, matrix W, vector B) {
    return(transpose(transpose(x) * W) + B);
  }
  vector decoder(vector input,
                 matrix W_in, matrix W_out,
                 vector B_in, vector B_out) {
    return(layer(elu(layer(input,W_in,B_in)),W_out,B_out));
  }
}
data {
  int N;
  int p_latent;
  int p_hidden;
  int p_x;
  matrix[p_latent, p_hidden] W_in;
  vector[p_hidden] B_in;
  matrix[p_hidden, N] W_out;
  vector[N] B_out;

  vector[N] y;
  vector[N] y_sigma;
  matrix[N, p_x] X;
}
parameters {
  real mu;
  vector[p_x] beta;
  vector[N] delta;
  vector[p_latent] z;
  real<lower=0> tau_phi;
  real<lower=0> tau_delta;
}
transformed parameters {
  vector[N] phi;
  vector[N] y_hat;
  real<lower=0> sigma_phi;
  real<lower=0> sigma_delta;

  phi = decoder(z, W_in, W_out, B_in, B_out);
  sigma_phi = 1/sqrt(tau_phi);
  sigma_delta = 1/sqrt(tau_delta);
  y_hat = mu + X * beta + sigma_phi * phi + delta;
}
model {
  //likelihood
  y ~ normal(y_hat,y_sigma);

  //priors
  mu ~ normal(0, 100);
  beta ~ normal(0, 100);
  delta ~ normal(0, sigma_delta);
  z ~ normal(0, 1);
  tau_phi ~ gamma(0.001, 0.001);
  tau_delta ~ gamma(0.001, 0.001);
}
