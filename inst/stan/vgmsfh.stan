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
  int p_y;
  int p_x;
  int p_latent;
  int p_hidden;
  matrix[p_latent, p_hidden] W_in;
  vector[p_hidden] B_in;
  matrix[p_hidden, N] W_out;
  vector[N] B_out;

  matrix[N, p_y] y;
  matrix[N, p_y] y_sigma;
  matrix[N, p_x] X;
  matrix[N, N] W;
}
parameters {
  vector[p_y] mu;
  matrix[p_x, p_y] beta;
  matrix[N, p_y] delta;
  vector[p_latent] z1;
  vector[p_latent] z2;
  vector<lower=0>[p_y] phi_tau;
  vector<lower=0>[p_y] delta_tau;
  vector[p_y] phi_eta;
}
transformed parameters {
  matrix[N, p_y] phi;
  matrix[N, p_y] y_hat;
  matrix[N, N] A;
  vector<lower=0>[p_y] phi_sigma;
  vector<lower=0>[p_y] delta_sigma;

  phi_sigma = 1/sqrt(phi_tau);
  delta_sigma = 1/sqrt(delta_tau);
  A = phi_eta[1]*diag_matrix(rep_vector(1, N)) + phi_eta[2]*W;
  phi[:, 2] = phi_sigma[2] * decoder(z2, W_in, W_out, B_in, B_out);
  phi[:, 1] = A * phi[:, 2] + phi_sigma[1] * decoder(z1, W_in, W_out, B_in, B_out);
  y_hat = rep_matrix(mu', N) + X * beta + phi + delta;
}
model {
  //likelihood
  to_vector(y) ~ normal(to_vector(y_hat),to_vector(y_sigma));

  //priors
  mu ~ normal(0, 100);
  to_vector(beta) ~ normal(0, 100);
  delta[:, 1] ~ normal(0, delta_sigma[1]);
  delta[:, 2] ~ normal(0, delta_sigma[2]);
  z1 ~ normal(0, 1);
  z2 ~ normal(0, 1);
  phi_tau ~ gamma(0.001, 0.001);
  delta_tau ~ gamma(0.001, 0.001);
  phi_eta ~ normal(0, 100);
}
