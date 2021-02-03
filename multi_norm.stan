data {
  int<lower=0> p;
  int<lower=0> N;
  matrix[N,p] y;
}

parameters {
  vector[p] theta;
  corr_matrix[p] H;
  real<lower = 0> sigmasq;
}

model {
  for(i in 1:N){
    y[i,:] ~ multi_normal(theta, sigmasq * H);
  }
}
