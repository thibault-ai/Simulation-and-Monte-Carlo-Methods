import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax import random, jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import truncnorm, norm
import tqdm
import pandas as pd
import numpy as np
import pyarrow.feather as feather 
import time
path = "results/thorisson/"

key = random.PRNGKey(62665478)

def sample_multivariate_normal(mean, cov, key):
    return random.multivariate_normal(key, mean, cov)

def multivariate_normal_pdf(x, mean, cov):
    return multivariate_normal.pdf(x, mean=mean, cov=cov)

def ThorissonCoupling(p_sample, q_sample, p_pdf, q_pdf, C, mean_p, cov_p, mean_q, cov_q, key):
    key, subkey = random.split(key)
    X = p_sample(mean_p, cov_p, subkey)
    U = random.uniform(key, minval=0.0, maxval=1.0)

    if U < min((q_pdf(X, mean_q, cov_q) / p_pdf(X, mean_p, cov_p)), C):
        Y = X
    else:
        A = 0
        while A != 1:
            key, subkey = random.split(key)
            U = random.uniform(key, minval=0.0, maxval=1.0)
            Z = q_sample(mean_q, cov_q, subkey)
            if U > min(1, (C * p_pdf(Z, mean_p, cov_p) / q_pdf(Z, mean_q, cov_q))):
                A = 1
                Y = Z
    return X, Y

mean_p = jnp.array([0, 0])
cov_p = jnp.identity(2)
mean_q = jnp.array([2, 2])
cov_q = jnp.identity(2) * 2
C = 0.3

num_samples = 100
samples_X, samples_Y = [], []
for _ in tqdm.tqdm(range(num_samples)):
    key, subkey = random.split(key)
    X, Y = ThorissonCoupling(sample_multivariate_normal, sample_multivariate_normal,
                             multivariate_normal_pdf, multivariate_normal_pdf,
                             C, mean_p, cov_p, mean_q, cov_q, subkey)
    samples_X.append(X)
    samples_Y.append(Y)

samples_X = jnp.array(samples_X)
samples_Y = jnp.array(samples_Y)


def compute_kde(samples):
    from scipy.stats import gaussian_kde
    samples = np.array(samples) 
    kde = gaussian_kde(samples.T, bw_method='silverman')
    x_min, y_min = samples.min(axis=0)
    x_max, y_max = samples.max(axis=0)
    X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    density = np.reshape(kde(positions).T, X.shape)
    return X, Y, density

X_p, Y_p, density_p = compute_kde(samples_X)
X_q, Y_q, density_q = compute_kde(samples_Y)

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X_p, Y_p, density_p, cmap='viridis')
ax.set_title('Density plot for p ~ N([0,0], I)')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Density')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X_q, Y_q, density_q, cmap='viridis')
ax.set_title('Density plot for q ~ N([2,2], I)')
ax.set_xlabel('Y1')
ax.set_ylabel('Y2')
ax.set_zlabel('Density')
plt.tight_layout()
plt.show()



@jit
def count_equal_within_tolerance(samples_X, samples_Y, tolerance=1e-5):
    return jnp.sum(jnp.abs(samples_X - samples_Y) <= tolerance, axis=1)

tolerance = 1e-5
equal_count = count_equal_within_tolerance(samples_X, samples_Y, tolerance)
P_X_equals_Y = jnp.mean(equal_count)

print(f"For C=1 , Estimated P(X = Y) with tolerance = {tolerance}: {P_X_equals_Y}")

num_samples = 1000
C_values = jnp.linspace(0.01, 0.99, 20)
P_X_equals_Y_estimates = []

for C in tqdm.tqdm(C_values):
    equal_count = 0
    for _ in range(num_samples):
        key, subkey = random.split(key)
        X, Y = ThorissonCoupling(sample_multivariate_normal, sample_multivariate_normal,
                                 multivariate_normal_pdf, multivariate_normal_pdf,
                                 C, mean_p, cov_p, mean_q, cov_q, subkey)
        
        if jnp.linalg.norm(X - Y) <= tolerance:
            equal_count += 1
    
    P_X_equals_Y_estimates.append(equal_count / num_samples)

pd.DataFrame(P_X_equals_Y_estimates).to_feather(path+"c_coupling.feather")


#EXPERIMENT 4
num_samples = 500
num_runs = 10  
med_running_times = []

def run_simulation(C, mean_p, cov_p, mean_q, cov_q, key):
    for _ in range(num_samples):
        key, subkey = random.split(key)
        X, Y = ThorissonCoupling(sample_multivariate_normal, sample_multivariate_normal,
                                 multivariate_normal_pdf, multivariate_normal_pdf,
                                 C, mean_p, cov_p, mean_q, cov_q, subkey)
    return key  
  
run_times = []

for C in tqdm.tqdm(C_values):
    for _ in range(num_runs):
        start_time = time.time() 
        # Run the simulation
        key = run_simulation(C, mean_p, cov_p, mean_q, cov_q, key)
        end_time = time.time()  
        running_time = end_time - start_time  
        run_times.append(running_time)
    med_running_times.append(jnp.median(jnp.array(run_times)))  

pd.DataFrame(run_times).to_feather(path+"run_times_exp1.feather")

