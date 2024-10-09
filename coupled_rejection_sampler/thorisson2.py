#EXPERIMENT 5
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
from scipy.integrate import quad
import random
import tqdm
import pandas as pd
path = "results/thorisson/"

def s_truncated_normal(mu, sig, lower_bound):
    a = (lower_bound - mu) / sig
    b = np.inf  # Pas lim supérieure

    return truncnorm(a, b, loc=mu, scale=sig).rvs()


def truncated_normalpdf(x, mu, sig, lower_bound):

    a = (lower_bound - mu) / sig
    b = np.inf  
    return truncnorm(a, b, loc=mu, scale=sig).pdf(x)


def Thorisson2(p_sample, q_sample, p_pdf, q_pdf, C, lower_bound_p, lower_bound_q):
    X = p_sample(0, 1, lower_bound_p)
    U = random.uniform(0, 1)
    if U < min((q_pdf(X, 2, 1, lower_bound_q) / p_pdf(X, 0, 1, lower_bound_p)), C):
        Y = X
    else:
        Y = None
        while Y is None:
            U = random.uniform(0, 1)
            Z = q_sample(2, 1, lower_bound_q)
            if U > min(1, (C * p_pdf(Z, 0, 1, lower_bound_p) / q_pdf(Z, 2, 1, lower_bound_q))):
                Y = Z
    return X, Y

C = 0.95
a_values = np.arange(6, 6.6, 0.02)
P_X1_Y1_estimates = []
n_samples = 1000
tol= 1e-5
P_X1_Y1_theoretical = []


for a in tqdm.tqdm(a_values):
    count = 0
    for _ in range(n_samples):
        X, Y = Thorisson2(s_truncated_normal, s_truncated_normal, 
                                 truncated_normalpdf, truncated_normalpdf, C, 6, a)
        if np.abs(X - Y) <= tol:
            count += 1
    P_X1_Y1_estimates.append(count / n_samples)


    p = lambda x: truncnorm.pdf(x, (6 - 0) / 1, np.inf, loc=0, scale=np.sqrt(1))
    q = lambda x: truncnorm.pdf(x, (a - 0) / 1, np.inf, loc=0, scale=np.sqrt(1))
    min_pq = lambda x: np.minimum(p(x), q(x))
    true_pxy, _ = quad(min_pq, 6, np.inf)
    P_X1_Y1_theoretical.append(true_pxy)

pd.DataFrame(P_X1_Y1_estimates).to_feather(path+"P_X1_Y1_estimates.feather")
pd.DataFrame(P_X1_Y1_theoretical).to_feather(path+"P_X1_Y1_theoretical.feather")

#EXPERIMENT 6

C = 0.95
nombre_points = 100
inc = (6.5 - 6) / nombre_points
a_values = np.arange(6, 6.5, inc)

P_X1_Y1_estimates = []
total_runtimes = []
n_samples=1000


for a in tqdm.tqdm(a_values):
    start_time = time.time()  # Start timing for runtime
    
    count = 0
    for _ in range(n_samples):
        X, Y = Thorisson2(s_truncated_normal, s_truncated_normal, 
                                 truncated_normalpdf, truncated_normalpdf, C, 6, a)
        if np.abs(X - Y) <= tol:
            count += 1

    end_time = time.time()  # End timing for runtime
    total_runtime = end_time - start_time
    total_runtimes.append(total_runtime)
    
    P_X1_Y1_estimates.append(count / n_samples)  


pd.DataFrame(total_runtimes).to_feather(path+"total_runtimes_exp2.feather")

plt.figure(figsize=(14, 7))


plt.subplot(1, 2, 2)
plt.plot(a_values, total_runtimes, '-o', color='red')
plt.xlabel('a')
plt.ylabel('Total Runtime ')
plt.title('Total Runtime of the Algorithm for Different Values of a')
plt.tight_layout()

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal
import random

def sample_multivariate_normal(mean, cov):
    return multivariate_normal.rvs(mean=mean, cov=cov)


def multivariate_normal_pdf(x, mean, cov):
    return multivariate_normal.pdf(x, mean=mean, cov=cov)


#C=1
def Thorisson3(p_sample, q_sample, p_pdf, q_pdf, mean_p, cov_p, mean_q, cov_q):
    X = p_sample(mean_p, cov_p)
    U = random.uniform(0, 1)
    if U < (q_pdf(X, mean_q, cov_q) / p_pdf(X, mean_p, cov_p)):
        Y = X
        coupling_C = 1

    else:
        A = 0
        coupling_C=0        
        while A != 1:
            U = random.uniform(0, 1)
            Z = q_sample(2, 1)
            if U > min(1, ( p_pdf(Z, 0, 1) / q_pdf(Z, 2, 1))):
                A = 1
                Y = Z
    return X, Y, coupling_C



dimension = range(1, 11)  # dimensions from 1 to 10
num_samples = 50_000  # number of samples per experiment

runtimes = []
coupling_prob = []

for d in tqdm.tqdm(dimension):
    mean_p = np.zeros(d)
    cov_p = 2 * np.identity(d)
    mean_q = np.ones(d)
    cov_q = 3 * np.identity(d)

    couplings_C = 0
    for i in range(num_samples):
        start_time = time.time()
        _, _, is_coupled = Thorisson3(
            sample_multivariate_normal, sample_multivariate_normal,
            multivariate_normal_pdf, multivariate_normal_pdf,mean_p, cov_p, mean_q, cov_q
        )
        couplings_C += is_coupled
        runtime = time.time() - start_time
        runtimes.append(runtime)
    coupling_prob.append(couplings_C / num_samples)

pd.DataFrame(runtimes).to_feather(path+"dim_runtimes.feather")
pd.DataFrame(coupling_prob).to_feather(path+"dim_couplings.feather")



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(dimension, runtimes, marker='o')
plt.xlabel('Number of Dimensions')
plt.ylabel('Runtime')
plt.title('Runtime as a Function of Dimensions')

# Plot Coupling Probability as a function of dimensions
plt.subplot(1, 2, 2)
plt.plot(dimension, coupling_prob, marker='o', color='r')
plt.xlabel('Number of Dimensions')
plt.ylabel('Coupling Probability P(X = Y)')
plt.title('Coupling Probability as a Function of Dimensions')
plt.tight_layout()
plt.show()
