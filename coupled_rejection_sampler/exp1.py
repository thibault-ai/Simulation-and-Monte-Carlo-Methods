from scripts.all_functions import *

#EXPERIMENT 1 -----------------------------------------------------------------
def experiment():
    
    JAX_KEY = jax.random.PRNGKey(15316)
    
    # Simulation parameters
    mu = 6.0
    M = 100_000
    DELTAS = np.linspace(1e-6, 0.5, num=200)
    
    # Defining the probability density function for the truncated distribution
    p = lambda x: truncnorm.pdf(x, mu, np.inf)
    
    # Initializing arrays to store the results
    pxy_list = np.empty((len(DELTAS), 2))
    x_samples = np.empty((len(DELTAS), M))
    y_samples = np.empty((len(DELTAS), M))
    runtimes = np.empty((len(DELTAS),))
    
    # Splitting the JAX key for random number generation
    keys = jax.random.split(JAX_KEY, M)
    
    # Compiling the simulation function for fast execution
    sampler = jax.jit(jax.vmap(coupled_gaussian_tails, in_axes=[0, None, None]))
    
    # Looping over delta values to get results for different eta values
    for n in tqdm.trange(len(DELTAS)):
        delta = DELTAS[n]
        eta = mu + delta
        
        # Defining the probability density function for the second truncated distribution
        q = lambda x: truncnorm.pdf(x, eta, np.inf)
        
        # Calculating the theoretical probability of coupling between the two distributions
        mpq = lambda x: np.minimum(p(x), q(x))
        true_pxy = scipy.integrate.quad(mpq, 0, np.inf)[0]
        
        # Simulating samples and measuring execution time
        tic = time.time()
        x_samples[n], y_samples[n], acc = sampler(keys, mu, eta)
        toc = time.time()
        runtimes[n] = toc - tic
        
        # Estimating the empirical probability of coupling from the samples
        pxy = np.mean(acc)
        pxy_list[n] = pxy, true_pxy
    
    np.savez("results/gauss_tails/gauss_tails.npz", x_samples=x_samples, y_samples=y_samples, pxy_list=pxy_list, M=M, etas=mu + DELTAS, mu=mu, runtimes=runtimes)
    
    return x_samples, y_samples, pxy_list, runtimes

def to_plot():
    data = np.load("results/gauss_tails/gauss_tails.npz")
    
    # Converting the data to DataFrame and saving to feather files
    for k in list(data.keys()):
        if k != "M" and k != "mu":
            df = pd.DataFrame(data[k])
            pyarrow.feather.write_feather(df, "results/gauss_tails/" + k + ".feather")

# Running the simulation
x_samples, y_samples, pxy_list, runtimes = experiment()

# Converting the data and saving the results to feather files
to_plot()
