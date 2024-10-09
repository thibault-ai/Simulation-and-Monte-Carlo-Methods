from scripts.all_functions import *

#EXPERIMENT 2
def run_mvn_experiments_with_increasing_dimensions(max_dim=10, N=300):
    # Ensure the directory for results exists
    base_path = "results/gauss_dim/"
    os.makedirs(base_path, exist_ok=True)

    # Iterate over dimensions from 1 to max_dim
    for dim in range(1, max_dim + 1):
        print(f"Running experiment for dimension: {dim}")

        m = jnp.zeros(dim)
        mu = jnp.ones(dim)
        chol_P = jnp.linalg.cholesky(jnp.eye(dim) * 2)
        chol_Sig = jnp.linalg.cholesky(jnp.eye(dim) * 3)
        key = jax.random.PRNGKey(0)

        # Lists to store results for the current dimension
        sample_times = []
        is_coupled = []
        n_trials = []

        # Start timing the experiment for this dimension
        start_time = time.time()

        # Run coupled sampling for the current dimension
        for _ in tqdm.tqdm(range(N)):
            key, subkey = jax.random.split(key)
            sample_start = time.time()
            X, Y, coupled, trials = coupled_mvns(subkey, m, chol_P, mu, chol_Sig, N=1)
            sample_end = time.time() - sample_start

            # Store results for this run
            sample_times.append(sample_end)
            is_coupled.append(bool(coupled))  # Convert JAX bool to Python bool
            n_trials.append(int(trials))      # Convert JAX int to Python int

        # Compute overall runtime for this dimension
        total_runtime = time.time() - start_time

        # Create dataframes for the current dimension
        df_results = pd.DataFrame({
            'sample_times': np.array(sample_times),  # Ensure data is in NumPy array format
            'is_coupled': np.array(is_coupled),
            'n_trials': np.array(n_trials),
            'total_runtime': np.full(N, total_runtime)  # Repeat total runtime for ease of analysis
        })

        # Save to Feather
        feather_path = os.path.join(base_path, f"dim_{dim}.feather")
        df_results.reset_index(drop=True).to_feather(feather_path)

        # Output summary for current dimension
        print(f"Dimension {dim} - Data saved to {feather_path}")
        print(f"Dimension {dim} - Total runtime: {total_runtime:.2f} seconds")


run_mvn_experiments_with_increasing_dimensions()
