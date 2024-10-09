from scripts.all_functions import *

#EXPERIMENT 2
def run_mvn_experiments_with_increasing_dimensions(max_dim=10, N=300):
    base_path = "results/gauss_dim/"
    os.makedirs(base_path, exist_ok=True)

    for dim in range(1, max_dim + 1):
        print(f"Running experiment for dimension: {dim}")

        m = jnp.zeros(dim)
        mu = jnp.ones(dim)
        chol_P = jnp.linalg.cholesky(jnp.eye(dim) * 2)
        chol_Sig = jnp.linalg.cholesky(jnp.eye(dim) * 3)
        key = jax.random.PRNGKey(0)

        sample_times = []
        is_coupled = []
        n_trials = []

        start_time = time.time()

        for _ in tqdm.tqdm(range(N)):
            key, subkey = jax.random.split(key)
            sample_start = time.time()
            X, Y, coupled, trials = coupled_mvns(subkey, m, chol_P, mu, chol_Sig, N=1)
            sample_end = time.time() - sample_start

            sample_times.append(sample_end)
            is_coupled.append(bool(coupled))  # Convert JAX bool to Python bool
            n_trials.append(int(trials))      # Convert JAX int to Python int

        total_runtime = time.time() - start_time

        df_results = pd.DataFrame({
            'sample_times': np.array(sample_times),  # Ensure data is in NumPy array format
            'is_coupled': np.array(is_coupled),
            'n_trials': np.array(n_trials),
            'total_runtime': np.full(N, total_runtime)  # Repeat total runtime for ease of analysis
        })

        feather_path = os.path.join(base_path, f"dim_{dim}.feather")
        df_results.reset_index(drop=True).to_feather(feather_path)

        print(f"Dimension {dim} - Data saved to {feather_path}")
        print(f"Dimension {dim} - Total runtime: {total_runtime:.2f} seconds")


run_mvn_experiments_with_increasing_dimensions()
