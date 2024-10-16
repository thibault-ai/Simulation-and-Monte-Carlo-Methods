import jax.numpy as jnp
import jax.random
jax.config.update('jax_platform_name', 'cpu')
from jax.scipy.special import logsumexp
from scipy.stats import truncnorm
import numpy as np
import pandas as pd
import pyarrow.feather
import tensorflow_probability.substrates.jax as tfp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate
import tikzplotlib
import time
import tqdm

from functools import partial

# Constants
LOG_HALF = jnp.log(0.5)

def log1mexp(x):
    """ Compute log(1-exp(-|x|)) safely """
    return jnp.where(x < LOG_HALF, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))

@partial(jnp.vectorize, signature="(),()->()")
def logsubexp(x1, x2):
    """ Subtract exponential logs: log(exp(x1) - exp(x2)) """
    amax = jnp.maximum(x1, x2)
    delta = jnp.abs(x1 - x2)
    return amax + log1mexp(-delta)

def get_alpha(mu):
    """ Compute the optimal alpha as per Robert (1995) """
    return 0.5 * (mu + jnp.sqrt(mu ** 2 + 4))

def coupled_sampler(key, Gamma_hat, p, q, log_p_hat, log_q_hat, log_p, log_q, log_M_p, log_M_q, N=1):
    """
    Coupled rejection sampling algorithm.
    """
    def _accept_proposal_and_acceptance_ratio(op_key, Xs_hat, Ys_hat, are_coupled):
        select_key, accept_key = jax.random.split(op_key, 2)
        log_w_X = log_p(Xs_hat) - log_p_hat(Xs_hat)
        log_w_Y = log_q(Ys_hat) - log_q_hat(Ys_hat)
        X_hat, Y_hat = Xs_hat[0], Ys_hat[0]
        coupled_proposal = are_coupled[0]
        X_acceptance_proba = log_w_X[0] - log_M_p
        Y_acceptance_proba = log_w_Y[0] - log_M_q
        log_u = jnp.log(jax.random.uniform(accept_key))
        accept_X = log_u < X_acceptance_proba
        accept_Y = log_u < Y_acceptance_proba
        return accept_X, accept_Y, X_hat, Y_hat, coupled_proposal

    def cond(carry):
        accept_X, accept_Y, *_ = carry
        return ~accept_X & ~accept_Y

    def body(carry):
        *_, i, curr_key = carry
        next_key, sample_key, accept_key = jax.random.split(curr_key, 3)
        Xs_hat, Ys_hat, are_coupled = Gamma_hat(sample_key, N)
        accept_X, accept_Y, X_hat, Y_hat, coupled_proposal = _accept_proposal_and_acceptance_ratio(
            accept_key, Xs_hat, Ys_hat, are_coupled)
        return accept_X, accept_Y, X_hat, Y_hat, coupled_proposal, i + 1, next_key

    init_key, key = jax.random.split(key)
    X_init = p(init_key)
    Y_init = q(init_key)
    output = jax.lax.while_loop(cond, body, (False, False, X_init, Y_init, False, 0, key))
    is_X_accepted, is_Y_accepted, X, Y, is_coupled, n_trials, _ = output
    X = jax.lax.select(is_X_accepted, X, X_init)
    Y = jax.lax.select(is_Y_accepted, Y, Y_init)
    is_coupled = is_coupled & is_X_accepted & is_Y_accepted
    return X, Y, is_coupled, n_trials

def coupled_gaussian_tails(key, mu, eta):
    """
    Coupled Gaussian tail sampling.
    """
    alpha_mu = get_alpha(mu)
    alpha_eta = get_alpha(eta)
    p = lambda k: _robert_sampler(k, mu, alpha_mu)
    q = lambda k: _robert_sampler(k, eta, alpha_eta)
    log_w_p = lambda x: -0.5 * (x - alpha_mu) ** 2
    log_w_q = lambda x: -0.5 * (x - alpha_eta) ** 2
    def body(carry):
        *_, i, curr_key = carry
        next_key, sample_key, accept_key = jax.random.split(curr_key, 3)
        X_hat, Y_hat, are_coupled = coupled_exponentials(sample_key, mu, alpha_mu, eta, alpha_eta)
        log_w_X = log_w_p(X_hat)
        log_w_Y = log_w_q(Y_hat)
        log_u = jnp.log(jax.random.uniform(accept_key))
        accept_X = log_u < log_w_X
        accept_Y = log_u < log_w_Y
        return accept_X, accept_Y, X_hat, Y_hat, are_coupled, i + 1, next_key

    def cond(carry):
        accept_X, accept_Y, *_ = carry
        return ~accept_X & ~accept_Y

    residual_key, loop_key = jax.random.split(key)
    output = jax.lax.while_loop(cond, body, (False, False, 0., 0., False, 0, loop_key))
    is_X_accepted, is_Y_accepted, X, Y, is_coupled, n_trials, _ = output
    X = jax.lax.cond(is_X_accepted, lambda _: X, p, residual_key)
    Y = jax.lax.cond(is_Y_accepted, lambda _: Y, q, residual_key)
    is_coupled = is_coupled & is_X_accepted & is_Y_accepted
    return X, Y, is_coupled


def _sampled_from_coupled_exponentials(key, mu, _eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    def C1_inv(log_u):
        return mu - logsubexp(eta_mu, log_u + logsubexp(eta_mu, gamma_mu)) / alpha_mu

    def C2_inv(log_u):
        return gamma - log_u / alpha_eta

    log_p1 = logsubexp(eta_mu, gamma_mu)
    log_p2 = gamma_eta
    log_p = log_p1 - jnp.logaddexp(log_p1, log_p2)

    log_u1, log_u2 = jnp.log(jax.random.uniform(key, shape=(2,)))

    res = jax.lax.cond(log_u1 < log_p, C1_inv, C2_inv, log_u2)
    return res

def _sample_from_first_marginal(key, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    key1, key2 = jax.random.split(key, 2)
    log_u1 = jnp.log(jax.random.uniform(key1))

    log_p1 = logsubexp(gamma_mu, gamma_eta)  # This has the same value as $\log(\tilde{Z})$
    log_p2 = log1mexp(eta_mu)
    log_p = log_p1 - jnp.logaddexp(log_p1, log_p2)

    def _sample_from_tail(log_u):
        return mu - log1mexp(log_u + log1mexp(eta_mu)) / alpha_mu

    def _sample_from_overlap(log_u):

        def log_f(x):
            return logsubexp(-alpha_mu * (x - mu), -alpha_eta * (x - eta)) - log_p1 - log_u

        # upper bound for the solution is given by a lower bounding of the density
        def upper_bound_loop(carry):

            curr_upper_bound, _ = carry
            curr_upper_bound = 1.5 * curr_upper_bound
            obj = log_f(curr_upper_bound)
            return curr_upper_bound, obj >= 0

        upper_bound, _ = jax.lax.while_loop(lambda carry: carry[-1], upper_bound_loop, (gamma, True))

        res, objective_at_estimated_root, *_ = tfp.math.find_root_chandrupatla(log_f, gamma, upper_bound,
                                                                               position_tolerance=1e-6,
                                                                               value_tolerance=1e-6)

        return res

    return jax.lax.cond(log_u1 < log_p, _sample_from_overlap, _sample_from_tail, jnp.log(jax.random.uniform(key2)))


def _sample_from_second_marginal(key, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    log_Zq_1 = jnp.logaddexp(0, gamma_mu)  # log(1 + exp(gamma_mu))
    log_Zq_2 = jnp.logaddexp(eta_mu, gamma_eta)
    log_Zq = logsubexp(log_Zq_1, log_Zq_2)
    log_u = jnp.log(jax.random.uniform(key))

    def log_f(x):
        res_1 = jnp.logaddexp(0, -alpha_mu * (x - mu))  # log(1 + exp(...))
        res_2 = jnp.logaddexp(eta_mu, -alpha_eta * (x - eta))
        res = logsubexp(res_1, res_2)
        res = res - log_Zq - log_u
        return res

    out, objective_at_estimated_root, *_ = tfp.math.find_root_chandrupatla(log_f, eta, gamma, position_tolerance=1e-6,
                                                                           value_tolerance=1e-6)

    return out


def coupled_exponentials(key, mu, alpha_mu, eta, alpha_eta):
    """
    Sampling from coupled exponentials.
    """
    gamma = (jnp.log(alpha_eta) - jnp.log(alpha_mu) + alpha_eta * eta - alpha_mu * mu) / (alpha_eta - alpha_mu)
    eta_mu = -alpha_mu * (eta - mu)
    gamma_mu = -alpha_mu * (gamma - mu)
    gamma_eta = -alpha_eta * (gamma - eta)
    log_max_coupling_proba = logsubexp(eta_mu, jnp.logaddexp(gamma_mu, gamma_eta))
    subkey1, subkey2 = jax.random.split(key)
    log_u = jnp.log(jax.random.uniform(subkey1, shape=()))
    are_coupled = (log_u <= log_max_coupling_proba)
    def if_coupled(k):
        x = _sampled_from_coupled_exponentials(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        return x, x
    def otherwise(k):
        x = _sample_from_first_marginal(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        y = _sample_from_second_marginal(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        return x, y
    x_out, y_out = jax.lax.cond(are_coupled, if_coupled, otherwise, subkey2)
    return x_out, y_out, are_coupled

def _robert_sampler(key, mu, alpha):
    """
    Robert's truncated normal sampler.
    """
    def body(carry):
        curr_k, *_ = carry
        curr_k, subkey = jax.random.split(curr_k, 2)
        u1, u2 = jax.random.uniform(subkey, shape=(2,))
        x = mu - jnp.log(1 - u1) / alpha
        accepted = u2 <= jnp.exp(-0.5 * (x - alpha) ** 2)
        return curr_k, x, accepted

    _, x_out, _ = jax.lax.while_loop(lambda carry: ~carry[-1], body, (key, 0., False))
    return x_out

def experiment():
    """
    Run the simulation.
    """
    JAX_KEY = jax.random.PRNGKey(0)
    mu = 6.0
    M = 100_000
    DELTAS = np.linspace(1e-6, 0.5, num=200)
    p = lambda x: truncnorm.pdf(x, mu, np.inf)
    pxy_list = np.empty((len(DELTAS), 2))
    x_samples = np.empty((len(DELTAS), M))
    y_samples = np.empty((len(DELTAS), M))
    runtimes = np.empty((len(DELTAS),))
    keys = jax.random.split(JAX_KEY, M)
    sampler = jax.jit(jax.vmap(coupled_gaussian_tails, in_axes=[0, None, None]))
    for n in tqdm.trange(len(DELTAS)):
        delta = DELTAS[n]
        eta = mu + delta
        q = lambda x: truncnorm.pdf(x, eta, np.inf)
        mpq = lambda x: np.minimum(p(x), q(x))
        true_pxy = scipy.integrate.quad(mpq, 0, np.inf)[0]
        tic = time.time()
        x_samples[n], y_samples[n], acc = sampler(keys, mu, eta)
        toc = time.time()
        runtimes[n] = toc - tic
        pxy = np.mean(acc)
        pxy_list[n] = pxy, true_pxy
    
    df=pd.DataFrame(runtimes)
    pyarrow.feather.write_feather(df,"results/runtimes_cpu.feather")
    return x_samples, y_samples, pxy_list, runtimes


if __name__ == "__main__":
    if True:  # RUN
        x_samples, y_samples, pxy_list, runtimes = experiment()
