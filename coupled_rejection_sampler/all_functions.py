import jax.numpy as jnp
import jax.random
from jax.scipy.special import logsumexp
from scipy.stats import truncnorm
import numpy as np
import pandas as pd
import pyarrow.feather
import tensorflow_probability.substrates.jax as tfp
import scipy.integrate
import time
import tqdm
import os
from functools import partial
import jax.scipy.linalg as jlinalg
import jax.scipy.stats as jstats
from jax.scipy.linalg import cho_solve
import math
from functools import partial

# Constants
LOG_HALF = jnp.log(0.5)
_LOG_2PI = math.log(2 * math.pi)


def log1mexp(x):
    """ Compute log(1-exp(-|x|)) safely """
    return jnp.where(x < LOG_HALF, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))

@partial(jnp.vectorize, signature="(),()->()")
def logsubexp(x1, x2):
    """ Subtract exponential logs: log(exp(x1) - exp(x2)) """
    amax = jnp.maximum(x1, x2)
    delta = jnp.abs(x1 - x2)
    return amax + log1mexp(-delta)


def log1mexp(x):
    """
    Computes the logarithm of 1 minus the exponential of x.
    """
    return jnp.where(x < LOG_HALF, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))

@partial(jnp.vectorize, signature="(),()->()")
def logsubexp(x1, x2):
    """
    Computes the logarithm of the subtraction of two numbers in a numerically stable way.
    """
    amax = jnp.maximum(x1, x2)
    delta = jnp.abs(x1 - x2)
    return amax + log1mexp(-delta)

def get_alpha(mu):
    """ Compute the optimal alpha as per Robert (1995) """
    return 0.5 * (mu + jnp.sqrt(mu ** 2 + 4))

def coupled_sampler(key, Gamma_hat, p, q, log_p_hat, log_q_hat, log_p, log_q, log_M_p, log_M_q, N=1):
    """
    Coupled rejection sampling algorithm.
    
    Args:
        key: JAX PRNGKey for random number generation.
        Gamma_hat: Function to generate coupled proposals.
        p: Probability density function of the first distribution.
        q: Probability density function of the second distribution.
        log_p_hat: Log probability density function approximation for the first distribution.
        log_q_hat: Log probability density function approximation for the second distribution.
        log_p: Log probability density function for the first distribution.
        log_q: Log probability density function for the second distribution.
        log_M_p: Log scaling factor for the first distribution.
        log_M_q: Log scaling factor for the second distribution.
        N: Number of samples to generate (default is 1).
        
    Returns:
        X: Samples from the first distribution.
        Y: Samples from the second distribution.
        is_coupled: Indicator of whether the samples are coupled.
        n_trials: Number of rejection sampling trials performed.
    """
    def _accept_proposal_and_acceptance_ratio(op_key, Xs_hat, Ys_hat, are_coupled):
        # Splitting the key for random number generation
        select_key, accept_key = jax.random.split(op_key, 2)
        
        # Compute the log weights and select the first proposal
        log_w_X = log_p(Xs_hat) - log_p_hat(Xs_hat)
        log_w_Y = log_q(Ys_hat) - log_q_hat(Ys_hat)
        X_hat, Y_hat = Xs_hat[0], Ys_hat[0]
        coupled_proposal = are_coupled[0]
        
        # Compute acceptance probabilities
        X_acceptance_proba = log_w_X[0] - log_M_p
        Y_acceptance_proba = log_w_Y[0] - log_M_q
        
        # Generate uniform random numbers
        log_u = jnp.log(jax.random.uniform(accept_key))
        
        # Check if proposals are accepted
        accept_X = log_u < X_acceptance_proba
        accept_Y = log_u < Y_acceptance_proba
        
        return accept_X, accept_Y, X_hat, Y_hat, coupled_proposal

    def cond(carry):
        # Continue looping if neither proposal is accepted
        accept_X, accept_Y, *_ = carry
        return ~accept_X & ~accept_Y

    def body(carry):
        # Body of the while loop
        *_, i, curr_key = carry
        next_key, sample_key, accept_key = jax.random.split(curr_key, 3)
        
        # Generate coupled proposals
        Xs_hat, Ys_hat, are_coupled = Gamma_hat(sample_key, N)
        
        # Accept proposals and compute acceptance ratio
        accept_X, accept_Y, X_hat, Y_hat, coupled_proposal = _accept_proposal_and_acceptance_ratio(
            accept_key, Xs_hat, Ys_hat, are_coupled)
        
        return accept_X, accept_Y, X_hat, Y_hat, coupled_proposal, i + 1, next_key

    # Initialize random key and proposals
    init_key, key = jax.random.split(key)
    X_init = p(init_key)
    Y_init = q(init_key)
    
    # Perform rejection sampling
    output = jax.lax.while_loop(cond, body, (False, False, X_init, Y_init, False, 0, key))
    
    # Retrieve outputs
    is_X_accepted, is_Y_accepted, X, Y, is_coupled, n_trials, _ = output
    
    # Select initial values if proposals are rejected
    X = jax.lax.select(is_X_accepted, X, X_init)
    Y = jax.lax.select(is_Y_accepted, Y, Y_init)
    
    # Update coupling indicator
    is_coupled = is_coupled & is_X_accepted & is_Y_accepted
    
    return X, Y, is_coupled, n_trials

def coupled_gaussian_tails(key, mu, eta):
    """
    Coupled Gaussian tail sampling.

    Args:
        key: JAX PRNGKey for random number generation.
        mu: Mean of the first Gaussian distribution.
        eta: Mean of the second Gaussian distribution.

    Returns:
        X: Sample from the first Gaussian distribution.
        Y: Sample from the second Gaussian distribution.
        is_coupled: Indicator of whether the samples are coupled.
    """
    # Compute the alpha values for the distributions
    alpha_mu = get_alpha(mu)
    alpha_eta = get_alpha(eta)
    
    # Define the probability density functions for rejection sampling
    p = lambda k: _robert_sampler(k, mu, alpha_mu)
    q = lambda k: _robert_sampler(k, eta, alpha_eta)
    
    # Define the log weights for acceptance probabilities
    log_w_p = lambda x: -0.5 * (x - alpha_mu) ** 2
    log_w_q = lambda x: -0.5 * (x - alpha_eta) ** 2
    
    # Define the body of the rejection sampling loop
    def body(carry):
        *_, i, curr_key = carry
        next_key, sample_key, accept_key = jax.random.split(curr_key, 3)
        
        # Generate coupled proposals from exponential distributions
        X_hat, Y_hat, are_coupled = coupled_exponentials(sample_key, mu, alpha_mu, eta, alpha_eta)
        
        # Compute log weights and generate uniform random numbers
        log_w_X = log_w_p(X_hat)
        log_w_Y = log_w_q(Y_hat)
        log_u = jnp.log(jax.random.uniform(accept_key))
        
        # Check if proposals are accepted
        accept_X = log_u < log_w_X
        accept_Y = log_u < log_w_Y
        
        return accept_X, accept_Y, X_hat, Y_hat, are_coupled, i + 1, next_key

    # Define the termination condition for the rejection sampling loop
    def cond(carry):
        accept_X, accept_Y, *_ = carry
        return ~accept_X & ~accept_Y

    # Split the PRNGKey for rejection sampling
    residual_key, loop_key = jax.random.split(key)
    
    # Perform rejection sampling loop
    output = jax.lax.while_loop(cond, body, (False, False, 0., 0., False, 0, loop_key))
    is_X_accepted, is_Y_accepted, X, Y, is_coupled, n_trials, _ = output
    
    # Conditional assignment for accepted samples
    X = jax.lax.cond(is_X_accepted, lambda _: X, p, residual_key)
    Y = jax.lax.cond(is_Y_accepted, lambda _: Y, q, residual_key)
    
    # Update coupling indicator
    is_coupled = is_coupled & is_X_accepted & is_Y_accepted
    
    return X, Y, is_coupled

def _sampled_from_coupled_exponentials(key, mu, _eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    """
    Sampled from coupled exponential distributions.

    Args:
        key: JAX PRNGKey for random number generation.
        mu: Mean of the first distribution.
        _eta: Not used.
        alpha_mu: Alpha value for the first distribution.
        alpha_eta: Alpha value for the second distribution.
        eta_mu: Mu value for the first distribution.
        gamma_mu: Gamma value for the first distribution.
        gamma_eta: Gamma value for the second distribution.
        gamma: Gamma value.

    Returns:
        Sampled value from coupled exponential distributions.
    """
    def C1_inv(log_u):
        # Inverse of the first cumulative distribution function
        return mu - logsubexp(eta_mu, log_u + logsubexp(eta_mu, gamma_mu)) / alpha_mu

    def C2_inv(log_u):
        # Inverse of the second cumulative distribution function
        return gamma - log_u / alpha_eta

    # Compute the log probability of choosing the first distribution
    log_p1 = logsubexp(eta_mu, gamma_mu)
    log_p2 = gamma_eta
    log_p = log_p1 - jnp.logaddexp(log_p1, log_p2)

    # Generate two uniform random numbers
    log_u1, log_u2 = jnp.log(jax.random.uniform(key, shape=(2,)))

    # Conditionally choose which cumulative distribution function to use based on the log probabilities
    res = jax.lax.cond(log_u1 < log_p, C1_inv, C2_inv, log_u2)
    return res

def _sample_from_first_marginal(key, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    """
    Sample from the first marginal distribution.

    Args:
        key: JAX PRNGKey for random number generation.
        mu: Mean of the first distribution.
        eta: Mean of the second distribution.
        alpha_mu: Alpha value for the first distribution.
        alpha_eta: Alpha value for the second distribution.
        eta_mu: Mu value for the first distribution.
        gamma_mu: Gamma value for the first distribution.
        gamma_eta: Gamma value for the second distribution.
        gamma: Gamma value.

    Returns:
        Sampled value from the first marginal distribution.
    """
    key1, key2 = jax.random.split(key, 2)
    log_u1 = jnp.log(jax.random.uniform(key1))

    # Compute the log probability of choosing the tail distribution
    log_p1 = logsubexp(gamma_mu, gamma_eta)  # This has the same value as $\log(\tilde{Z})$
    log_p2 = log1mexp(eta_mu)
    log_p = log_p1 - jnp.logaddexp(log_p1, log_p2)

    def _sample_from_tail(log_u):
        # Sample from the tail distribution
        return mu - log1mexp(log_u + log1mexp(eta_mu)) / alpha_mu

    def _sample_from_overlap(log_u):
        # Sample from the overlapping region
        def log_f(x):
            return logsubexp(-alpha_mu * (x - mu), -alpha_eta * (x - eta)) - log_p1 - log_u

        # Compute an upper bound for finding the root
        def upper_bound_loop(carry):
            curr_upper_bound, _ = carry
            curr_upper_bound = 1.5 * curr_upper_bound
            obj = log_f(curr_upper_bound)
            return curr_upper_bound, obj >= 0

        upper_bound, _ = jax.lax.while_loop(lambda carry: carry[-1], upper_bound_loop, (gamma, True))

        # Use root finding to find the sample in the overlap region
        res, objective_at_estimated_root, *_ = tfp.math.find_root_chandrupatla(log_f, gamma, upper_bound,
                                                                               position_tolerance=1e-6,
                                                                               value_tolerance=1e-6)

        return res

    # Conditionally choose which sampling method to use based on the log probabilities
    return jax.lax.cond(log_u1 < log_p, _sample_from_overlap, _sample_from_tail, jnp.log(jax.random.uniform(key2)))

def _sample_from_second_marginal(key, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    """
    Sample from the second marginal distribution.

    Args:
        key: JAX PRNGKey for random number generation.
        mu: Mean of the first distribution.
        eta: Mean of the second distribution.
        alpha_mu: Alpha value for the first distribution.
        alpha_eta: Alpha value for the second distribution.
        eta_mu: Mu value for the first distribution.
        gamma_mu: Gamma value for the first distribution.
        gamma_eta: Gamma value for the second distribution.
        gamma: Gamma value.

    Returns:
        Sampled value from the second marginal distribution.
    """
    # Compute the log partition function of the second marginal distribution
    log_Zq_1 = jnp.logaddexp(0, gamma_mu)  # log(1 + exp(gamma_mu))
    log_Zq_2 = jnp.logaddexp(eta_mu, gamma_eta)
    log_Zq = logsubexp(log_Zq_1, log_Zq_2)

    # Generate a uniform random number
    log_u = jnp.log(jax.random.uniform(key))

    def log_f(x):
        # Compute the objective function for root finding
        res_1 = jnp.logaddexp(0, -alpha_mu * (x - mu))  # log(1 + exp(...))
        res_2 = jnp.logaddexp(eta_mu, -alpha_eta * (x - eta))
        res = logsubexp(res_1, res_2)
        res = res - log_Zq - log_u
        return res

    # Use root finding to find the sample from the second marginal distribution
    out, objective_at_estimated_root, *_ = tfp.math.find_root_chandrupatla(log_f, eta, gamma, position_tolerance=1e-6,
                                                                           value_tolerance=1e-6)

    return out


def coupled_exponentials(key, mu, alpha_mu, eta, alpha_eta):
    """
    Sampling from coupled exponentials.

    Args:
        key: JAX PRNGKey for random number generation.
        mu: Mean of the first distribution.
        alpha_mu: Alpha value for the first distribution.
        eta: Mean of the second distribution.
        alpha_eta: Alpha value for the second distribution.

    Returns:
        x_out: Sample from the first distribution.
        y_out: Sample from the second distribution.
        are_coupled: Indicator of whether the samples are coupled.
    """
    # Compute the gamma and mu values for the coupled distributions
    gamma = (jnp.log(alpha_eta) - jnp.log(alpha_mu) + alpha_eta * eta - alpha_mu * mu) / (alpha_eta - alpha_mu)
    eta_mu = -alpha_mu * (eta - mu)
    gamma_mu = -alpha_mu * (gamma - mu)
    gamma_eta = -alpha_eta * (gamma - eta)

    # Compute the log maximum coupling probability
    log_max_coupling_proba = logsubexp(eta_mu, jnp.logaddexp(gamma_mu, gamma_eta))

    # Generate a random number to determine if the distributions are coupled
    subkey1, subkey2 = jax.random.split(key)
    log_u = jnp.log(jax.random.uniform(subkey1, shape=()))
    are_coupled = (log_u <= log_max_coupling_proba)

    def if_coupled(k):
        # If coupled, sample from the coupled exponentials
        x = _sampled_from_coupled_exponentials(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        return x, x

    def otherwise(k):
        # If not coupled, sample from the marginals separately
        x = _sample_from_first_marginal(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        y = _sample_from_second_marginal(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        return x, y

    # Conditionally sample from the coupled exponentials or from the marginals
    x_out, y_out = jax.lax.cond(are_coupled, if_coupled, otherwise, subkey2)
    return x_out, y_out, are_coupled


def _robert_sampler(key, mu, alpha):
    """
    Robert's truncated normal sampler.

    Args:
        key: JAX PRNGKey for random number generation.
        mu: Mean of the distribution.
        alpha: Alpha value for the distribution.

    Returns:
        Sampled value from the truncated normal distribution.
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

def get_optimal_covariance(chol_P, chol_Sig):
    """
    Get the optimal covariance according to the objective defined in Section 3 of [1].

    The notations roughly follow the ones in the article.

    Parameters
    ----------
    chol_P: jnp.ndarray
        Square root of the covariance of X. Lower triangular.
    chol_Sig: jnp.ndarray
        Square root of the covariance of Y. Lower triangular.
    Returns
    -------
    chol_Q: jnp.ndarray
        Cholesky of the resulting dominating matrix.
    """
    # Get the dimensionality of the covariance matrices
    d = chol_P.shape[0]
    
    # If the dimension is 1, return the element-wise maximum of the two covariances
    if d == 1:
        return jnp.maximum(chol_P, chol_Sig)

    # Otherwise, perform the optimization steps
    right_Y = jlinalg.solve_triangular(chol_P, chol_Sig, lower=True)  # Y = RY.T RY
    
    # Eigen decomposition of Y^T Y
    w_Y, v_Y = jlinalg.eigh(right_Y.T @ right_Y)
    
    # Ensure eigenvalues are bounded between 0 and 1
    w_Y = jnp.minimum(w_Y, 1)
    
    # Compute the inverse square root of the eigenvalues
    i_w_Y = 1. / jnp.sqrt(w_Y)

    # Construct the left part of the optimal covariance matrix
    left_Q = chol_Sig @ (v_Y * i_w_Y[None, :])
    
    # Compute the Cholesky decomposition of the matrix product of left_Q and its transpose
    return jlinalg.cholesky(left_Q @ left_Q.T, lower=True)

def coupled_mvns(key, m, chol_P, mu, chol_Sig, N=1, chol_Q=None):
    """
    Get the optimal covariance according to the objective defined in Section 3 of [1].

    Parameters
    ----------
    key: jnp.ndarray
        JAX random key
    m: array_like
        Mean of X
    chol_P: array_like
        Square root of the covariance of X. Lower triangular.
    mu: array_like
        Mean of Y
    chol_Sig: array_like
        Square root of the covariance of Y. Lower triangular.
    N: int
        Number of samples used in the underlying coupling rejection sampler
    chol_Q: jnp.ndarray, optional
        Square root of the resulting dominating matrix. Default uses get_optimal_covariance.

    Returns
    -------
    X: jnp.ndarray
        The resulting sample for p
    Y: jnp.ndarray
        The resulting sampled for q
    is_coupled: bool
        Do we have X = Y? Note that if the distributions are not continuous this may be False even if X=Y.
    n_trials: int
        The number of trials before acceptance
    """

    # Compute the optimal covariance matrix if not provided
    if chol_Q is None:
        chol_Q = get_optimal_covariance(chol_P, chol_Sig)
    
    # Compute log determinants of the covariance matrices
    log_det_chol_P = tril_log_det(chol_P)
    log_det_chol_Sig = tril_log_det(chol_Sig)
    log_det_chol_Q = tril_log_det(chol_Q)

    # Compute log-M terms for the rejection sampling algorithm
    log_M_P_Q = jnp.maximum(log_det_chol_Q - log_det_chol_P, 0.)
    log_M_Sigma_Q = jnp.maximum(log_det_chol_Q - log_det_chol_Sig, 0.)

    # Define the reflection function for coupling
    Gamma_hat = partial(reflection_maximal, m=m, mu=mu, chol_Q=chol_Q)
    
    # Define the log probability functions for p and q and for the dominating coupling
    log_p = lambda x: mvn_logpdf(x, m, chol_P)
    log_q = lambda x: mvn_logpdf(x, mu, chol_Sig)
    log_p_hat = lambda x: mvn_logpdf(x, m, chol_Q)
    log_q_hat = lambda x: mvn_logpdf(x, mu, chol_Q)
    
    # Define the samplers for p and q
    p = lambda k: mvn_sampler(k, 1, m, chol_P)[0]
    q = lambda k: mvn_sampler(k, 1, mu, chol_Sig)[0]

    # Use coupled sampler to get the samples
    return coupled_sampler(key, Gamma_hat, p, q, log_p_hat, log_q_hat, log_p, log_q, log_M_P_Q, log_M_Sigma_Q, N)


@partial(jnp.vectorize, signature="(d),(d),(d,d)->()")
def mvn_logpdf(x, m, chol_P):
    # Calculate dimensionality
    d = m.shape[0]
    # Compute the log determinant of the lower triangular matrix chol_P
    log_det_chol_P = tril_log_det(chol_P)
    # Compute constant term for log PDF calculation
    const = -0.5 * d * _LOG_2PI - log_det_chol_P
    # Compute the scaled difference vector
    scaled_diff = jlinalg.solve_triangular(chol_P, x - m, lower=True)
    # Compute the log PDF
    return const - 0.5 * jnp.dot(scaled_diff, scaled_diff)

def mvn_sampler(key, N, m, chol_P):
    # Sample from multivariate normal distribution
    d = m.shape[0]
    # Generate standard normal samples
    eps = jax.random.normal(key, (N, d))
    # Transform to the desired distribution
    return m[None, :] + eps @ chol_P.T

def tril_log_det(chol):
    # Compute the log determinant of the lower triangular matrix chol
    return jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))

def reflection_maximal(key, N, m: jnp.ndarray, mu: jnp.ndarray, chol_Q: jnp.ndarray):
    """
    Reflection maximal coupling for Gaussians with the same covariance matrix

    Parameters
    ----------
    key: jnp.ndarray
        The random key for JAX
    N: int
        Number of samples required
    m:
    mu
    chol_Q

    Returns
    -------
    """

    dim = m.shape[0]
    # Calculate the vector from m to mu in the Q space
    z = jlinalg.solve_triangular(chol_Q, m - mu, lower=True)
    # Compute the unit vector in the direction of z
    e = z / jnp.linalg.norm(z)

    # Generate random numbers for sampling
    normal_key, uniform_key = jax.random.split(key, 2)
    # Generate N samples from a standard normal distribution
    norm = jax.random.normal(normal_key, (N, dim))
    # Generate N uniform random numbers for acceptance decisions
    log_u = jnp.log(jax.random.uniform(uniform_key, (N,)))

    # Calculate the acceptance condition
    temp = norm + z[None, :]
    mvn_loglikelihood = lambda x: - 0.5 * jnp.sum(x ** 2, -1)
    do_accept = log_u + mvn_loglikelihood(norm) < mvn_loglikelihood(temp)

    # Reflect samples that are not accepted
    reflected_norm = jnp.where(do_accept[:, None], temp, norm - 2 * jnp.outer(jnp.dot(norm, e), e))

    # Transform the samples back to the original space
    res_1 = m[None, :] + norm @ chol_Q.T
    res_2 = mu[None, :] + reflected_norm @ chol_Q.T

    return res_1, res_2, do_accept

def lower_bound(m: jnp.ndarray, chol_P: jnp.ndarray, mu: jnp.ndarray, chol_Sig: jnp.ndarray, chol_Q: jnp.ndarray):
    """
    Compute the lower bound for the coupling probabilities.

    Parameters
    ----------
    m: jnp.ndarray
        Mean vector of the first distribution.
    chol_P: jnp.ndarray
        Cholesky decomposition of the covariance matrix of the first distribution.
    mu: jnp.ndarray
        Mean vector of the second distribution.
    chol_Sig: jnp.ndarray
        Cholesky decomposition of the covariance matrix of the second distribution.
    chol_Q: jnp.ndarray
        Cholesky decomposition of the covariance matrix of the dominating distribution.

    Returns
    -------
    float:
        Lower bound for the coupling probabilities.
    """
    # Dimensionality
    d = m.shape[0]

    # Identity matrix
    eye = jnp.eye(d)

    # Compute the inverse of chol_P, chol_Sig, and chol_Q
    iP = jlinalg.cho_solve((chol_P, True), eye)
    iSig = jlinalg.cho_solve((chol_Sig, True), eye)
    iQ = jlinalg.cho_solve((chol_Q, True), eye)

    # Compute the inverse of the matrix H = (P + Sigma - Q)
    iH = iP + iSig - iQ
    chol_iH = jlinalg.cholesky(iH, lower=True)
    H = jlinalg.cho_solve((chol_iH, True), eye)

    # Compute intermediate values
    a = H @ (iP @ m + (iSig - iQ) @ mu)
    b = m.T @ iP @ m + mu.T @ (iSig - iQ) @ mu - a.T @ iH @ a
    d = H @ (iSig @ mu + (iP - iQ) @ m)
    g = mu.T @ iSig @ mu + m.T @ (iP - iQ) @ m - d.T @ iH @ d

    # Avoid division by zero in F
    jitter = 1e-7
    m = jax.lax.select(jnp.linalg.norm(m - mu) < jitter, m + jitter, m)

    # Define the function F
    def F(u: jnp.ndarray, chol_V: jnp.ndarray):
        num = 0.5 * (m.T @ iQ @ m - mu.T @ iQ @ mu - 2 * u.T @ iQ @ (m - mu))
        den = jnp.linalg.norm(chol_V.T @ iQ @ (m - mu))
        return jstats.norm.cdf(num / den)

    # Compute the Cholesky decomposition of H
    chol_H = jlinalg.cholesky(H, lower=True)

    # Compute acceptance_part and coupling_part
    acceptance_part = jnp.exp(tril_log_det(chol_H) - tril_log_det(chol_Q))
    coupling_part = jnp.exp(-b / 2) * F(a, chol_H) + jnp.exp(-g / 2) * (1 - F(d, chol_H))

    # Compute the final lower bound
    return acceptance_part * coupling_part

def lower_bound_Devroye_et_al(m: jnp.ndarray, chol_P: jnp.ndarray, mu: jnp.ndarray, chol_Sig: jnp.ndarray):
    """
    Theorem 1.2 in https://arxiv.org/abs/1810.08693

    Parameters
    ----------
    m
    chol_P
    mu
    chol_Sig

    Returns
    -------

    """
    d = m.shape[0]
    v = m - mu
    jitter = 1e-6
    v = jax.lax.select(jnp.linalg.norm(v) < jitter, jitter * jnp.ones_like(v), v)

    P = chol_P @ chol_P.T
    Sig = chol_Sig @ chol_Sig.T

    if d > 1:
        # complete an orthonormal basis given by vectors orthogonal to v
        U, *_ = jlinalg.svd(v[:, None])
        Pi = U[:, 1:]

        # compute eigenvals of auxiliary matrix
        aux = jlinalg.solve(Pi.T @ P @ Pi, Pi.T @ Sig @ Pi, sym_pos=True) - jnp.eye(d - 1)
        chol_aux = jlinalg.cholesky(aux)
        eig_vals = jnp.nan_to_num(jnp.diag(chol_aux))
    else:
        eig_vals = jnp.array([0.])

    # Compute the lowercase TV
    tv_1 = jnp.abs(jnp.dot(v, (Sig - P) @ v)) / jnp.dot(v, (P @ v))
    tv_2 = jnp.dot(v, v) / jnp.dot(v, (P @ v)) ** 0.5
    tv_3 = jnp.sum(eig_vals ** 2) ** 0.5
    tv = jnp.maximum(jnp.maximum(tv_1, tv_2), tv_3)

    return jnp.maximum(1 - 4.5 * jnp.minimum(tv, 1), 0)

def compute_asymptotic_bound(chol_P, chol_Q, chol_Sigma):
    """
    Computes the asymptotic lower bound as a function of N for Gaussians

    Parameters
    ----------
    chol_P: jnp.ndarray
        Cholesky decomposition of the covariance matrix of the reference distribution P.
    chol_Q: jnp.ndarray
        Cholesky decomposition of the covariance matrix of the dominating distribution Q.
    chol_Sigma: jnp.ndarray
        Cholesky decomposition of the covariance matrix of another distribution Sigma.

    Returns
    -------
    bound_func: function
        A function that computes the lower and upper bounds given a sample size N.
    """

    d = chol_P.shape[0]
    eye = jnp.eye(d)

    def stddev_term(chol_A, chol_B):
        # Compute the terms involving the logarithm of determinants of the Cholesky decompositions
        B_part = tril_log_det(chol_Q)
        A_part = -2 * tril_log_det(chol_P)

        # Compute the inverse of the matrices
        A_inv = cho_solve((chol_A, True), eye)
        B_inv = cho_solve((chol_B, True), eye)

        # Compute the square root of the determinant of the resulting matrix
        temp = jnp.linalg.cholesky(2 * A_inv - B_inv)
        AB_part = tril_log_det(temp)
        return jnp.sqrt(jnp.exp(A_part + B_part + AB_part) - 1)

    # Compute standard deviation terms for P-Q and Sigma-Q
    var_P_Q = stddev_term(chol_P, chol_Q)
    var_Sig_Q = stddev_term(chol_Sigma, chol_Q)

    # Compute upper and lower standard deviation bounds
    upper_std = jnp.maximum(var_P_Q, var_Sig_Q)
    lower_std = jnp.minimum(var_P_Q, var_Sig_Q)

    # Compute the ratio of determinants of Q to P and Q to Sigma
    M_P = jnp.exp(tril_log_det(chol_Q) - tril_log_det(chol_P))
    M_Sig = jnp.exp(tril_log_det(chol_Q) - tril_log_det(chol_Sigma))

    @partial(jnp.vectorize, signature="()->(m)")
    def bound_func(N):
        # Compute terms A, B, and C used in the bounds
        A = N / (N - 1 + M_P)
        B = N / (N - 1 + M_Sig)
        C = jnp.sqrt(2 * N * jnp.log(jnp.log(N))) / N

        # Compute the lower and upper bounds using the computed terms
        lower_bound = A * B / (1 + lower_std * C)
        upper_bound = 1. / (1. - upper_std * C)

        return jnp.array([lower_bound, upper_bound])

    return bound_func


def coupling_probability_same_cov(m, mu, chol_Q):
    """
    Computes the coupling probability of a maximal coupling between N(m, Q) and N(mu, Q)

    Parameters
    ----------
    m
    mu
    chol_Q

    Returns
    -------

    """
    return 2 * jstats.norm.cdf(-0.5 * jnp.linalg.norm(jlinalg.solve_triangular(chol_Q, m - mu, lower=True)))
