import numpy as np
import pandas as pd
from scipy.optimize import minimize

from viral_variant_comp.simulate_count_data import calculate_frequencies

def neg_log_likelihood_and_grad_vectorised(params, c_t):
    """
    Negative log-likelihood function as defined above and gradient of the negative log-likelihood with respect to s and o.
    s_1 and o_1 need to be fixed to 0 to find one optimal solution. 

    params: array od shape (2n,) where first n are s_i and last n o_i
    c_t: array of shape (T+1, n), where c_t[t, i] = c_i^(t)
    """

    T_plus_1, n = c_t.shape
    T = T_plus_1 - 1

    s = np.insert(params[:n-1], 0, 0.0)
    o = np.insert(params[n-1:], 0, 0.0)
    t_vec = np.arange(T+1)
    N_vec = np.sum(c_t, axis = 1)
    N_times_t_vec = np.multiply(N_vec, t_vec)

    logits = np.outer(t_vec, s) + o     # shape T + 1 times n 
    max_logits = np.max(logits, axis = 1)    
    shifted_exp_logits = np.exp(logits - max_logits[:, np.newaxis])  # max_logits here vector of shape T+1 times 1
    sum_shifted_exp_logits = np.sum(shifted_exp_logits, axis = 1)
    probs = shifted_exp_logits / sum_shifted_exp_logits[:, np.newaxis]

    grad_s = np.matmul(t_vec, c_t) - np.matmul(N_times_t_vec, probs)
    grad_o = np.sum(c_t, axis = 0) - np.matmul(N_vec, probs)

    log_probs = logits - (np.log(sum_shifted_exp_logits) + max_logits)[:, np.newaxis]
    ll = np.sum(c_t * log_probs)

    return (-ll, -np.concatenate([grad_s[1:], grad_o[1:]]))  # we minimize the negative log-likelihood,  gradient of negative log-likelihood excluding s_1

def neg_log_likelihood_and_grad_stepwise(params, c_t, fixed_params):
    """
    Negative log-likelihood function as defined above and gradient of the negative log-likelihood with respect to s and o.
    s_1 and o_1 need to be fixed to 0 to find one optimal solution. 

    params: array od shape (2n,) where first n are s_i and last n o_i
    c_t: array of shape (T+1, n), where c_t[t, i] = c_i^(t)
    """

    T_plus_1, n = c_t.shape
    T = T_plus_1 - 1

    fixed_len = len(fixed_params) // 2

    s = np.concatenate([fixed_params[:fixed_len], params[:n-fixed_len]])
    o = np.concatenate([fixed_params[fixed_len:], params[n-fixed_len:]])

    t_vec = np.arange(T+1)
    N_vec = np.sum(c_t, axis = 1)
    N_times_t_vec = np.multiply(N_vec, t_vec)

    logits = np.outer(t_vec, s) + o     # shape T + 1 times n 
    max_logits = np.max(logits, axis = 1)    
    shifted_exp_logits = np.exp(logits - max_logits[:, np.newaxis])  # max_logits here vector of shape T+1 times 1
    sum_shifted_exp_logits = np.sum(shifted_exp_logits, axis = 1)
    probs = shifted_exp_logits / sum_shifted_exp_logits[:, np.newaxis]

    grad_s = np.matmul(t_vec, c_t) - np.matmul(N_times_t_vec, probs)
    grad_o = np.sum(c_t, axis = 0) - np.matmul(N_vec, probs)

    log_probs = logits - (np.log(sum_shifted_exp_logits) + max_logits)[:, np.newaxis]
    ll = np.sum(c_t * log_probs)

    return (-ll, -np.concatenate([grad_s[fixed_len:], grad_o[fixed_len:]]))  # we minimize the negative log-likelihood,  gradient of negative log-likelihood excluding s_1

def neg_log_likelihood_and_grad(params, c_t):
    """
    Negative log-likelihood function as defined above and gradient of the negative log-likelihood with respect to s and o.
    s_1 and o_1 need to be fixed to 0 to find one optimal solution. 

    params: array od shape (2n,) where first n are s_i and last n o_i
    c_t: array of shape (T+1, n), where c_t[t, i] = c_i^(t)
    """

    T_plus_1, n = c_t.shape
    T = T_plus_1 - 1

    s = np.insert(params[:n-1], 0, 0.0)
    o = np.insert(params[n-1:], 0, 0.0)

    grad_s = np.zeros(n)
    grad_o = np.zeros(n)

    ll = 0.0

    for t in range(T + 1):
        logits = s * t + o
        max_logit = np.max(logits)
        exp_logit = np.exp(logits - max_logit)
        sum_exp_logit = np.sum(exp_logit)

        N_t = np.sum(c_t[t])

        log_sum_exp = max_logit + np.log(sum_exp_logit)
        ll += np.dot(c_t[t], logits) - N_t * log_sum_exp

        probs = exp_logit / sum_exp_logit
        grad_t = c_t[t] - N_t * probs   # one time step
        grad_s += grad_t * t
        grad_o += grad_t

    return (-ll, -np.concatenate([grad_s[1:], grad_o[1:]]))  # we minimize the negative log-likelihood,  gradient of negative log-likelihood excluding s_1

def reorder_variants(count_data):

    count_data = count_data.copy()

    variant_normalized_counts = count_data['counts'] / np.sum(count_data['counts'], axis = 0)
    t_peaks = np.sum(variant_normalized_counts * np.arange(count_data['counts'].shape[0])[:, np.newaxis], axis = 0)

    variant_position = np.argsort(t_peaks)
    count_data['counts'] = count_data['counts'].copy()[:, variant_position]

    count_data['growth_rates'] = count_data['growth_rates'].copy()[variant_position]
    count_data['log_init_freq'] = count_data['log_init_freq'].copy()[variant_position]
    count_data['freq'] = count_data['freq'].copy()[:, variant_position]

    return count_data

def calculate_inital_params(n_variants):
    initial_params = np.concatenate([np.repeat(0.1, n_variants - 1), np.repeat(-5, n_variants - 1)])
    return initial_params

def calculate_viral_composition(counts):

    initial_params = calculate_inital_params(counts.shape[1])
    
    result = minimize(fun = neg_log_likelihood_and_grad_vectorised, x0 = initial_params, args=(counts,), method = 'BFGS', jac = True)
    result.x = np.insert(result.x, 0, 0.0)
    result.x = np.insert(result.x, counts.shape[1], 0.0)

    T = counts.shape[0]
    n = len(result.x) // 2
    growth_r = result.x[:n]
    log_initial_freq = result.x[n:]
    composition_estimate = np.zeros(counts.shape)

    for t in range(T):
        logits = growth_r * t + log_initial_freq
        max_l = np.max(logits)
        exp_growth = np.exp(logits - max_l)
        composition_estimate[t] = exp_growth / np.sum(exp_growth)

    return result, composition_estimate

def evaluate_result(result, growth_rates, log_init_freq):

    true_params = np.concatenate([growth_rates, log_init_freq])
    print(true_params.shape)
    
    z = 1.96    # 95% confidence interval
    standard_errors =  np.insert(z * np.sqrt(np.diag(result.hess_inv)), 0, 0.0)
    standard_errors =  np.insert(standard_errors, true_params.shape[0] // 2, 0.0)
    lower_bounds = result.x - standard_errors
    upper_bounds = result.x + standard_errors

    deviations = result.x - true_params
    abs_deviations = np.abs(deviations)
    estimated_correctly = abs_deviations <= standard_errors

    df = pd.DataFrame({
        'parameter_estimate': result.x,
        'true_parameter': true_params,
        'deviation': deviations,
        'abs_deviation': abs_deviations,
        'estimated_correctly': estimated_correctly,
        'standard_error': standard_errors,
        'ci_lower': lower_bounds,
        'ci_upper': upper_bounds,
        'n_variants': np.repeat(result.x.shape[0] // 2, result.x.shape[0]),
        'parameter_type': np.concatenate([np.repeat('growth_rate', result.x.shape[0] // 2), np.repeat('log_initial_freq', result.x.shape[0] // 2)])
    })

    return df


def calculate_viral_composition_stepwise(count_data, partition_size, overlap_size):
    data_reordered = reorder_variants(count_data)

    step_size = partition_size - overlap_size
    n_partitions = int(1 + np.ceil((data_reordered['n_variants'] - partition_size) / step_size))

    s_vec = np.zeros(data_reordered['n_variants'])
    o_vec = np.zeros(data_reordered['n_variants'])

    fixed_params = np.zeros(2)  # first variant is fixed to (0, 0()

    for i in range(n_partitions):

        interval_start = i * step_size
        interval_end = min(data_reordered['n_variants'], interval_start + partition_size)

        initial_params = calculate_inital_params(interval_end - interval_start - (len(fixed_params) // 2) + 1)  # adapt function to parameter range
        relevant_counts = data_reordered['counts'][:,interval_start:interval_end]

        param_estimate = minimize(fun = neg_log_likelihood_and_grad_stepwise, x0 = initial_params, args=(relevant_counts, fixed_params), method = 'BFGS', jac = True)

        n_new_estimates = len(param_estimate.x) // 2
        s_estimate = param_estimate.x[:n_new_estimates]
        o_estimate = param_estimate.x[n_new_estimates:]

        s_vec[interval_start + (len(fixed_params) // 2) : interval_end] = s_estimate
        o_vec[interval_start + (len(fixed_params) // 2) : interval_end] = o_estimate

        fixed_params = np.concatenate([s_estimate[- overlap_size:], o_estimate[-overlap_size:]])

    composition_estimate = calculate_frequencies(data_reordered['counts'].shape[0], s_vec, o_vec)

    return {'growth_rate_estimate': s_vec, "log_init_freq_estimate": o_vec, "composition_estimate": composition_estimate, "reordered_data": data_reordered}





