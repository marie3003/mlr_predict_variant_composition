import numpy as np
import pandas as pd
from scipy.optimize import minimize

from viral_variant_comp.simulate_count_data import calculate_frequencies, reorder_variants

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

def calculate_cooccurence(counts):
    presence = (counts > 0).astype(int)
    co_occurrence_matrix = np.matmul(presence.T, presence)
    return co_occurrence_matrix

def calculate_inital_params(n_variants):
    initial_params = np.concatenate([np.repeat(0.1, n_variants - 1), np.repeat(-5, n_variants - 1)])
    return initial_params

def calculate_viral_composition(data):

    initial_params = calculate_inital_params(data['counts'].shape[1])
    
    result = minimize(fun = neg_log_likelihood_and_grad_vectorised, x0 = initial_params, args=(data['counts'],), method = 'BFGS', jac = True)
    result.x = np.insert(result.x, 0, 0.0)
    result.x = np.insert(result.x, data['counts'].shape[1], 0.0)

    T = data['counts'].shape[0]
    n = len(result.x) // 2
    growth_r = result.x[:n]
    log_initial_freq = result.x[n:]
    composition_estimate = np.zeros(data['counts'].shape)

    for t in range(T):
        logits = growth_r * t + log_initial_freq
        max_l = np.max(logits)
        exp_growth = np.exp(logits - max_l)
        composition_estimate[t] = exp_growth / np.sum(exp_growth)

    return result, composition_estimate

def evaluate_result(parameter_estimates, inv_hessian, growth_rates, log_init_freq):

    true_params = np.concatenate([growth_rates, log_init_freq])
    print(true_params.shape)
    
    z = 1.96    # 95% confidence interval
    standard_errors =  np.insert(z * np.sqrt(np.diag(inv_hessian)), 0, 0.0)
    standard_errors =  np.insert(standard_errors, true_params.shape[0] // 2, 0.0)
    lower_bounds = parameter_estimates - standard_errors
    upper_bounds = parameter_estimates + standard_errors

    deviations = parameter_estimates - true_params
    abs_deviations = np.abs(deviations)
    estimated_correctly = abs_deviations <= standard_errors

    df = pd.DataFrame({
        'parameter_estimate': parameter_estimates,
        'true_parameter': true_params,
        'deviation': deviations,
        'abs_deviation': abs_deviations,
        'estimated_correctly': estimated_correctly,
        'standard_error': standard_errors,
        'ci_lower': lower_bounds,
        'ci_upper': upper_bounds,
        'n_variants': np.repeat(parameter_estimates.shape[0] // 2, parameter_estimates.shape[0]),
        'parameter_type': np.concatenate([np.repeat('growth_rate', parameter_estimates.shape[0] // 2), np.repeat('log_initial_freq', parameter_estimates.shape[0] // 2)])
    })

    return df

def calculateHessian(counts, s_vec, o_vec, ignore_pivot = False):

    T, n = counts.shape

    t_vec = np.arange(T)
    N_vec = np.sum(counts, axis = 1)

    logits = np.outer(t_vec, s_vec) + o_vec     # shape T + 1 times n 
    max_logits = np.max(logits, axis = 1)   # shape T
    shifted_exp_logits = np.exp(logits - max_logits[:, np.newaxis])  # max_logits here vector of shape T+1 times 1
    sum_shifted_exp_logits = np.sum(shifted_exp_logits, axis = 1)   # shape T
    freq = shifted_exp_logits / sum_shifted_exp_logits[:, np.newaxis]

    N_freq = N_vec[:, np.newaxis] * freq
    N_t_freq = t_vec[:, np.newaxis] * N_freq
    N_t2_freq = t_vec[:, np.newaxis] * N_t_freq
    counter_freq = 1 - freq

    d_si_sj = np.matmul(np.transpose(N_t2_freq), freq) # n times n
    d_si_si = -np.sum(np.multiply(N_t2_freq, counter_freq), axis = 0) # n
    np.fill_diagonal(d_si_sj, d_si_si)

    d_oi_oj = np.matmul(np.transpose(N_freq), freq)
    d_oi_oi = - np.sum(np.multiply(N_freq, counter_freq), axis = 0)
    np.fill_diagonal(d_oi_oj, d_oi_oi)

    d_oi_sj = np.matmul(np.transpose(N_t_freq), freq)
    d_oi_si = - np.sum(np.multiply(N_t_freq, counter_freq), axis = 0)
    np.fill_diagonal(d_oi_sj, d_oi_si)

    top = np.hstack((d_si_sj, d_oi_sj))   # shape: (n, 2n)
    bottom = np.hstack((d_oi_sj, d_oi_oj))  # shape: (n, 2n)

    hessian = - np.vstack((top, bottom))  # shape: (2n, 2n), minus because we're dealing with neg log likelihood
    
    if (ignore_pivot):
        idx_to_remove = [0, len(s_vec)]
        hessian = np.delete(hessian, idx_to_remove, axis=0)  # remove row
        hessian = np.delete(hessian, idx_to_remove, axis=1)  # remove column

    inv_hessian = np.linalg.inv(hessian)
    eigenvalues, eigenvectors = np.linalg.eig(hessian)

    return {"hessian": hessian, "inverse_hessian": inv_hessian, "eigenvalues_hessian": eigenvalues, "eigenvectors_hessian": eigenvectors}






def neg_log_likelihood_and_grad_stepwise(params, c_t, fixed_params):
    """
    Negative log-likelihood function as defined above and gradient of the negative log-likelihood with respect to s and o.
    s_1 and o_1 need to be fixed to 0 to find one optimal solution. 

    params: array od shape (2n,) where first n are s_i and last n o_i
    c_t: array of shape (T+1, n), where c_t[t, i] = c_i^(t)
    fixed_params: parameter vector containing the parameters that are fixed (first to mth variant)
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


def calculate_viral_composition_stepwise(data_reordered, partition_size, overlap_size):
    """
    Estimates growth rate and log. initial frequency for all viral variants in a stepwise manner.
    In each step a window of variants is estimated and moved over the variants which were pre-sorted by average time. 
    In each estimation step all variants that were included before the current window are included in the likelihood calculation but not estimated again.
    For parameters that enter a second window, the estimate from the previous window is chosen as new initial guess for the parameter.
    In the first iteration, the first parameter needs to be fixed to (0,0) as a reference.
    @param partition_size: size of window, number of variants newly estimated at once
    @param overlap_size: overlap between windows, number of previously already estimated variants that are estimated again in new window

    Afterwards, these parameter estimates are used to calculate viral frequencies for each observed time point.
    """

    step_size = partition_size - overlap_size
    n_partitions = int(1 + np.ceil((data_reordered['n_variants'] - partition_size) / step_size))

    s_vec = np.zeros(data_reordered['n_variants'])
    o_vec = np.zeros(data_reordered['n_variants'])

    fixed_params = np.zeros(2)  # first variant is fixed to (0, 0)
    s_estimate = None
    o_estimate = None
    n_new_estimates = None

    for i in range(n_partitions):

        print("Loop ", i)

        interval_start = i * step_size
        interval_end = min(data_reordered['n_variants'], interval_start + partition_size)

        relevant_counts = data_reordered['counts'][:,:interval_end]

        # initial params are chosen to be the estimates from the previous estimation combined with their mean for the next values where no initial guess exists
        if (i == 0):
            initial_params = calculate_inital_params(partition_size)
        else:
            n_new_initial_param = interval_end - interval_start - len(s_new_initial_params)
            initial_params = np.concatenate([s_new_initial_params, np.repeat(np.mean(s_new_initial_params), n_new_initial_param), o_new_initial_params, np.repeat(np.mean(o_new_initial_params), n_new_initial_param)])

        param_estimate = minimize(fun = neg_log_likelihood_and_grad_stepwise, x0 = initial_params, args=(relevant_counts, fixed_params), method = 'BFGS', jac = True)

        n_new_estimates = len(param_estimate.x) // 2
        s_estimate = param_estimate.x[:n_new_estimates]
        o_estimate = param_estimate.x[n_new_estimates:]

        if(i == 0): # first parameter needs to be fixed to 0, in subsequent rounds already parameters from previous rounds are fixed
            s_vec[interval_start + 1: interval_start + step_size] = s_estimate[:step_size - 1] 
            o_vec[interval_start + 1: interval_start + step_size] = o_estimate[:step_size - 1]
        else:
            s_vec[interval_start : interval_start + step_size] = s_estimate[:step_size]  
            o_vec[interval_start : interval_start + step_size] = o_estimate[:step_size]

        fixed_params = np.concatenate([s_vec[:interval_end - overlap_size], o_vec[:interval_end - overlap_size]])
        
        s_new_initial_params = s_estimate[-overlap_size:]
        o_new_initial_params = o_estimate[-overlap_size:]

    n_remaining_param = len(s_estimate) - overlap_size
    if(n_remaining_param > 0):
        s_vec[- n_remaining_param:] = s_estimate[- n_remaining_param:]
        o_vec[- n_remaining_param:] = o_estimate[- n_remaining_param:]

    composition_estimate = calculate_frequencies(data_reordered['counts'].shape[0], s_vec, o_vec)
    hessian_result = calculateHessian(data_reordered['counts'], s_vec, o_vec, ignore_pivot=True)

    return {'growth_rate_estimate': s_vec, "log_init_freq_estimate": o_vec, "composition_estimate": composition_estimate, "reordered_data": data_reordered, "hessian": hessian_result['hessian'], "hess_inv": hessian_result["inverse_hessian"]}