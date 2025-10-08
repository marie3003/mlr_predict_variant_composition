import numpy as np
import pandas as pd
from scipy.optimize import minimize

from viral_variant_comp.simulate_count_data import calculate_frequencies, reorder_variants, create_count_data

from abc import ABC, abstractmethod

def calculate_inital_params(n_variants):
    initial_params = np.concatenate([np.repeat(0.1, n_variants - 1), np.repeat(-5, n_variants - 1)])
    return initial_params

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


class BaseCompositionEstimator(ABC):
    def __init__(self, counts):
        self.counts = counts
        self.T, self.n_variants = counts.shape
        self.growth_rate_estimate = None
        self.log_init_freq_estimate = None
        self.composition_estimate = None
        self.hessian = None
        self.hess_inv = None
        self.hess_inv_estimate = None

    @abstractmethod
    def fit(self):
        pass

    def _postprocess_estimates(self):
        self.composition_estimate = calculate_frequencies(self.T, self.growth_rate_estimate, self.log_init_freq_estimate)
        
        hessian_result = calculateHessian(self.counts, self.growth_rate_estimate, self.log_init_freq_estimate, ignore_pivot=True)
        self.hessian = hessian_result['hessian']
        self.hess_inv = hessian_result['inverse_hessian']


    def get_results(self):
        std_errors = np.sqrt(np.diag(self.hess_inv))
        return {
            'growth_rate_estimate': self.growth_rate_estimate,
            'log_init_freq_estimate': self.log_init_freq_estimate,
            'composition_estimate': self.composition_estimate,
            'hessian': self.hessian,
            'hess_inv': self.hess_inv,
            'std_errors': std_errors,
        }
    
class BFGSCompositionEstimator(BaseCompositionEstimator):
    
    @staticmethod
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
    
    def fit(self):

        initial_params = calculate_inital_params(self.n_variants)
    
        result = minimize(fun = self.neg_log_likelihood_and_grad, x0 = initial_params, args=(self.counts,), method = 'BFGS', jac = True, options={'disp': False})
        result.x = np.insert(result.x, 0, 0.0)
        result.x = np.insert(result.x, self.n_variants, 0.0)

        self.growth_rate_estimate = result.x[:self.n_variants]
        self.log_init_freq_estimate = result.x[self.n_variants:]

        self._postprocess_estimates()


class StepwiseBFGSCompositionEstimator(BaseCompositionEstimator):
    
    def __init__(self, counts, partition_size, overlap_size):
        super().__init__(counts)
        self.partition_size = partition_size
        self.overlap_size = overlap_size

    @staticmethod
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

    def fit(self):

        """
        Estimates growth rate and log. initial frequency for all viral variants in a stepwise manner.
        In each step a window of variants is estimated and moved over the variants which were pre-sorted by average time. 
        In each estimation step all variants that were included before the current window are included in the likelihood calculation but not estimated again.
        For parameters that enter a second window, the estimate from the previous window is chosen as new initial guess for the parameter.
        In the first iteration, the first parameter needs to be fixed to (0,0) as a reference.
        @param partition_size: size of window, number of variants newly estimated at once (can't be greater than number of variants in data)
        @param overlap_size: overlap between windows, number of previously already estimated variants that are estimated again in new window

        Afterwards, these parameter estimates are used to calculate viral frequencies for each observed time point.
        """

        step_size = self.partition_size - self.overlap_size
        n_partitions = int(1 + np.ceil((self.n_variants - self.partition_size) / step_size))

        s_vec = np.zeros(self.n_variants)
        o_vec = np.zeros(self.n_variants)

        fixed_params = np.zeros(2)  # first variant is fixed to (0, 0)
        s_estimate = None
        o_estimate = None
        n_new_estimates = None

        for i in range(n_partitions):

            interval_start = i * step_size
            interval_end = min(self.n_variants, interval_start + self.partition_size)

            relevant_counts = self.counts[:,:interval_end]

            # initial params are chosen to be the estimates from the previous estimation combined with their mean for the next values where no initial guess exists
            if (i == 0):
                initial_params = calculate_inital_params(self.partition_size)
            else:
                n_new_initial_param = interval_end - interval_start - len(s_new_initial_params)
                initial_params = np.concatenate([s_new_initial_params, np.repeat(np.mean(s_new_initial_params), n_new_initial_param), o_new_initial_params, np.repeat(np.mean(o_new_initial_params), n_new_initial_param)])

            param_estimate = minimize(fun = self.neg_log_likelihood_and_grad_stepwise, x0 = initial_params, args=(relevant_counts, fixed_params), method = 'BFGS', jac = True)

            n_new_estimates = len(param_estimate.x) // 2
            s_estimate = param_estimate.x[:n_new_estimates]
            o_estimate = param_estimate.x[n_new_estimates:]

            if(i == 0): # first parameter needs to be fixed to 0, in subsequent rounds already parameters from previous rounds are fixed
                s_vec[interval_start + 1: interval_start + step_size] = s_estimate[:step_size - 1] 
                o_vec[interval_start + 1: interval_start + step_size] = o_estimate[:step_size - 1]
            else:
                s_vec[interval_start : interval_start + step_size] = s_estimate[:step_size]  
                o_vec[interval_start : interval_start + step_size] = o_estimate[:step_size]

            fixed_params = np.concatenate([s_vec[:interval_end - self.overlap_size], o_vec[:interval_end - self.overlap_size]])
            
            s_new_initial_params = s_estimate[-self.overlap_size:]
            o_new_initial_params = o_estimate[-self.overlap_size:]

        n_remaining_param = len(s_estimate) - self.overlap_size
        if(n_remaining_param > 0):
            s_vec[- n_remaining_param:] = s_estimate[- n_remaining_param:]
            o_vec[- n_remaining_param:] = o_estimate[- n_remaining_param:]

        self.growth_rate_estimate = s_vec
        self.log_init_freq_estimate = o_vec

        self._postprocess_estimates()






### HELPERS

def calculate_cooccurence(counts):
    presence = (counts > 0).astype(int)
    co_occurrence_matrix = np.matmul(presence.T, presence)
    return co_occurrence_matrix

import numpy as np
import copy

def choose_pivot_variant(data_pango, pivot):  # e.g. pivot = 'A'

    data_pango_copy = copy.deepcopy(data_pango)


    idx_pivot = np.where(data_pango_copy['variant_names'] == pivot)[0][0]

    # Create a new ordering with pivot first
    new_order = [idx_pivot] + [i for i in range(len(data_pango_copy['variant_names'])) if i != idx_pivot]

    # Reorder all relevant arrays
    data_pango_copy['counts'] = data_pango_copy['counts'][:, new_order]
    data_pango_copy['variant_names'] = data_pango_copy['variant_names'][new_order]
    data_pango_copy['mean_time'] = data_pango_copy['mean_time'][new_order]

    return data_pango_copy

def calculate_mean_fitness(freq, growth_rates):
    return np.matmul(freq, growth_rates)