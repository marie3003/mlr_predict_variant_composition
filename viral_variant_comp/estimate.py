import numpy as np
import pandas as pd
from scipy.optimize import minimize

import evofr as ef
from viral_variant_comp.simulate_count_data import calculate_frequencies, reorder_variants, prepare_count_data_evofr, create_count_data
from viral_variant_comp.plotting import plot_viral_composition_dual, plot_confidence_intervals_deviation

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

    @abstractmethod
    def fit(self):
        pass

    def _postprocess_estimates(self):
        self.composition_estimate = calculate_frequencies(self.T, self.growth_rate_estimate, self.log_init_freq_estimate)
        
        hessian_result = calculateHessian(self.counts, self.growth_rate_estimate, self.log_init_freq_estimate, ignore_pivot=True)
        self.hessian = hessian_result['hessian']
        self.hess_inv = hessian_result['inverse_hessian']


    def get_results(self):
        return {
            'growth_rate_estimate': self.growth_rate_estimate,
            'log_init_freq_estimate': self.log_init_freq_estimate,
            'composition_estimate': self.composition_estimate,
            'hessian': self.hessian,
            'hess_inv': self.hess_inv,
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
    
        result = minimize(fun = self.neg_log_likelihood_and_grad, x0 = initial_params, args=(self.counts,), method = 'BFGS', jac = True)
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


class EvofrCompositionEstimator(BaseCompositionEstimator):
    def __init__(self, counts, iterations=300000, learning_rate=4e-3, generation_time = 1):
        super().__init__(counts)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.generation_time = generation_time

    def fit(self):

        counts_df = prepare_count_data_evofr(self.counts)

        # need to reorder variants because evofr always chooses last element in variant vector as pivot
        variant_names = [f"variant_{i+1}" for i in range(counts_df.variant.unique().shape[0]-1)]
        variant_names.append('variant_0')
        variant_frequencies = ef.VariantFrequencies(counts_df, var_names=variant_names)    #pivot="variant_0"

        mlr = ef.MultinomialLogisticRegression(tau=self.generation_time) # tau: average generation time (1 day)
        inference_method = ef.InferMAP(iters = self.iterations, lr = self.learning_rate)
        posterior = inference_method.fit(mlr, variant_frequencies)

        #forecast_L = 50
        #posterior.samples = mlr.forecast_frequencies(posterior.samples, forecast_L)

        o_vec = np.median(posterior.samples['beta'][:,0,:], axis = 0)
        o_vec = np.concatenate(([o_vec[-1]], o_vec[:-1]))

        s_vec = np.median(posterior.samples['beta'][:,1,:], axis = 0)
        s_vec = np.concatenate(([s_vec[-1]], s_vec[:-1]))

        self.growth_rate_estimate = s_vec
        self.log_init_freq_estimate = o_vec

        self._postprocess_estimates()



### EVALUATE RESULT

def calculate_cooccurence(counts):
    presence = (counts > 0).astype(int)
    co_occurrence_matrix = np.matmul(presence.T, presence)
    return co_occurrence_matrix


def evaluate_result(s_vec_est, o_vec_est, inv_hessian, growth_rates, log_init_freq):

    parameter_estimates = np.concatenate([s_vec_est, o_vec_est])
    true_params = np.concatenate([growth_rates, log_init_freq])
    
    z = 1.96    # 95% confidence interval
    variance = np.insert(np.diag(inv_hessian), 0, 0.0)
    variance = np.insert(variance, true_params.shape[0] // 2, 0.0)
    standard_errors =  z * np.sqrt(variance)
    lower_bounds = parameter_estimates - standard_errors
    upper_bounds = parameter_estimates + standard_errors

    deviations = parameter_estimates - true_params
    abs_deviations = np.abs(deviations)
    estimated_correctly = abs_deviations <= standard_errors

    # calculate error of differences
    diff_to_next_true = np.diff(true_params)
    diff_to_next_true[growth_rates.shape[0] - 1] = np.nan
    diff_to_next_true = np.append(diff_to_next_true, np.nan)
    diff_to_next_est = np.diff(parameter_estimates)
    diff_to_next_est[growth_rates.shape[0] - 1] = np.nan
    diff_to_next_est = np.append(diff_to_next_est, np.nan)
    squared_error = (diff_to_next_est - diff_to_next_true)**2

    # calculate variance of differences
    var_i = variance[:-1]
    var_next = variance[1:]
    covariance = np.insert(np.diag(inv_hessian, k = 1), 0, 0.0) # choose covariance of pivot to second element as 0
    covariance = np.insert(covariance, growth_rates.shape[0] - 1, 0.0)
    var_diff = var_i + var_next - 2*covariance
    var_diff[growth_rates.shape[0] - 1] = np.nan
    var_diff = np.append(var_diff, np.nan)

    # Confidence intervals for the differences between estimates
    standard_errors_diff = z * np.sqrt(var_diff)
    ci_diff_lower = diff_to_next_est - standard_errors_diff
    ci_diff_upper = diff_to_next_est + standard_errors_diff

    df = pd.DataFrame({
        'parameter_estimate': parameter_estimates,
        'true_parameter': true_params,
        'deviation': deviations,
        'abs_deviation': abs_deviations,
        'estimated_correctly': estimated_correctly,
        'standard_error': standard_errors,
        'ci_lower': lower_bounds,
        'ci_upper': upper_bounds,
        'difference_to_next_estimate': diff_to_next_est,
        'difference_to_next_true': diff_to_next_true,
        'difference_to_next_se': squared_error,
        'variance_of_diff': var_diff,
        'standard_error_of_diff': standard_errors_diff,
        'ci_lower_of_diff': ci_diff_lower,
        'ci_upper_of_diff': ci_diff_upper,
        'n_variants': np.repeat(parameter_estimates.shape[0] // 2, parameter_estimates.shape[0]),
        'parameter_type': np.concatenate([np.repeat('growth_rate', parameter_estimates.shape[0] // 2), np.repeat('log_initial_freq', parameter_estimates.shape[0] // 2)])
    })

    return df

def simulate_estimate_evaluate(n_variants = 50, n_days = 1000, delta_gr_range = 0.2, new_var_rate = 1/14, freq_entering_variants = 0.0001, n_samples = 1000, s_0 = 0., o_0 = 0., reorder=True, seed = 5,
                               estimation_method = 'BFGS',
                               partition_size = None, overlap_size = None,
                               iterations = None, learning_rate = None,
                               plot_results = False):
    
    data = create_count_data(n_variants = n_variants, n_days = n_days, delta_gr_range = delta_gr_range, new_var_rate = new_var_rate, freq_entering_variants = freq_entering_variants, n_samples = n_samples, s_0 = s_0, o_0 = o_0, reorder=reorder, seed = seed)
    
    if estimation_method == 'BFGS':
        estimator = BFGSCompositionEstimator(data['counts'])
    elif estimation_method == 'stepwiseBFGS':
        estimator = StepwiseBFGSCompositionEstimator(data['counts'], partition_size=partition_size, overlap_size=overlap_size) # should overlap size depend on number of variants (possibly write function that determines overlap size based on co-occurence matrix)
    elif estimation_method == 'evofr':
        estimator = EvofrCompositionEstimator(data['counts'], iterations = iterations, learning_rate= learning_rate, generation_time=1)

    estimator.fit()
    result = estimator.get_results()
    df = evaluate_result(result['growth_rate_estimate'], result['log_init_freq_estimate'], result['hess_inv'], data['growth_rates'], data['log_init_freq'])

    if plot_results:
        plot_viral_composition_dual(data['counts'], data['freq'], result['composition_estimate'], y_range = (1e-5, 2))
        plot_confidence_intervals_deviation(df)

    return df

def run_sampling_experiment(
    variable_to_vary: str,
    variable_values: list,
    estimation_methods: list = ['BFGS', 'stepwiseBFGS', 'evofr'],
    n_variants=50,
    n_days=1000,
    delta_gr_range=0.2,
    new_var_rate=1/14,
    freq_entering_variants=0.0001,
    n_samples=1000,
    partition_size=10,
    overlap_size=4,
    iterations=300000,
    learning_rate=4e-3,
    seed=5,
    n_repeats=5,
    plot_results=False
):
    all_results = []

    if variable_to_vary == 'n_samples':
        days_vec = np.repeat(n_days, len(variable_values))
    elif variable_to_vary == 'n_variants':
        days_vec = np.array(variable_values) * 18
    elif variable_to_vary == 'new_var_rate':
        days_vec = np.ceil(n_variants / np.array(variable_values)).astype(int) * 2
    else:
        raise ValueError(f"Unsupported variable_to_vary: {variable_to_vary}")

    for i, value in enumerate(variable_values):
        n_days = days_vec[i]

        for method in estimation_methods:
            print(f'Estimation method: {method}, {variable_to_vary} = {value}')

            for r in range(n_repeats):
                current_seed = seed + r

                method_kwargs = {}
                if method == 'stepwiseBFGS':
                    method_kwargs = {'partition_size': partition_size, 'overlap_size': overlap_size}
                elif method == 'evofr':
                    method_kwargs = {'iterations': iterations, 'learning_rate': learning_rate}

                params = {
                    'n_variants': n_variants,
                    'n_days': n_days,
                    'delta_gr_range': delta_gr_range,
                    'new_var_rate': new_var_rate,
                    'freq_entering_variants': freq_entering_variants,
                    'n_samples': n_samples,
                    'estimation_method': method,
                    'seed': current_seed,
                    'plot_results': plot_results
                }

                # Override the varied parameter
                params[variable_to_vary] = value

                # Simulate
                df = simulate_estimate_evaluate(**params, **method_kwargs)
                mse_gr = np.nanmean(df[df.parameter_type == 'growth_rate']['difference_to_next_se'])
                mse_lif = np.nanmean(df[df.parameter_type != 'growth_rate']['difference_to_next_se'])

                all_results.append({
                    'method': method,
                    'gr_mse': mse_gr,
                    'lif_mse': mse_lif,
                    'n_samples': params['n_samples'],
                    'n_variants': params['n_variants'],
                    'new_var_rate': params['new_var_rate'],
                    'n_days': params['n_days']
                })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Group and aggregate
    grouped = df.groupby(['n_samples', 'n_variants', 'new_var_rate', 'n_days', 'method']).agg(
        gr_mse_mean=('gr_mse', 'mean'),
        gr_mse_std=('gr_mse', 'std'),
        lif_mse_mean=('lif_mse', 'mean'),
        lif_mse_std=('lif_mse', 'std')
    ).reset_index()

    # Pivot for mean
    df_mean = grouped.pivot(index=['n_samples', 'n_variants', 'new_var_rate', 'n_days'], columns='method', values=['gr_mse_mean', 'lif_mse_mean'])
    df_mean.columns = [f"{metric}_{method}" for metric, method in df_mean.columns]

    # Pivot for std
    df_std = grouped.pivot(index=['n_samples', 'n_variants', 'new_var_rate', 'n_days'], columns='method', values=['gr_mse_std', 'lif_mse_std'])
    df_std.columns = [f"{metric}_{method}" for metric, method in df_std.columns]

    # Combine mean and std
    df_final = pd.concat([df_mean, df_std], axis=1).reset_index()

    return df_final


