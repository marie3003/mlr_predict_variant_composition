import pandas as pd
import numpy as np

from viral_variant_comp.estimate import BFGSCompositionEstimator, StepwiseBFGSCompositionEstimator, EvofrCompositionEstimator
from viral_variant_comp.simulate_count_data import create_count_data
from viral_variant_comp.plotting import plot_confidence_intervals_deviation, plot_viral_composition_dual

## HELPER FUNCTIONS

def calculate_weighted_mse(df, parameter_type, alpha=1):
    """
    Calculate the weighted mean squared error (MSE) between the estimated and true differences.
    The weights are based on the variant counts raised to the power of alpha.
    """
    df_half = df[df.parameter_type == parameter_type]

    counts_i = df_half.variant_counts.iloc[:-1].values
    counts_ip1 = df_half.variant_counts.iloc[1:].values
    
    # Compute average counts for transitions
    avg_counts = (counts_i + counts_ip1) / 2
    weights = avg_counts ** alpha

    weighted_mse = np.sum(weights * df_half.difference_to_next_se[:-1])/ np.sum(weights)
    return weighted_mse

## MAIN FUNCTIONS

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


def evaluate_result_high_counts(s_vec_est, o_vec_est, inv_hessian, growth_rates, log_init_freq, variant_counts, variant_count_threshold=50):
    """
    Evaluate the results of the estimation by calculating confidence intervals, deviations, and other statistics.
    Variants with low counts (below variant count threshold) are excluded from the evaluation.
    """

    indices = np.where(variant_counts > variant_count_threshold)[0]
    n = len(indices)
    indices = np.concatenate([indices, indices + len(variant_counts)])

    parameter_estimates = np.concatenate([s_vec_est, o_vec_est])[indices]
    true_params = np.concatenate([growth_rates, log_init_freq])[indices]

    # add 0 row to covariance matrix, also between variables
    inv_hessian =  np.vstack([np.zeros((1, inv_hessian.shape[1])), inv_hessian])
    inv_hessian = np.hstack([np.zeros((inv_hessian.shape[0], 1)), inv_hessian])
    inv_hessian = np.insert(inv_hessian, true_params.shape[0] // 2, 0, axis=0)
    inv_hessian = np.insert(inv_hessian, true_params.shape[0] // 2, 0, axis=1)
    inv_hessian = inv_hessian[indices][:,indices]

    z = 1.96    # 95% confidence interval
    variance = np.diag(inv_hessian)

    standard_errors =  z * np.sqrt(variance)
    lower_bounds = parameter_estimates - standard_errors
    upper_bounds = parameter_estimates + standard_errors 

    deviations = parameter_estimates - true_params
    abs_deviations = np.abs(deviations)
    estimated_correctly = abs_deviations <= standard_errors

    # calculate error of differences
    diff_to_next_true = np.diff(true_params)
    diff_to_next_true[n - 1] = np.nan
    diff_to_next_true = np.append(diff_to_next_true, np.nan)
    diff_to_next_est = np.diff(parameter_estimates)
    diff_to_next_est[n - 1] = np.nan
    diff_to_next_est = np.append(diff_to_next_est, np.nan)
    abs_error = np.abs(diff_to_next_est - diff_to_next_true)
    squared_error = abs_error**2

    # calculate variance of differences
    var_i = variance[:-1]
    var_next = variance[1:]
    covariance = np.diag(inv_hessian, k = 1) # choose covariance of pivot to second element as 0
    var_diff = var_i + var_next - 2*covariance
    var_diff[n - 1] = np.nan
    var_diff = np.append(var_diff, np.nan)

    # Confidence intervals for the differences between estimates
    standard_errors_diff = z * np.sqrt(var_diff)
    ci_diff_lower = diff_to_next_est - standard_errors_diff
    ci_diff_upper = diff_to_next_est + standard_errors_diff

    df = pd.DataFrame({
        'indices': indices,
        'variant_counts': np.concatenate([variant_counts, variant_counts])[indices],
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
        'difference_to_next_ae': abs_error,
        'difference_to_next_se': squared_error,
        'variance_of_diff': var_diff,
        'standard_error_of_diff': standard_errors_diff,
        'ci_lower_of_diff': ci_diff_lower,
        'ci_upper_of_diff': ci_diff_upper,
        'n_variants': np.repeat(parameter_estimates.shape[0] // 2, parameter_estimates.shape[0]),
        'parameter_type': np.concatenate([np.repeat('growth_rate', parameter_estimates.shape[0] // 2), np.repeat('log_initial_freq', parameter_estimates.shape[0] // 2)])
    })

    return df

def simulate_estimate_evaluate(n_variants = 50, delta_gr_range = 0.2, new_var_rate = 1/14, freq_entering_variants = 0.0001, n_samples = 1000, s_0 = 0., o_0 = 0., reorder=True, seed = 5,
                               estimation_method = 'BFGS',
                               partition_size = None, overlap_size = None,
                               iterations = None, learning_rate = None,
                               plot_results = False, variant_count_percentage = 0.1):
    
    data = create_count_data(n_variants = n_variants, delta_gr_range = delta_gr_range, new_var_rate = new_var_rate, freq_entering_variants = freq_entering_variants, n_samples = n_samples, s_0 = s_0, o_0 = o_0, reorder=reorder, seed = seed)
    
    if estimation_method == 'BFGS':
        estimator = BFGSCompositionEstimator(data['counts'])
    elif estimation_method == 'stepwiseBFGS':
        estimator = StepwiseBFGSCompositionEstimator(data['counts'], partition_size=partition_size, overlap_size=overlap_size) # should overlap size depend on number of variants (possibly write function that determines overlap size based on co-occurence matrix)
    elif estimation_method == 'evofr':
        estimator = EvofrCompositionEstimator(data['counts'], iterations = iterations, learning_rate= learning_rate, generation_time=1)

    estimator.fit()
    result = estimator.get_results()
    df = evaluate_result(result['growth_rate_estimate'], result['log_init_freq_estimate'], result['hess_inv'], data['growth_rates'], data['log_init_freq'])
    df['mean_entropy'] = data['entropy_data']['mean_entropy']

    variant_counts = np.sum(data['counts'], axis = 0)
    df['variant_counts'] = np.tile(variant_counts, 2)

    variant_count_threshold = n_samples * variant_count_percentage
    #variant_count_threshold = 100
    df_reduced = evaluate_result_high_counts(result['growth_rate_estimate'], result['log_init_freq_estimate'], result['hess_inv'], data['growth_rates'], data['log_init_freq'], variant_counts, variant_count_threshold=variant_count_threshold)

    df['n_days'] = np.repeat(data['counts'].shape[0], df.shape[0])
    df_reduced['n_days'] = np.repeat(data['counts'].shape[0], df_reduced.shape[0])

    if plot_results:
        plot_viral_composition_dual(data['counts'], data['freq'], result['composition_estimate'], y_range = (1e-5, 2))
        plot_confidence_intervals_deviation(df)
        plot_confidence_intervals_deviation(df, plot_differences=True)

    return df, df_reduced

def run_sampling_experiment(
    variable_to_vary: str,
    variable_values: list,
    estimation_methods: list = ['BFGS', 'stepwiseBFGS', 'evofr'],
    n_variants=50,
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
    alpha = 1,
    plot_results=False,
    variant_count_percentage = 0.1
):
    all_results = []


    for i, value in enumerate(variable_values):

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
                    'delta_gr_range': delta_gr_range,
                    'new_var_rate': new_var_rate,
                    'freq_entering_variants': freq_entering_variants,
                    'n_samples': n_samples,
                    'estimation_method': method,
                    'seed': current_seed,
                    'plot_results': plot_results,
                    'variant_count_percentage': variant_count_percentage
                }

                # Override the varied parameter
                params[variable_to_vary] = value

                # Simulate
                df, df_reduced = simulate_estimate_evaluate(**params, **method_kwargs)
                mse_gr = np.nanmean(df[df.parameter_type == 'growth_rate']['difference_to_next_se'])
                mse_lif = np.nanmean(df[df.parameter_type != 'growth_rate']['difference_to_next_se'])
                mse_gr_reduced = np.nanmean(df_reduced[df_reduced.parameter_type == 'growth_rate']['difference_to_next_se'])
                mse_lif_reduced = np.nanmean(df_reduced[df_reduced.parameter_type != 'growth_rate']['difference_to_next_se'])

                all_results.append({
                    'method': method,
                    'gr_mse': mse_gr,
                    'gr_mse_reduced': mse_gr_reduced,
                    'lif_mse': mse_lif,
                    'lif_mse_reduced': mse_lif_reduced,
                    'n_samples': params['n_samples'],
                    'n_variants_requested': params['n_variants'],
                    'n_variants_real': df.n_variants[0],
                    'mean_entropy': df.mean_entropy[0],
                    'new_var_rate': params['new_var_rate'],
                    'n_days': df.n_days[0]
                })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    n_variants_real_df = df.groupby(['n_samples', 'n_variants_requested', 'new_var_rate']).agg(
        n_variants_real_mean=('n_variants_real', 'mean'),
        mean_entropy_mean=('mean_entropy', 'mean'),
        n_days_mean=('n_days', 'mean') 
    ).reset_index()

    # Group and aggregate by *requested* number of variants
    grouped = df.groupby(['n_samples', 'n_variants_requested', 'new_var_rate', 'method']).agg(
        gr_mse_mean=('gr_mse', 'mean'),
        gr_mse_mean_reduced=('gr_mse_reduced', 'mean'),
        gr_mse_std=('gr_mse', 'std'),
        gr_mse_std_reduced=('gr_mse_reduced', 'std'),
        lif_mse_mean=('lif_mse', 'mean'),
        lif_mse_mean_reduced=('lif_mse_reduced', 'mean'),
        lif_mse_std=('lif_mse', 'std'),
        lif_mse_std_reduced=('lif_mse_reduced', 'std'),
    ).reset_index()

    # Pivot for mean
    df_mean = grouped.pivot(index=['n_samples', 'n_variants_requested', 'new_var_rate'], columns='method', values=['gr_mse_mean','gr_mse_mean_reduced', 'lif_mse_mean', 'lif_mse_mean_reduced'])
    df_mean.columns = [f"{metric}_{method}" for metric, method in df_mean.columns]

    # Pivot for std
    df_std = grouped.pivot(index=['n_samples', 'n_variants_requested', 'new_var_rate'], columns='method', values=['gr_mse_std', 'gr_mse_std_reduced','lif_mse_std', 'lif_mse_std_reduced'])
    df_std.columns = [f"{metric}_{method}" for metric, method in df_std.columns]

    # Combine mean and std
    df_final = pd.concat([df_mean, df_std], axis=1).reset_index()

    # Merge mean realized variants cleanly
    df_final = df_final.merge(n_variants_real_df, on=['n_samples', 'n_variants_requested', 'new_var_rate'], how='left')

    return df_final, df


def create_pango_clade_mapping(data_clades, result_clades, data_pango, result_pango, path = "data/seq_to_clade_mapping.tsv", save_df = False, path_to_save = "data/covid_results/pango_clade_mapping_parameters.csv"):

    pango_clade_mapping = pd.read_csv(path, sep="\t")

    # Create mapping dictionaries
    pango_growth_dict = dict(zip(data_pango['variant_names'], result_pango['growth_rate_estimate']))
    clade_growth_dict = dict(zip(data_clades['variant_names'], result_clades['growth_rate_estimate']))

    pango_freq_dict = dict(zip(data_pango['variant_names'], result_pango['log_init_freq_estimate']))
    clade_freq_dict = dict(zip(data_clades['variant_names'], result_clades['log_init_freq_estimate']))

    pango_time_dict = dict(zip(data_pango['variant_names'], data_pango['mean_time']))
    clade_time_dict = dict(zip(data_clades['variant_names'], data_clades['mean_time']))

    # Map the growth rates into the DataFrame
    pango_clade_mapping['pango_growth'] = pango_clade_mapping['seqName'].map(pango_growth_dict)
    pango_clade_mapping['clade_growth'] = pango_clade_mapping['clade'].map(clade_growth_dict)
    pango_clade_mapping['pango_lif'] = pango_clade_mapping['seqName'].map(pango_freq_dict)
    pango_clade_mapping['clade_lif'] = pango_clade_mapping['clade'].map(clade_freq_dict)
    pango_clade_mapping['pango_time'] = pango_clade_mapping['seqName'].map(pango_time_dict)
    pango_clade_mapping['clade_time'] = pango_clade_mapping['clade'].map(clade_time_dict)

    pango_clade_mapping = pango_clade_mapping.dropna(subset=['pango_growth', 'clade_growth', 'pango_lif', 'clade_lif'])
    pango_clade_mapping = pango_clade_mapping.sort_values(['clade_time', 'pango_time'])

    if save_df:
        pango_clade_mapping.to_csv(path_to_save, index = False)

    return pango_clade_mapping

