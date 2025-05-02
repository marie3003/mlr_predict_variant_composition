import pandas as pd
import numpy as np

from viral_variant_comp.estimate import BFGSCompositionEstimator, StepwiseBFGSCompositionEstimator, EvofrCompositionEstimator
from viral_variant_comp.simulate_count_data import create_count_data
from viral_variant_comp.plotting import plot_confidence_intervals_deviation, plot_viral_composition_dual

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
    df['mean_entropy'] = data['entropy_data']['mean_entropy']

    if plot_results:
        plot_viral_composition_dual(data['counts'], data['freq'], result['composition_estimate'], y_range = (1e-5, 2))
        plot_confidence_intervals_deviation(df)
        plot_confidence_intervals_deviation(df, plot_differences=True)

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
        scaling_factors = np.round(np.linspace(21, 21-len(variable_values)+1, len(variable_values))).astype(int)
        days_vec = np.array(variable_values) * scaling_factors
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
                    'n_variants_requested': params['n_variants'],
                    'n_variants_real': df.n_variants[0],
                    'mean_entropy': df.mean_entropy[0],
                    'new_var_rate': params['new_var_rate'],
                    'n_days': params['n_days']
                })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    n_variants_real_df = df.groupby(['n_samples', 'n_variants_requested', 'new_var_rate', 'n_days']).agg(
        n_variants_real_mean=('n_variants_real', 'mean'),
        mean_entropy_mean=('mean_entropy', 'mean')
    ).reset_index()

    # Group and aggregate by *requested* number of variants
    grouped = df.groupby(['n_samples', 'n_variants_requested', 'new_var_rate', 'n_days', 'method']).agg(
        gr_mse_mean=('gr_mse', 'mean'),
        gr_mse_std=('gr_mse', 'std'),
        lif_mse_mean=('lif_mse', 'mean'),
        lif_mse_std=('lif_mse', 'std')
    ).reset_index()

    # Pivot for mean
    df_mean = grouped.pivot(index=['n_samples', 'n_variants_requested', 'new_var_rate', 'n_days'], columns='method', values=['gr_mse_mean', 'lif_mse_mean'])
    df_mean.columns = [f"{metric}_{method}" for metric, method in df_mean.columns]

    # Pivot for std
    df_std = grouped.pivot(index=['n_samples', 'n_variants_requested', 'new_var_rate', 'n_days'], columns='method', values=['gr_mse_std', 'lif_mse_std'])
    df_std.columns = [f"{metric}_{method}" for metric, method in df_std.columns]

    # Combine mean and std
    df_final = pd.concat([df_mean, df_std], axis=1).reset_index()

    # Merge mean realized variants cleanly
    df_final = df_final.merge(n_variants_real_df, on=['n_samples', 'n_variants_requested', 'new_var_rate', 'n_days'], how='left')

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

