import numpy as np
import pandas as pd

### HELPER FUNCTIONS

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

def calculate_frequencies(n_days, growth_rates, log_initial_freq):

    t_vec = np.arange(n_days)
    freqs = np.zeros((n_days, len(growth_rates)))

    logits = np.outer(t_vec, growth_rates) + log_initial_freq     # shape T + 1 times n 
    max_logits = np.max(logits, axis = 1)    
    shifted_exp_logits = np.exp(logits - max_logits[:, np.newaxis])  # max_logits here vector of shape T+1 times 1
    
    freqs = shifted_exp_logits / np.sum(shifted_exp_logits, axis = 1, keepdims=True)
    return freqs

def calculate_entropy(freq, cutoff):
    log_freq = np.zeros_like(freq)
    log_freq[freq > 0] = np.log(freq[freq > 0])

    freq_log_freq = -1 * np.multiply(freq,log_freq)
    entropy_vec = np.sum(freq_log_freq, axis = 1)

    # cut end of entropy vector with cutoff to avoid end where no new variants are added biasing result
    indices = np.where(entropy_vec > cutoff)
    last_valid_idx = (np.max(indices) + 1) if indices[0].size > 0 else len(entropy_vec)
    last_valid_idx = np.min([last_valid_idx, len(entropy_vec)])
    entropy_vec = entropy_vec[:last_valid_idx]

    mean_entropy = np.mean(entropy_vec)
    return {"entropies": entropy_vec, "mean_entropy": mean_entropy, "n_days_included": last_valid_idx}

### SIMULATE COUNT DATA

def set_parameters(n_variants, delta_gr_range, rate, freq_entering_variants, s_0 = 0, o_0 = 0):

    s_vec = np.zeros(n_variants)
    o_vec = np.zeros(n_variants)

    s_vec[0] = s_0
    o_vec[0] = o_0

    var_count = 1
    t = 0

    while(var_count < n_variants):

        t += 1
        n_new_variants = min(np.random.poisson(rate), n_variants - var_count)
        
        if (n_new_variants > 0):
            
            logits = s_vec[0:var_count] * t + o_vec[0:var_count]
            max_l = np.max(logits)
            log_sum_exp = np.exp(logits - max_l)
            avg_s = np.dot(s_vec[0:var_count], log_sum_exp / np.sum(log_sum_exp))

            s_vec[var_count:var_count + n_new_variants] = avg_s + np.random.uniform(0, delta_gr_range, n_new_variants)
            o_vec[var_count:var_count + n_new_variants] = np.log(freq_entering_variants/(1 - freq_entering_variants)) +  np.log(np.sum(log_sum_exp)) + max_l - s_vec[var_count:var_count + n_new_variants] * t

            var_count += n_new_variants

    return s_vec, o_vec

def sample(frequencies, n_samples):
    samples = np.array([np.random.multinomial(n_samples, row) for row in frequencies])
    return samples

def create_count_data(n_variants, n_days, delta_gr_range, new_var_rate, freq_entering_variants, n_samples, s_0, o_0, reorder = True, seed=None, entropy_cutoff = 1e-4):

    if seed is not None:
        np.random.seed(seed)
    
    s_vec, o_vec = set_parameters(n_variants = n_variants, delta_gr_range = delta_gr_range, rate = new_var_rate, freq_entering_variants = freq_entering_variants, s_0 = s_0, o_0 = o_0)

    frequencies = calculate_frequencies(n_days, s_vec, o_vec)
    counts = sample(frequencies, n_samples)

    # remvove variants with all 0 counts
    non_zero_cols = ~(counts == 0).all(axis = 0)
    counts = counts[:, non_zero_cols]
    frequencies = frequencies[:, non_zero_cols]
    n_variants = np.sum(non_zero_cols)
    s_vec = s_vec[:n_variants]
    o_vec = o_vec[:n_variants]

    entropy_data = calculate_entropy(frequencies, entropy_cutoff)

    data = {"growth_rates": s_vec, "log_init_freq": o_vec, "freq": frequencies, "counts": counts, "n_variants": n_variants, "entropy_data": entropy_data}
    
    if(reorder):
        data = reorder_variants(data)

    return data


### PREPROCESSING EVOFR

def convert_count_data_to_df(counts):
    # Convert to DataFrame with column names as variants
    counts_df = pd.DataFrame(counts, columns=[f"variant_{i}" for i in range(counts.shape[1])])
    counts_df["day"] = counts_df.index

    # Melt to long format
    counts_df_long = counts_df.melt(id_vars="day", var_name="variant", value_name="count")
    counts_df_long.to_csv('data/counts_5var_180day.csv', index = False)
    return counts_df_long

def prepare_count_data_evofr(counts):

    counts_df = convert_count_data_to_df(counts)
    
    start_date = pd.to_datetime("2025-01-01")
    dates = [start_date + pd.Timedelta(days=day - 1) for day in counts_df.day]

    counts_df['day'] = dates

    counts_df = counts_df.rename(columns = {"day": "date", "count": "sequences"})

    counts_df = counts_df[counts_df.sequences != 0]

    return counts_df


### COVID DATA

def reorder_variants_realdata(counts, var_names):

    variant_normalized_counts = counts / np.sum(counts, axis = 0)
    t_peaks = np.sum(variant_normalized_counts * np.arange(counts.shape[0], dtype=float)[:, np.newaxis], axis=0)

    variant_position = np.argsort(t_peaks)
    counts = counts.copy()[:, variant_position]
    var_names = var_names.copy()[variant_position]
    t_peaks = t_peaks[variant_position]
    
    return counts, var_names, t_peaks

def preprocess_covid_data(counts_df, grouping_col = 'nextstrainClade'):
    counts_df['date'] = pd.to_datetime(counts_df['date'])
    counts_pivot = counts_df.pivot_table(index='date', columns=grouping_col, values='count', aggfunc='sum', fill_value=0).sort_index()  #ignores rows with nan values in grouping col (same values are nan for pango lineage and clade)
    if grouping_col == 'nextstrainClade':
        counts_pivot = counts_pivot.drop('recombinant', axis = 1)
    counts_matrix = counts_pivot.to_numpy()

    counts_matrix, var_names, t_peaks = reorder_variants_realdata(counts_matrix, counts_pivot.columns)
    
    return {'counts_df': counts_df, 'counts_df_pivot': counts_pivot, 'counts': counts_matrix, 'variant_names': var_names, 'n_variants': counts_matrix.shape[1], 'mean_time': t_peaks}