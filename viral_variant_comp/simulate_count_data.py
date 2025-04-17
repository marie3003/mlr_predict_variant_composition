import numpy as np
import pandas as pd


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


def calculate_frequencies(n_days, growth_rates, log_initial_freq):

    freqs = np.zeros((n_days, len(growth_rates)))

    for d in range(n_days):
        logits = growth_rates * d + log_initial_freq
        max_l = np.max(logits)
        freqs[d] = np.exp(logits - max_l)   #log-sum-exp trick

    freqs = freqs / np.sum(freqs, axis = 1, keepdims= True)
    return freqs


def sample(frequencies, n_samples):
    samples = np.array([np.random.multinomial(n_samples, row) for row in frequencies])
    return samples

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

def reorder_variants_realdata(counts, var_names):

    variant_normalized_counts = counts / np.sum(counts, axis = 0)
    t_peaks = np.sum(variant_normalized_counts * np.arange(counts.shape[0])[:, np.newaxis], axis = 0)

    variant_position = np.argsort(t_peaks)
    counts = counts.copy()[:, variant_position]
    var_names = var_names.copy()[variant_position]
    
    return counts, var_names



def create_count_data(n_variants, n_days, delta_gr_range, new_var_rate, freq_entering_variants, n_samples, s_0, o_0, reorder = False):
    
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

    data = {"growth_rates": s_vec, "log_init_freq": o_vec, "freq": frequencies, "counts": counts, "n_variants": n_variants}
    
    if(reorder):
        data = reorder_variants(data)

    return data

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