import numpy as np



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


def create_count_data(n_variants, n_days, delta_gr_range, new_var_rate, freq_entering_variants, n_samples, s_0, o_0):
    
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

    return {"growth_rates": s_vec, "log_init_freq": o_vec, "freq": frequencies, "counts": counts, "n_variants": n_variants}