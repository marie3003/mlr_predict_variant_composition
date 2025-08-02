import pandas as pd
import numpy as np

import cvxpy as cp
from scipy.linalg import cholesky

from matplotlib import pyplot as plt
import seaborn as sns

################# Prepare Covid Data #################

def create_substitutions_df(consensus_seq_df):
    """
    Create a DataFrame showing assignment of amino acid substitutions from parent lineages of different pango lineages.
    Lineages are sorted alphabetically.
    """
    substitutions = sorted(set(sub for sublist in consensus_seq_df.loc['aaSubstitutionsNew'] for sub in sublist)) #substitutions are sorted alphabetically
    if(substitutions[0] == ''):
        substitutions.remove('')


    def row_to_binary(sub_list):
        return pd.Series([1 if sub in sub_list else 0 for sub in substitutions], index=substitutions)

    substitutions_df = consensus_seq_df.loc['aaSubstitutionsNew'].apply(row_to_binary)
    
    return substitutions_df

def create_pango_lineage_information_df(pango_clade_mapping, consensus_seq_df):

    """
    Use covid fitness information and add information about parent lineages as well as growth advantage of child lineages compared to their parent lineage.
    Filter out lineages without matching parent lineage.
    """

    pango_clade_mapping = pango_clade_mapping.rename(columns={"seqName": "lineage"})

    # add parent lineage info
    parent_df = pd.DataFrame(consensus_seq_df.loc['parent'].values, columns=['parent'], index=consensus_seq_df.loc['lineage']).reset_index()
    pango_clade_mapping = pango_clade_mapping.merge(parent_df, on="lineage", how="left")

    # get vector of growth rate differences of child to parent pango lineages (some lineages have negative growth advantage compared to their parent)
    pango_clade_mapping['parent_growth_advantage'] = pango_clade_mapping['pango_growth'] - pango_clade_mapping['parent'].map(pango_clade_mapping.set_index('lineage')['pango_growth'])

    # filters out 50 lineage with no growth rate estimate of parent lineage
    pango_clade_mapping = pango_clade_mapping.dropna(subset=['parent_growth_advantage'])

    return pango_clade_mapping

def calculate_fitness_advantage_covariance(hess_inv, lineage_names, pango_clade_mapping):
    """
    Calculate the covariance of fitness advantage between child and parent lineages.
    Result is sorted as in pango_clade_mapping.
    """
    # only work with growth rates here and add 0 variance and covariance for pivot
    n = hess_inv.shape[0] // 2
    hess_inv = hess_inv[: n, : n]
    hess_inv_augmented = np.zeros((n + 1, n + 1))
    hess_inv_augmented[1:, 1:] = hess_inv
    n = n + 1
    
    # Create a mapping from lineage names to indices (lineage names has same sorting as hess_inv)
    name_to_index = {name: idx for idx, name in enumerate(lineage_names)}

    
    # Initialize the covariance matrix
    m = len(pango_clade_mapping)
    D = np.zeros((m, n))
    
    # Iterate through each row in the PANGO clade mapping
    for i, (_, row) in enumerate(pango_clade_mapping.iterrows()):
        child = row['lineage']
        parent = row['parent']
        
        D[i, name_to_index[child]] = 1
        D[i, name_to_index[parent]] = -1
    
    child_parent_covariance = D @ hess_inv_augmented @ D.T
    return child_parent_covariance


################# Estimate AA Substitution Impact #################

def estimate_aa_impact(delta_s, C, W, _lambda=0.1):
    """
    Estimate the impact of amino acid substitutions on fitness.
    """
    # Ensure C is positive definite
    eps = 1e-10
    C += eps * np.eye(C.shape[0])

    # Cholesky decomposition for numerical stability: C = L * L^T
    L = cholesky(C, lower=True)
    L_inv = np.linalg.inv(L)

    # Define optimization variable
    a = cp.Variable(W.shape[1])

    # Objective function
    #objective = cp.Minimize(cp.sum_squares(delta_s - W @ a))
    #objective = cp.Minimize(cp.quad_form(W @ a - delta_s, np.linalg.inv(C)) + _lambda * cp.norm(a, 1))
    objective = cp.Minimize((1 / delta_s.shape[0])* cp.sum_squares(L_inv @ ( W @ a - delta_s)) + _lambda * cp.norm(a, 1))

    # Solve the optimization problem
    prob = cp.Problem(objective)
    prob.solve(solver=cp.OSQP)

    print("Status:", prob.status)
    print("Optimal value of objective function:", prob.value)

    return {'a_est': a.value, 'objective_value': prob.value, 'status': prob.status}

def calculate_impact_estimates(consensus_seq_df, hess_inv, lineage_names, pango_clade_mapping, _lambda, diag_covariance=False, calc_max_lambda_loss = False):
    """
    Calculate the impact estimates of amino acid substitutions on fitness advantage. Final sorting of variants follows the order of pango clade mapping df
    """
    # df containing assignment of aa substitutions to pango lineages
    substitutions_df = create_substitutions_df(consensus_seq_df)

    # df containing pango lineage, clade, parent assignment and growth rate advantage
    lineage_info_df = create_pango_lineage_information_df(pango_clade_mapping, consensus_seq_df)

    # covariance of fitness advantage between child and parent lineages
    C = calculate_fitness_advantage_covariance(hess_inv, lineage_names, lineage_info_df)
    if diag_covariance:
        C = np.diag(np.diag(C))    # reduced complexity: only consider variance

    # define parameters for optimization
    lineages = lineage_info_df.lineage.values
    W = substitutions_df.loc[lineages].values # we know aa substitutions for all vairants with growth rate advantages
    print(W.shape)
    delta_s = lineage_info_df.parent_growth_advantage.values

    a_est_result = estimate_aa_impact(delta_s, C, W, _lambda=_lambda)

    if calc_max_lambda_loss:
        # calculate loss if lambfa = infinity
        max_lambda_loss = 1/delta_s.shape[0] * delta_s.transpose() @ np.linalg.inv(C) @ delta_s
        a_est_result['max_lambda_loss'] = max_lambda_loss
        print(f"Max lambda loss: {max_lambda_loss}")


    return a_est_result

def bootstrap_aa_impact_estimates(consensus_seq_df, hess_inv, lineage_names, pango_clade_mapping, _lambda, n_runs = 5, diag_covariance=True):
    """
    Calculate the impact estimates of amino acid substitutions on fitness advantage. Final sorting of variants follows the order of pango clade mapping df
    """
    # df containing assignment of aa substitutions to pango lineages
    substitutions_df = create_substitutions_df(consensus_seq_df)

    # df containing pango lineage, clade, parent assignment and growth rate advantage
    lineage_info_df = create_pango_lineage_information_df(pango_clade_mapping, consensus_seq_df)

    # covariance of fitness advantage between child and parent lineages
    C = calculate_fitness_advantage_covariance(hess_inv, lineage_names, lineage_info_df)
    if diag_covariance:
        C = np.diag(np.diag(C))    # reduced complexity: only consider variance

    # define parameters for optimization
    lineages = lineage_info_df.lineage.values
    W = substitutions_df.loc[lineages].values # we know aa substitutions for all vairants with growth rate advantages
    delta_s = lineage_info_df.parent_growth_advantage.values

    ######## BOOTSTRAP ########

    bootstrap_estimates = []
    n_samples = len(delta_s)
    for i in range(n_runs):
        # Sample indices with replacement (number of draws is equal to the number of samples)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        delta_s_bootstrap = delta_s[bootstrap_indices]
        W_bootstrap = W[bootstrap_indices, :]
        C_bootstrap = C[np.ix_(bootstrap_indices, bootstrap_indices)]
        
        eigenvalues = np.linalg.eigvals(C_bootstrap)
        print("Min eigenvalue:", np.min(eigenvalues))

        # Estimate amino acid impact for bootstrap sample
        a_est_result = estimate_aa_impact(delta_s_bootstrap, C_bootstrap, W_bootstrap, _lambda=_lambda)
        bootstrap_estimates.append(a_est_result['a_est'])
        print(f"Bootstrap run {i+1}:")
        print(a_est_result['objective_value'])

    bootstrap_array = np.stack(bootstrap_estimates)

    # Calculate statistics
    mean_est = np.mean(bootstrap_array, axis=0)
    std_est = np.std(bootstrap_array, axis=0)
    ci_lower = np.percentile(bootstrap_array, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_array, 97.5, axis=0)
    

    df = pd.DataFrame({
        'mean': mean_est,
        'std': std_est,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    df['aa_substitution'] = substitutions_df.columns
    df['n_events'] = substitutions_df.loc[lineages].sum(axis=0).values

    sort_indices = df['mean'].argsort()[::-1]  # descending; use [::1] for ascending
    df = df.iloc[sort_indices].reset_index(drop=True)
    bootstrap_array = bootstrap_array[:, sort_indices]

    return df, bootstrap_array


def calculate_impact_estimates_timesplit(consensus_seq_df, hess_inv, lineage_names, pango_clade_mapping, _lambda, clade_split, diag_covariance=False):
    """
    Calculate the impact estimates of amino acid substitutions on fitness advantage. Final sorting of variants follows the order of pango clade mapping df
    """
    # df containing assignment of aa substitutions to pango lineages
    substitutions_df = create_substitutions_df(consensus_seq_df)

    # df containing pango lineage, clade, parent assignment and growth rate advantage
    lineage_info_df = create_pango_lineage_information_df(pango_clade_mapping, consensus_seq_df)

    # covariance of fitness advantage between child and parent lineages
    C = calculate_fitness_advantage_covariance(hess_inv, lineage_names, lineage_info_df)
    if diag_covariance:
        C = np.diag(np.diag(C))    # reduced complexity: only consider variance

    # define parameters for optimization
    lineages = lineage_info_df.lineage.values
    W = substitutions_df.loc[lineages].values # we know aa substitutions for all vairants with growth rate advantages
    delta_s = lineage_info_df.parent_growth_advantage.values

    lineages_list = []
    aa_impact_time_df = pd.DataFrame({'aa_substitution': substitutions_df.columns,})
    aa_impact_time_df['n_events'] = substitutions_df.loc[lineages].sum(axis=0).values

    # split data by clade
    for i, clade_list in enumerate(clade_split):
        indices = lineage_info_df.index.get_indexer_for(lineage_info_df[lineage_info_df['clade'].isin(clade_list)].index)
        delta_s_period = delta_s[indices]
        W_period = W[indices, :]
        C_period = C[np.ix_(indices, indices)]

        a_est_result = estimate_aa_impact(delta_s_period, C_period, W_period, _lambda=_lambda)
        aa_impact_time_df[f'impact_clades{i+1}'] = a_est_result['a_est']

        aa_impact_time_df[f'events_clades{i+1}'] = substitutions_df.loc[lineages[indices]].sum(axis=0).values

        lineages_list.append(lineages[indices])

    aa_impact_time_df['max_impact'] = aa_impact_time_df.filter(like='impact', axis=1).max(axis=1)
    aa_impact_time_df.sort_values(by='max_impact', ascending=False, inplace=True)

    return aa_impact_time_df, lineages_list

################## Evaluate Results #################
def get_most_impactful_substitutions(aa_impact_vec, n, substitutions_df, lineage_info_df, abundancy_threshold = 10):
    aa_impact_df = pd.DataFrame({'aa_substitution': substitutions_df.columns, 'gr_impact': aa_impact_vec, "n_events": substitutions_df.loc[lineage_info_df.lineage.values].sum(axis = 0)})
    aa_impact_df = aa_impact_df[aa_impact_df.n_events > abundancy_threshold]
    return aa_impact_df.sort_values(by='gr_impact', ascending=False).head(n)


################# HELPERS #################

def calculate_root_mse(est_a, true_a):
    """
    Calculate the mean squared error between estimated and true amino acid impacts.
    """
    return np.sqrt(np.mean((est_a - true_a) ** 2))

def calculate_mae(est_a, true_a):
    """
    Calculate the mean absolute error between estimated and true amino acid impacts.
    """
    return np.mean(np.abs(est_a - true_a))