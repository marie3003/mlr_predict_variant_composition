import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors

from viral_variant_comp.estimate import calculate_mean_fitness

### PLOT viral composition data

def plot_viral_composition(counts, freq = None, composition_estimate = None, var_names = None, y_range = (1e-5, 2), logscale = True):

    plt.figure(figsize=(counts.shape[0] // 50, 6))
    
    base_colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf"   # cyan
    ]

    brighter_colors = [
    "#6baed6",  # lighter blue
    "#ffae6b",  # lighter orange
    "#66c266",  # lighter green
    "#ff6666",  # lighter red
    "#c2a5e2",  # lighter purple
    "#b38f87",  # lighter brown
    "#f7a6d8",  # lighter pink
    "#bfbfbf",  # lighter gray
    "#d4e157",  # lighter olive
    "#66ddee"   # lighter cyan
    ]

    n_samples = np.sum(counts, axis = 1)
    rel_abund = counts / n_samples[:, np.newaxis]

    if var_names is None:
        var_names = [f'Variant {i+1}' for i in range(counts.shape[1])]

    for i in range(counts.shape[1]):

        plt.scatter(np.arange(counts.shape[0]), rel_abund[:, i], s= 10, alpha = 0.2, label = var_names[i], color = base_colors[i % 10])
        if freq is not None:
            plt.plot(np.arange(freq.shape[0]), freq[:, i], color = base_colors[i % 10])
        if composition_estimate is not None:
            plt.plot(np.arange(composition_estimate.shape[0]), composition_estimate[:, i], color = brighter_colors[i % 10], linestyle='--')

    plt.xlabel("Days")
    plt.ylabel("Abundancy [%]")
    if(logscale):
        plt.yscale('log')
        plt.ylim(y_range)
    plt.title("Disease Variants in Population")

    if(counts.shape[1] > 10):
        plt.legend(ncol = counts.shape[1] // 10, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    else:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_viral_composition_dual(counts, freq=None, composition_estimate=None, var_names=None, y_range=(1e-5, 2), show_legend = True):

    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    brighter_colors = [
        "#6baed6", "#ffae6b", "#66c266", "#ff6666", "#c2a5e2",
        "#b38f87", "#f7a6d8", "#bfbfbf", "#d4e157", "#66ddee"
    ]

    n_samples = np.sum(counts, axis=1)
    rel_abund = counts / n_samples[:, np.newaxis]

    if var_names is None:
        var_names = [f'Variant {i+1}' for i in range(counts.shape[1])]

    fig, axes = plt.subplots(2, 1, figsize=(counts.shape[0] // 50 + 4, 10), sharex=True)

    for ax_idx, logscale in enumerate([False, True]):
        ax = axes[ax_idx]

        for i in range(counts.shape[1]):
            ax.scatter(np.arange(counts.shape[0]), rel_abund[:, i], s=10, alpha=0.2, label=var_names[i], color=base_colors[i % 10])
            if freq is not None:
                ax.plot(np.arange(freq.shape[0]), freq[:, i], color=base_colors[i % 10])
            if composition_estimate is not None:
                ax.plot(np.arange(composition_estimate.shape[0]), composition_estimate[:, i], color=brighter_colors[i % 10], linestyle='--')

        ax.set_ylabel("Abundancy [%]")
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(y_range)
            ax.set_title("Disease Variants in Population (log scale)")
        else:
            ax.set_title("Disease Variants in Population (linear scale)")

        ax.grid(True)

    axes[-1].set_xlabel("Days")

    if show_legend:
        if counts.shape[1] > 10:
            axes[-1].legend(ncol=counts.shape[1] // 10, loc='upper center', bbox_to_anchor=(0.5, -0.25))
        else:
            axes[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))

    plt.tight_layout()
    plt.show()


def plot_viral_composition_interactive(counts, freq=None, composition_estimate=None, var_names=None, logscale=True):
    
    n_samples = np.sum(counts, axis=1)
    rel_abund = counts / n_samples[:, np.newaxis]

    if var_names is None:
        var_names = [f'Variant {i+1}' for i in range(counts.shape[1])]

    days = np.arange(counts.shape[0])
    n_variants = counts.shape[1]

    # Get color palette (e.g. 20 Plotly colors)
    colors = plotly.colors.qualitative.Plotly * ((n_variants // len(plotly.colors.qualitative.Plotly)) + 1)

    fig = go.Figure()

    for i in range(n_variants):
        color = colors[i % len(colors)]

        # Counts (scatter)
        fig.add_trace(go.Scatter(
            x=days,
            y=rel_abund[:, i],
            mode='markers',
            name=var_names[i],
            marker=dict(size=6, opacity=0.4, color=color),
            hovertemplate=(
                f"Variant: {var_names[i]}<br>" +
                "Day: %{x}<br>" +
                "Abundance: %{y:.2%}<extra></extra>"
            )
        ))

        # Frequency line (optional)
        if freq is not None:
            fig.add_trace(go.Scatter(
                x=days,
                y=freq[:, i],
                mode='lines',
                line=dict(width=2, dash = 'solid', color=color),
                name=f"{var_names[i]} (freq)",
                showlegend=False,
                hoverinfo='skip'
            ))

        # Composition estimate line (optional, dashed)
        if composition_estimate is not None:
            fig.add_trace(go.Scatter(
                x=days,
                y=composition_estimate[:, i],
                mode='lines',
                line=dict(width=2, dash='dot', color=color),
                name=f"{var_names[i]} (estimate)",
                showlegend=False,
                hoverinfo='skip'
            ))

    fig.update_layout(
        title="Disease Variants in Population",
        xaxis_title="Days",
        yaxis_title="Abundancy [%]",
        yaxis_type='log' if logscale else 'linear',
        template='plotly_white',
        hovermode='closest',
    )

    fig.show()



### EVALUATE ESTIMATION RESULTS

def plot_confidence_intervals(param_df):

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    gr_df = param_df[param_df.parameter_type == 'growth_rate']
    lif_df = param_df[param_df.parameter_type == 'log_initial_freq']

    # Plot for s_i (first half)
    max_gr = np.max(gr_df.parameter_estimate)
    growth_rate_range = (np.min(gr_df.parameter_estimate) - 0.5 * max_gr, max_gr + 0.5 * max_gr)

    for i in range(len(gr_df.parameter_estimate)):
        color = "green" if gr_df.ci_lower[i] <= gr_df.true_parameter[i] <= gr_df.ci_upper[i] else "red"
        axes[0].errorbar(i, gr_df.parameter_estimate[i], yerr=gr_df.standard_error[i], fmt='o', capsize=5, color='black')
        axes[0].plot(i, gr_df.true_parameter[i], 'x', color=color, markersize=10, label='True' if i == 0 else "")

    axes[0].set_title("Growth rate (s_i)")
    axes[0].set_ylim(growth_rate_range)
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Estimated value")
    axes[0].grid(True)

    # Plot for o_i (second half)
    min_lif = np.min(lif_df.parameter_estimate)
    freq_range = (min_lif + 0.5 * min_lif, np.max(lif_df.parameter_estimate) - 0.5 * min_lif)

    for i in range(len(lif_df.parameter_estimate)):
        idx = i + len(lif_df.parameter_estimate)
        color = "green" if lif_df.ci_lower[idx] <= lif_df.true_parameter[idx] <= lif_df.ci_upper[idx] else "red"
        axes[1].errorbar(i, lif_df.parameter_estimate[idx], yerr=lif_df.standard_error[idx], fmt='o', capsize=5, color='black')
        axes[1].plot(i, lif_df.true_parameter[idx], 'x', color=color, markersize=10, label='True' if i == 0 else "")

    axes[1].set_title("Log. initial frequencies (o_i)")
    axes[1].set_xlabel("Index")
    axes[1].set_ylim(freq_range)
    axes[1].set_ylabel("Estimated value")
    axes[1].grid(True)

    legend_elements = [
            Line2D([0], [0], marker='o', color='black', linestyle='None', label='Parameter estimate'),
            Line2D([0], [0], marker='x', color='green', linestyle='None', label='True parameter inside CI', markersize=10),
            Line2D([0], [0], marker='x', color='red', linestyle='None', label='True parameter outside CI', markersize=10)
        ]
    fig.legend(handles=legend_elements, loc="upper right", ncol=1)

    plt.suptitle("95% Confidence Intervals for Parameters with Ground Truth", fontsize = 16)
    plt.show()


def plot_confidence_intervals_deviation(param_df, plot_differences=False):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
    gr_df = param_df[param_df.parameter_type == 'growth_rate'].reset_index(drop=True)
    lif_df = param_df[param_df.parameter_type == 'log_initial_freq'].reset_index(drop=True)

    def plot_single_panel(ax, df, param_label, diff=False):
        for i in range(len(df)):
            if diff and pd.isna(df.difference_to_next_estimate.iloc[i]):
                continue

            if diff:
                estimate = df.difference_to_next_estimate.iloc[i]
                true_val = df.difference_to_next_true.iloc[i]
                err = df.standard_error_of_diff.iloc[i]
                ci_low = df.ci_lower_of_diff.iloc[i]
                ci_up = df.ci_upper_of_diff.iloc[i]
            else:
                estimate = df.parameter_estimate.iloc[i]
                true_val = df.true_parameter.iloc[i]
                err = df.standard_error.iloc[i]
                ci_low = df.ci_lower.iloc[i]
                ci_up = df.ci_upper.iloc[i]

            color = "green" if ci_low <= true_val <= ci_up else "red"
            ax.errorbar(i, estimate, yerr=err, fmt='o', capsize=5, color='black')
            ax.plot(i, true_val, 'x', color=color, markersize=10)

        ax.set_xlabel("Index")
        ax.set_ylabel("Difference" if diff else "Estimated value")
        ax.set_title(f"{param_label} {'(Δ)' if diff else ''}")
        ax.grid(True)

    def plot_deviation_panel(ax, df, param_label, diff=False):
        for i in range(len(df)):
            if diff and pd.isna(df.difference_to_next_estimate.iloc[i]):
                continue

            if diff:
                deviation = df.difference_to_next_estimate.iloc[i] - df.difference_to_next_true.iloc[i]
                err = df.standard_error_of_diff.iloc[i]
                ci_low = df.ci_lower_of_diff.iloc[i]
                ci_up = df.ci_upper_of_diff.iloc[i]
                true_val = df.difference_to_next_true.iloc[i]
            else:
                deviation = df.deviation.iloc[i]
                err = df.standard_error.iloc[i]
                ci_low = df.ci_lower.iloc[i]
                ci_up = df.ci_upper.iloc[i]
                true_val = df.true_parameter.iloc[i]

            point_color = "green" if ci_low <= true_val <= ci_up else "red"

            ax.errorbar(i, 0, yerr=err, capsize=5, color='black', zorder=1)
            ax.plot(i, deviation, 'o', color=point_color, markersize=6, zorder=2)

        ax.axhline(0, linestyle='--', color='gray')
        ax.set_title(f"Deviation {'(Δ)' if diff else ''}: Estimated - True ({param_label})")
        ax.set_xlabel("Index")
        ax.set_ylabel("Deviation")
        ax.grid(True)

    # Top row: point estimates
    plot_single_panel(axes[0, 0], gr_df, "Growth rate (s_i)", diff=plot_differences)
    plot_single_panel(axes[0, 1], lif_df, "Log. initial frequencies (o_i)", diff=plot_differences)

    # Bottom row: deviations
    plot_deviation_panel(axes[1, 0], gr_df, "s_i", diff=plot_differences)
    plot_deviation_panel(axes[1, 1], lif_df, "o_i", diff=plot_differences)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', label='Estimate inside CI'),
        Line2D([0], [0], marker='x', color='green', linestyle='None', label='True parameter inside CI', markersize=10),
        Line2D([0], [0], marker='x', color='red', linestyle='None', label='True parameter outside CI', markersize=10),
        Line2D([0], [0], marker='o', color='green', linestyle='None', label='Deviation inside CI'),
        Line2D([0], [0], marker='o', color='red', linestyle='None', label='Deviation outside CI'),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5)
    plt.suptitle(
        "95% Confidence Intervals and Deviations of " + 
        ("Parameter Differences (Δθ)" if plot_differences else "Parameter Estimates"),
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_mse_sampling_size(df, methods=['BFGS', 'stepwiseBFGS']):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    targets = {
        'gr': 'Growth Rate',
        'lif': 'Log Initial Frequency'
    }

    markers = ['s', 'o','^', 'D', 'v', '*', 'P', 'X']
    colors = [
    "#a6444f",  # reddish
    "#57a8b8",  # teal
    "#80557e",  # purple
    "#b5d2f2",  # light blue
    "#d991b4",  # pink
    "#397398",  # dark blue
    "#7394c2",  # mid blue
    "#7a7a7a"]   # gray

    for ax, target in zip(axes, targets.keys()):
        for idx, method in enumerate(methods):
            mean_col = f"{target}_mse_mean_{method}"
            std_col = f"{target}_mse_std_{method}"

            # Plot as unconnected points with error bars
            ax.errorbar(
                df['n_samples'],
                df[mean_col],
                yerr=df[std_col],
                fmt=markers[idx % len(markers)],              # point only (no line)
                capsize=5,
                color=colors[idx % len(colors)],
                label=method
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Δ{targets[target]} MSE vs. Sampling Size")
        ax.set_xlabel("Number of samples per day")
        ax.set_ylabel("Mean Squared Error")
        ax.grid(True)
        ax.legend(title="Estimation Method")

    plt.tight_layout()
    plt.show()

def plot_mse_n_variants(df, methods=['BFGS', 'stepwiseBFGS']):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    targets = {
        'gr': 'Growth Rate',
        'lif': 'Log Initial Frequency'
    }

    markers = ['s', 'o', '^', 'D', 'v', '*', 'P', 'X']
    colors = [
    "#a6444f",  # reddish
    "#57a8b8",  # teal
    "#80557e",  # purple
    "#b5d2f2",  # light blue
    "#d991b4",  # pink
    "#397398",  # dark blue
    "#7394c2",  # mid blue
    "#7a7a7a"]   # gray

    for ax, target in zip(axes, targets.keys()):
        for idx, method in enumerate(methods):
            mean_col = f"{target}_mse_mean_{method}"
            std_col = f"{target}_mse_std_{method}"

            ax.errorbar(
                df['n_variants_real_mean'],
                df[mean_col],
                yerr=df[std_col],
                fmt=markers[idx % len(markers)],
                capsize=5,
                color=colors[idx % len(colors)],
                label=method
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Δ{targets[target]} MSE vs. Number of Variants")
        ax.set_xlabel("Number of variants")
        ax.set_ylabel("Mean Squared Error")
        ax.grid(True)
        ax.legend(title="Estimation Method")

    plt.tight_layout()
    plt.show()

def plot_mse_variant_appearance(df, methods=['BFGS', 'stepwiseBFGS'], x_col= 'new_var_rate'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    targets = {
        'gr': 'Growth Rate',
        'lif': 'Log Initial Frequency'
    }

    markers = ['s', 'o', '^', 'D', 'v', '*', 'P', 'X']
    colors = [
        "#a6444f",  # reddish
        "#57a8b8",  # teal
        "#80557e",  # purple
        "#b5d2f2",  # light blue
        "#d991b4",  # pink
        "#397398",  # dark blue
        "#7394c2",  # mid blue
        "#7a7a7a"   # gray
    ]

    for ax, target in zip(axes, targets.keys()):
        for idx, method in enumerate(methods):
            mean_col = f"{target}_mse_mean_{method}"
            std_col = f"{target}_mse_std_{method}"

            ax.errorbar(
                df[x_col],
                df[mean_col],
                yerr=df[std_col],
                fmt=markers[idx % len(markers)],
                capsize=5,
                color=colors[idx % len(colors)],
                label=method
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        if(x_col == 'mean_entropy_mean'):
            ax.set_title(f"Δ{targets[target]} MSE vs. Variant Diversity")
            ax.set_xlabel("Mean Entropy of variant compostition (freq.)")
        else:
            ax.set_title(f"Δ{targets[target]} MSE vs. Variant Appearance Rate")
            ax.set_xlabel("Variant appearance rate (per day)")
        ax.set_ylabel("Mean Squared Error")
        ax.grid(True)
        ax.legend(title="Estimation Method")

    plt.tight_layout()
    plt.show()


### PLOTS COVID DATA


def plot_clade_pango_over_time(clades_df, pangos_df, color_palette='range', plot_type='growth'):

    assert plot_type in ['growth', 'lif'], "plot_type must be 'growth' or 'lif'"

    if color_palette == 'range':    
        palette = sns.color_palette('husl', n_colors=len(clades_df))
        clade_colors = {clade: f'rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})' for clade, (r, g, b) in zip(clades_df['clade'], palette)}
    elif color_palette == 'jumps':
        palette = (px.colors.qualitative.Dark24 * (len(clades_df) // 24 + 1))[:len(clades_df)]
        clade_colors = {clade: color for clade, color in zip(clades_df['clade'], palette)}

    # --- Select column names & titles based on plot_type ---
    if plot_type == 'growth':
        clade_y = 'clade_growth'
        pango_y = 'pango_growth'
        y_title_clade = 'Clade Growth Rate'
        y_title_pango = 'Pango Growth Rate'
        hover_label = 'Growth'
        main_title = "Clade and Pango Lineage Growth Rates Over Time"
    else:
        clade_y = 'clade_lif'
        pango_y = 'pango_lif'
        y_title_clade = 'Clade Log Init Freq'
        y_title_pango = 'Pango Log Init Freq'
        hover_label = 'Log Init Freq'
        main_title = "Clade and Pango Lineage Log Initial Frequencies Over Time"

    # --- Subplots ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        subplot_titles=(y_title_clade + " over Time", y_title_pango + " over Time")
    )

    # --- Top plot: Clade ---
    for idx, row in clades_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['clade_time']],
                y=[row[clade_y]],
                mode='markers+text',
                marker=dict(color=clade_colors[row['clade']], size=10),
                text=[row['clade']],
                textposition='top center',
                textfont=dict(size=12),
                name=row['clade'],
                hovertemplate=f"Clade: {row['clade']}<br>Time: {row['clade_time']:.2f}<br>{hover_label}: {row[clade_y]:.2f}<extra></extra>"
            ),
            row=1, col=1
        )

    # --- Bottom plot: Pango ---
    for idx, row in pangos_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['pango_time']],
                y=[row[pango_y]],
                mode='markers',
                marker=dict(color=clade_colors.get(row['clade'], 'gray'), size=5),
                name=row['seqName'],
                hovertemplate=f"Pango: {row['seqName']}<br>Clade: {row['clade']}<br>Time: {row['pango_time']:.2f}<br>{hover_label}: {row[pango_y]:.2f}<extra></extra>",
                showlegend=False
            ),
            row=2, col=1
        )

    # --- Layout ---
    fig.update_layout(
        height=900,
        title_text=main_title,
        xaxis_title="Time (Clades)",
        yaxis_title=y_title_clade,
        xaxis2_title="Time (Pango Lineages)",
        yaxis2_title=y_title_pango,
        legend_title_text='Clade',
        legend=dict(
            orientation='h',
            y=-0.25,
            x=0.5,
            xanchor='center',
            title_text=None
        ),
        margin=dict(t=100, b=150)
    )

    fig.show()


def plot_mean_fitness_time(result_pango, result_clades):
    time = np.arange(1, result_pango['composition_estimate'].shape[0] + 1, 1)

    # Compute mean fitness over time
    pango_mean_fitness = calculate_mean_fitness(result_pango['composition_estimate'], result_pango['growth_rate_estimate'])
    clade_mean_fitness = calculate_mean_fitness(result_clades['composition_estimate'], result_clades['growth_rate_estimate'])

    # Compute change in mean fitness (delta)
    delta_pango = np.diff(pango_mean_fitness)
    delta_clade = np.diff(clade_mean_fitness)
    delta_time = time[1:]

    # Determine shared y-limits
    mf_min = min(pango_mean_fitness.min(), clade_mean_fitness.min())
    mf_max = max(pango_mean_fitness.max(), clade_mean_fitness.max())
    delta_min = min(delta_pango.min(), delta_clade.min())
    delta_max = max(delta_pango.max(), delta_clade.max())
    mf_add = 0.1
    delta_add = 0.001

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8), sharex='col')

    # Row 1: Mean fitness
    axes[0, 0].plot(time, pango_mean_fitness, color='tab:blue')
    axes[0, 0].set_ylabel('Mean Fitness')
    axes[0, 0].set_title('Pango Mean Fitness')
    axes[0, 0].set_ylim(mf_min - mf_add, mf_max + mf_add)

    axes[0, 1].plot(time, clade_mean_fitness, color='tab:green')
    axes[0, 1].set_ylabel('Mean Fitness')
    axes[0, 1].set_title('Clade Mean Fitness')
    axes[0, 1].set_ylim(mf_min - mf_add, mf_max + mf_add)

    # Row 2: ΔFitness
    axes[1, 0].plot(delta_time, delta_pango, color='tab:blue')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('ΔFitness')
    axes[1, 0].set_title('Δ Pango Mean Fitness')
    axes[1, 0].set_ylim(delta_min - delta_add, delta_max + delta_add)

    axes[1, 1].plot(delta_time, delta_clade, color='tab:green')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('ΔFitness')
    axes[1, 1].set_title('Δ Clade Mean Fitness')
    axes[1, 1].set_ylim(delta_min - delta_add, delta_max + delta_add)

    # Styling
    for ax in axes.flat:
        ax.grid(True)

    plt.tight_layout()
    plt.show()




### HELPERS

def plot_heatmap(matrix, title, x_label, y_label):

    mask = (matrix == 0)

    sns.heatmap(matrix, cmap='viridis', mask = mask)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()