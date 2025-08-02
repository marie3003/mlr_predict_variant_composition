import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.patches import Patch
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


def plot_viral_composition_dual(counts, freq=None, composition_estimate=None,
                                 var_names=None, y_range=(1e-5, 2),
                                 show_legend=True, start_date=None, path_to_save=None, title = "Disease Variants in Population", color_scheme='colorful'):

    #base_colors = [
    #    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    #    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    #]
    #brighter_colors = [
    #    "#6baed6", "#ffae6b", "#66c266", "#ff6666", "#c2a5e2",
    #    "#b38f87", "#f7a6d8", "#bfbfbf", "#d4e157", "#66ddee"
    #]
    if color_scheme == 'reds':
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
    elif color_scheme == 'colorful':
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]

    n_samples = np.sum(counts, axis=1)
    rel_abund = counts / n_samples[:, np.newaxis]
    days = np.arange(counts.shape[0])

    if var_names is None:
        var_names = [f'Variant {i+1}' for i in range(counts.shape[1])]

    # Create time labels
    if start_date is not None:
        time_labels = [datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=int(day)) for day in days]
    else:
        time_labels = days

    # Presentation-level font sizes
    title_fontsize = 24
    label_fontsize = 20
    tick_fontsize = 18
    legend_fontsize = 12

    fig, axes = plt.subplots(2, 1, figsize=(counts.shape[0] // 50 + 6, 12), sharex=True)

    for ax_idx, logscale in enumerate([False, True]):
        ax = axes[ax_idx]

        for i in range(counts.shape[1]):
            ax.scatter(time_labels, rel_abund[:, i], s=30, alpha=0.3,  # bigger points
                       label=var_names[i], color=colors[i % len(colors)])
            if freq is not None:
                ax.plot(time_labels, freq[:, i], color=colors[i % len(colors)], linewidth=2)
            if composition_estimate is not None:
                ax.plot(time_labels, composition_estimate[:, i],
                        color=colors[i % len(colors)], linestyle='--', linewidth=2)

        ax.set_ylabel("Abundancy [%]", fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(y_range)
            ax.set_title(f"{title} (log scale)", fontsize=title_fontsize)
        else:
            ax.set_title(f"{title} (linear scale)", fontsize=title_fontsize)

        ax.grid(True)

    axes[-1].set_xlabel("Date" if start_date else "Days", fontsize=label_fontsize)
    axes[-1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Format x-axis with ticks every 3 months if dates are used
    if start_date:
        axes[-1].xaxis.set_major_locator(MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(DateFormatter('%b %Y'))
        fig.autofmt_xdate()

    if show_legend:
        ncol_legend = max(1, counts.shape[1] // 10)
        axes[-1].legend(ncol=ncol_legend, loc='upper center', bbox_to_anchor=(0.5, -0.35), fontsize=legend_fontsize)

    plt.tight_layout()
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=500, format = 'png')
    plt.show()


def plot_viral_composition_interactive(counts, freq=None, composition_estimate=None,
                                       var_names=None, logscale=True, start_date=None):

    n_samples = np.sum(counts, axis=1)
    rel_abund = counts / n_samples[:, np.newaxis]

    if var_names is None:
        var_names = [f'Variant {i+1}' for i in range(counts.shape[1])]

    days = np.arange(counts.shape[0])
    if start_date is not None:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        x_vals = [start + timedelta(days=int(d)) for d in days]
        xaxis_title = "Date"
    else:
        x_vals = days
        xaxis_title = "Days"

    n_variants = counts.shape[1]
    colors = plotly.colors.qualitative.Plotly * ((n_variants // len(plotly.colors.qualitative.Plotly)) + 1)

    fig = go.Figure()

    for i in range(n_variants):
        color = colors[i % len(colors)]

        # Scatter points
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=rel_abund[:, i],
            mode='markers',
            name=var_names[i],
            marker=dict(size=6, opacity=0.4, color=color),
            hovertemplate=(
                f"Variant: {var_names[i]}<br>" +
                "Time: %{x|%b %d, %Y}<br>" +
                "Abundance: %{y:.2%}<extra></extra>"
            )
        ))

        if freq is not None:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=freq[:, i],
                mode='lines',
                line=dict(width=2, dash='solid', color=color),
                name=f"{var_names[i]} (freq)",
                showlegend=False,
                hoverinfo='skip'
            ))

        if composition_estimate is not None:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=composition_estimate[:, i],
                mode='lines',
                line=dict(width=2, dash='dot', color=color),
                name=f"{var_names[i]} (estimate)",
                showlegend=False,
                hoverinfo='skip'
            ))

    fig.update_layout(
        title="Disease Variants in Population",
        xaxis_title=xaxis_title,
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
        color = "#6c9a8b" if gr_df.ci_lower[i] <= gr_df.true_parameter[i] <= gr_df.ci_upper[i] else "#a6444f"
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
            Line2D([0], [0], marker='x', color="#6c9a8b", linestyle='None', label='True parameter inside CI', markersize=10),
            Line2D([0], [0], marker='x', color="#a6444f", linestyle='None', label='True parameter outside CI', markersize=10)
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

            color = "#6c9a8b" if ci_low <= true_val <= ci_up else "#a6444f"
            ax.errorbar(i, estimate, yerr=err, fmt='o', capsize=5, color='black')
            ax.plot(i, true_val, 'x', color=color, markersize=10)

        ax.set_xlabel("Index", fontsize=14)
        ax.set_ylabel("Difference" if diff else "Value", fontsize=14)
        ax.set_title(f"{param_label} {'(Δ)' if diff else ''}", fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
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

            point_color = "#6c9a8b" if ci_low <= true_val <= ci_up else "#a6444f"

            ax.errorbar(i, 0, yerr=err, capsize=5, color='black', zorder=1)
            ax.plot(i, deviation, 'o', color=point_color, markersize=6, zorder=2)

        ax.axhline(0, linestyle='--', color='gray')
        ax.set_title(f"Deviation {'(Δ)' if diff else ''}: Estimated - True ({param_label})", fontsize=16)
        ax.set_xlabel("Index", fontsize=14)
        ax.set_ylabel("Deviation", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)

    # Top row: point estimates
    plot_single_panel(axes[0, 0], gr_df, "Growth rate (s_i)", diff=plot_differences)
    plot_single_panel(axes[0, 1], lif_df, "Log. initial frequencies (o_i)", diff=plot_differences)

    # Bottom row: deviations
    plot_deviation_panel(axes[1, 0], gr_df, "s_i", diff=plot_differences)
    plot_deviation_panel(axes[1, 1], lif_df, "o_i", diff=plot_differences)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', label='Estimate'),
        Line2D([0], [0], marker='x', color="#6c9a8b", linestyle='None', label='True parameter inside CI', markersize=10),
        Line2D([0], [0], marker='x', color="#a6444f", linestyle='None', label='True parameter outside CI', markersize=10),
        Line2D([0], [0], marker='o', color="#6c9a8b", linestyle='None', label='Deviation inside CI'),
        Line2D([0], [0], marker='o', color="#a6444f", linestyle='None', label='Deviation outside CI'),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=13,
        title_fontsize=14
    )
    plt.suptitle(
        "95% Confidence Intervals and Deviations of " + 
        ("Parameter Differences (Δθ)" if plot_differences else "Parameter Estimates"),
        fontsize=20, y = 0.95
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])



def plot_mse_sampling_size(df, methods=['BFGS', 'stepwiseBFGS'], include_reduced=True, dodge_factor=0.05):
    import matplotlib.pyplot as plt
    import numpy as np

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

    # Build method-label mapping: (method, reduced_flag)
    plot_methods = []
    for method in methods:
        plot_methods.append((method, False))  # Normal
        if include_reduced:
            plot_methods.append((method, True))  # Reduced

    for ax, target in zip(axes, targets.keys()):
        for idx, (method, reduced) in enumerate(plot_methods):
            reduced_prefix = 'reduced_' if reduced else ''
            label_suffix = ' (reduced)' if reduced else ''

            mean_col = f"{target}_mse_mean_{reduced_prefix}{method}"
            std_col = f"{target}_mse_std_{reduced_prefix}{method}"

            if mean_col not in df.columns:
                continue  # skip missing columns

            # Apply horizontal dodge (log-space)
            x_base = df['n_samples'].values
            offset = dodge_factor * (idx - len(plot_methods)/2)
            x_dodged = x_base * (1 + offset)

            ax.errorbar(
                x_dodged,
                df[mean_col],
                yerr=df[std_col],
                fmt=markers[idx % len(markers)],
                capsize=5,
                color=colors[idx % len(colors)],
                label=f"{method}{label_suffix}"
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


def plot_mse_n_variants(df, methods=['BFGS', 'stepwiseBFGS'], dodge_width = 0.05, reduced_variants = False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)

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
        
            if reduced_variants:
                mean_col = f"{target}_mse_mean_reduced_{method}"
                std_col = f"{target}_mse_std_reduced_{method}"
            else:
                mean_col = f"{target}_mse_mean_{method}"
                std_col = f"{target}_mse_std_{method}"

            # Apply dodge in log-space
            log_x = np.log10(df['n_variants_requested'])
            dodge_offset = (idx - (len(methods)-1)/2) * dodge_width
            log_x_dodged = log_x + dodge_offset
            x_dodged = 10 ** log_x_dodged  # convert back to linear scale

            ax.errorbar(
                x_dodged,
                df[mean_col],
                yerr=df[std_col],
                fmt=markers[idx % len(markers)],
                capsize=5,
                color=colors[idx % len(colors)],
                label=method
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Δ {targets[target]} MSE vs. Number of Variants", size = 18)
        ax.set_xlabel("Number of variants", fontsize = 14)
        if reduced_variants:
            ax.set_ylabel("Mean Squared Error (variants > 100 counts)", fontsize = 14)
        else:
            ax.set_ylabel("Mean Squared Error", fontsize = 14)
        ax.grid(True)
        ax.legend(title="Estimation Method", fontsize=13, title_fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_mse_variant_appearance(df, methods=['BFGS', 'stepwiseBFGS'], x_col='new_var_rate', dodge_width=0.03, reduced_variants=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)

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
            if reduced_variants:
                mean_col = f"{target}_mse_mean_reduced_{method}"
                std_col = f"{target}_mse_std_reduced_{method}"
            else:
                mean_col = f"{target}_mse_mean_{method}"
                std_col = f"{target}_mse_std_{method}"

            # Apply dodge in log-space
            log_x = np.log10(df[x_col])
            dodge_offset = (idx - (len(methods)-1)/2) * dodge_width
            log_x_dodged = log_x + dodge_offset
            x_dodged = 10 ** log_x_dodged  # convert back to linear scale

            ax.errorbar(
                x_dodged,
                df[mean_col],
                yerr=df[std_col],
                fmt=markers[idx % len(markers)],
                capsize=5,
                color=colors[idx % len(colors)],
                label=method
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        if x_col == 'mean_entropy_mean':
            ax.set_title(f"Δ {targets[target]} MSE vs. Variant Diversity", fontsize=16)
            ax.set_xlabel("Mean Entropy of variant composition (freq.)", fontsize=14)
        else:
            ax.set_title(f"Δ {targets[target]} MSE vs. Variant Appearance Rate", fontsize=16)
            ax.set_xlabel("Variant appearance rate (per day)", fontsize=14)
        
        ax.set_ylabel("Mean Squared Error", fontsize=14)
        ax.grid(True)
        ax.legend(title="Estimation Method", fontsize=13, title_fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_mse_sampling_size_multi(dataframes, labels, methods=['BFGS', 'stepwiseBFGS'], dodge_factor=0.05):
    import matplotlib.pyplot as plt
    import numpy as np

    assert len(dataframes) == len(labels), "Number of dataframes and labels must match."

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    targets = {
        'gr': 'Growth Rate',
        'lif': 'Log. Initial Frequency'
    }

    markers = ['s', 'o', '^', 'D', 'v', '*', 'P', 'X']
    colors = [
        "#a6444f", "#57a8b8", "#80557e", "#b5d2f2",
        "#d991b4", "#397398", "#7394c2", "#7a7a7a"
    ]

    # Build plotting tasks: (df_idx, method, reduced_flag, label)
    plot_tasks = []

    # First add non-reduced methods of df_idx == 0 in order (BFGS first)
    df_idx = 0
    label = labels[df_idx]
    for method in methods:
        plot_tasks.append((df_idx, method, False, label))  # non-reduced first

    # Then add reduced methods for all dataframes (BFGS first, then stepwiseBFGS)
    for df_idx, label in enumerate(labels):
        for method in methods:
            plot_tasks.append((df_idx, method, True, label))  # reduced versions

    # Plotting loop
    for ax, target in zip(axes, targets.keys()):
        for idx, (df_idx, method, reduced, label_suffix) in enumerate(plot_tasks):
            df = dataframes[df_idx]

            reduced_prefix = 'reduced_' if reduced else ''
            mean_col = f"{target}_mse_mean_{reduced_prefix}{method}"
            std_col = f"{target}_mse_std_{reduced_prefix}{method}"

            if mean_col not in df.columns:
                continue  # skip missing columns

            # Apply horizontal dodge (log-space)
            x_base = df['n_samples'].values
            offset = dodge_factor * (idx - len(plot_tasks)/2)
            x_dodged = x_base * (1 + offset)

            # Label logic
            if reduced:
                label_text = f"{method} ({label_suffix})"
            else:
                label_text = f"{method}"

            ax.errorbar(
                x_dodged,
                df[mean_col],
                yerr=df[std_col],
                fmt=markers[idx % len(markers)],
                capsize=5,
                color=colors[idx % len(colors)],
                label=label_text
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Δ {targets[target]} MSE vs. Sampling Size", size = 18)
        ax.set_xlabel("Number of samples per day", size = 14)
        ax.set_ylabel("Mean Squared Error", size = 14)
        ax.grid(True)
        ax.legend(title="Estimation Method & Filter")

    plt.tight_layout()
    plt.show()




### PLOTS COVID DATA

def plot_count_distribution_over_time(counts_df):
    """
    Plot the distribution of counts over time.
    """

    # Count values per date and sort
    date_counts = counts_df['date'].value_counts().sort_index()
    ax = date_counts.plot(kind='bar', figsize=(12, 5))

    # Format x-axis ticks: keep only one tick per month
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    dates = pd.to_datetime(xticklabels, errors='coerce')

    # Replace x-axis labels with month-start labels only
    new_labels = []
    last_month = None
    for d in dates:
        if pd.isna(d):
            new_labels.append("")
        elif d.month != last_month:
            new_labels.append(d.strftime("%b %Y"))  # e.g., "Apr 2025"
            last_month = d.month
        else:
            new_labels.append("")

    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Counts per Date')
    plt.tight_layout()
    plt.show()


def plot_clade_pango_over_time(pango_clade_mapping_df, color_palette='range', plot_type='growth', show_clade_labels=True):

    assert plot_type in ['growth', 'lif'], "plot_type must be 'growth' or 'lif'"

    clades_df = pango_clade_mapping_df[['clade', 'clade_growth', 'clade_lif', 'clade_time']].drop_duplicates()

    if color_palette == 'range':    
        palette = sns.color_palette('husl', n_colors=len(clades_df))
        clade_colors = {clade: f'rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})' for clade, (r, g, b) in zip(clades_df['clade'], palette)}
    elif color_palette == 'jumps':
        palette = (px.colors.qualitative.Dark24 * (len(clades_df) // 24 + 1))[:len(clades_df)]
        clade_colors = {clade: color for clade, color in zip(clades_df['clade'], palette)}

    if plot_type == 'growth':
        clade_y = 'clade_growth'
        pango_y = 'pango_growth'
        y_title_clade = 'Clade Fitness'
        y_title_pango = 'Pango Fitness'
        hover_label = 'Fitness'
        main_title = "Clade and Pango Lineage Fitness Over Time"
    else:
        clade_y = 'clade_lif'
        pango_y = 'pango_lif'
        y_title_clade = 'Clade Log Init Freq'
        y_title_pango = 'Pango Log Init Freq'
        hover_label = 'Log Init Freq'
        main_title = "Clade and Pango Lineage Log Initial Frequencies Over Time"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        subplot_titles=(y_title_clade + " over Time", y_title_pango + " over Time"),
        vertical_spacing=0.15,
    )

    for idx, row in clades_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['clade_time']],
                y=[row[clade_y]],
                mode='markers+text' if show_clade_labels else 'markers',
                marker=dict(color=clade_colors[row['clade']], size=10),
                text=[row['clade']] if show_clade_labels else None,
                textposition='top center',
                textfont=dict(size=12),
                name=row['clade'],
                hovertemplate=f"Clade: {row['clade']}<br>Time: {row['clade_time']:.2f}<br>{hover_label}: {row[clade_y]:.2f}<extra></extra>",
                showlegend=not show_clade_labels
            ),
            row=1, col=1
        )

    for idx, row in pango_clade_mapping_df.iterrows():
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

    fig.update_layout(
        height=900,
        title_text=main_title,
        title_font=dict(size=24),
        xaxis_title="Mean Time",
        yaxis_title=y_title_clade,
        xaxis2_title="Mean Time",
        yaxis2_title=y_title_pango,
        legend_title_text='Clade',
        showlegend=not show_clade_labels,
        legend=dict(
            orientation='h',
            y=-0.1,
            x=0.5,
            xanchor='center',
            title_text=None
        ),
        font=dict(size=16),
        margin=dict(t=100, b=80)
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=20)

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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8), sharex=False)

    # Row 1: Mean fitness
    axes[0, 0].plot(time, pango_mean_fitness, color="#397398", linewidth=2.5)
    axes[0, 0].set_ylabel('Mean Fitness')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_title('Pango Mean Fitness')
    axes[0, 0].set_ylim(mf_min - mf_add, mf_max + mf_add)

    axes[0, 1].plot(time, clade_mean_fitness, color="#6c9a8b", linewidth=2.5)
    axes[0, 1].set_ylabel('Mean Fitness')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_title('Clade Mean Fitness')
    axes[0, 1].set_ylim(mf_min - mf_add, mf_max + mf_add)

    # Row 2: ΔFitness
    axes[1, 0].plot(delta_time, delta_pango, color="#397398", linewidth=2.5)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Δ Fitness')
    axes[1, 0].set_title('Δ Pango Mean Fitness')
    axes[1, 0].set_ylim(delta_min - delta_add, delta_max + delta_add)

    axes[1, 1].plot(delta_time, delta_clade, color="#6c9a8b", linewidth=2.5)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Δ Fitness')
    axes[1, 1].set_title('Δ Clade Mean Fitness')
    axes[1, 1].set_ylim(delta_min - delta_add, delta_max + delta_add)

    # Styling
    for ax in axes.flat:
        ax.grid(True)
        ax.set_title(ax.get_title(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_pango_lineages_of_single_clade(clade, data_pango, result_pango, pango_clade_mapping):
    var_clade = list(pango_clade_mapping[pango_clade_mapping.clade == clade].seqName)
    mask = np.isin(data_pango['variant_names'], var_clade)
    counts_clade = data_pango['counts'][:,mask]
    var_names_clade = data_pango['variant_names'][mask]

    plot_viral_composition_interactive(counts_clade,composition_estimate=result_pango['composition_estimate'][:,mask], var_names=var_names_clade, logscale = False, start_date="2020-01-01")
    #plot_viral_composition_dual(counts_clade, composition_estimate=result_pango['composition_estimate'][:,mask], var_names=var_names_clade, y_range = (1e-5, 2), show_legend=False)


### AMINO ACID SUBSTITUTION IMPACT ESTIMATION

def plot_true_vs_est(true_a, est_a):
    
    plt.figure(figsize=(6, 6))
    plt.scatter(true_a, est_a, alpha=0.8)
    plt.plot([-0.2, 0.2], [-0.2, 0.2], 'k--', label='Ideal')
    plt.xlabel("True a")
    plt.ylabel("Estimated a")
    plt.title("True vs Estimated Amino Acid Impacts")
    plt.grid(True)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_aa_impact_dist_cov(aa_impact, aa_impact_diag_C):
    """
    Plot the distribution of amino acid impacts with and without covariance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Plot 1: Diagonal covariance
    axes[0].hist(aa_impact_diag_C, bins=100, edgecolor='black')
    axes[0].set_xlabel("Amino Acid Substitution Impact")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram: Diagonal Covariance")
    axes[0].set_yscale('log')

    # Plot 2: Full covariance
    axes[1].hist(aa_impact, bins=100, edgecolor='black')
    axes[1].set_xlabel("Amino Acid Substitution Impact")
    axes[1].set_title("Histogram: Full Covariance")
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_aa_impac_distr_lambdas(lambda_values, aa_impact_C_dict):

    # Compute global x-limits and y-limits
    all_impacts = np.concatenate([imp['a_est'] for imp in aa_impact_C_dict.values()])    
    x_min, x_max = np.min(all_impacts), np.max(all_impacts)

    # Compute y-limits based on histogram counts
    hist_counts = []
    for impacts in aa_impact_C_dict.values():
        counts, _ = np.histogram(impacts['a_est'], bins=100)
        hist_counts.append(counts)

    y_max = np.max(hist_counts) + 100


    fig, axes = plt.subplots(1, len(lambda_values), figsize=(6 * len(lambda_values), 5), sharey=True)

    if len(lambda_values) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, lam in zip(axes, lambda_values):
        ax.hist(aa_impact_C_dict[lam]['a_est'], bins=100, edgecolor='black')
        ax.set_xlabel("Amino Acid Substitution Impact")
        ax.set_title(f"Histogram: lambda = {lam}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(1, y_max)  # log scale, so start at 1
        ax.set_yscale('log')

    axes[0].set_ylabel("Frequency")

    plt.tight_layout()
    plt.suptitle("Distribution of Amino Acid Substitution Impact for Different Lambda Values", fontsize=16, y=1.04)
    plt.show()


def plot_bootstrap_aa_impact(bootstrap_results_df, top_n, abundancy_threshold=10):
    """
    Plot the mean and confidence intervals of amino acid impacts.
    """
    bootstrap_results_df = bootstrap_results_df[bootstrap_results_df.n_events > abundancy_threshold]
    top_results_df = bootstrap_results_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.errorbar(top_results_df['aa_substitution'], top_results_df['mean'], yerr=[top_results_df['mean'] - top_results_df['ci_lower'], top_results_df['ci_upper'] - top_results_df['mean']], fmt='o', capsize=5)
    plt.xticks(rotation=90)
    plt.ylabel('Amino Acid Substitution Impact on Fitness')
    plt.title('Bootstrap Estimates of Amino Acid Impacts with 95% Confidence Intervals')
    plt.grid(True)
    plt.show()

def plot_bootstrap_aa_impact_violin(bootstrap_results_df, bootstrap_array, top_n, abundancy_threshold=10):

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
    
    bootstrap_results_df = bootstrap_results_df[bootstrap_results_df.n_events > abundancy_threshold]
    top_results_df = bootstrap_results_df.head(top_n)
    top_indices = top_results_df.index
    top_bootstrap_array = bootstrap_array[:, top_indices]

    # Prepare long-form DataFrame for Seaborn
    aa_labels = top_results_df['aa_substitution'].values
    plot_df = pd.DataFrame(top_bootstrap_array, columns=aa_labels)
    plot_df = plot_df.melt(var_name='aa_substitution', value_name='impact_estimate')

    plt.figure(figsize=(max(8, top_n), 6))
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder = 0)
    for i, aa in enumerate(aa_labels):
        color = colors[i % len(colors)]
        subset = plot_df[plot_df['aa_substitution'] == aa]
        sns.violinplot(x='aa_substitution', y='impact_estimate', data=subset, inner=None, color=color)

    sns.stripplot(x='aa_substitution', y='impact_estimate', data=plot_df, color='black', alpha=0.6, size=4)

    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Amino Acid Substitution', fontsize=16)
    plt.ylabel('Fitness Impact Estimate', fontsize=16)
    plt.title(f'Top {top_n} Amino Acid Substitution Impact Estimates (> {abundancy_threshold} events)', fontsize= 18)
    plt.tight_layout()
    plt.show()

def plot_aa_impact_time(aa_impact_time_df, labels, top_n = 20, abundancy_threshold = 10):

    df = aa_impact_time_df.copy()
    df = df[df.n_events > abundancy_threshold].head(top_n)

    # Plot settings
    impact_cols = [f'impact_clades{i+1}' for i in range(4)]
    events_cols = [f'events_clades{i+1}' for i in range(4)]

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

    x = np.arange(len(df['aa_substitution']))  # x-axis positions
    width = 0.2  # width of each bar

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 6))

    for i in range(4):
        impacts = df[impact_cols[i]]
        events = df[events_cols[i]]
        ax.bar(x + i*width, impacts, width, label=labels[i], color=colors[i])

        # Annotate bars with event counts
        for xi, yi, ev in zip(x + i*width, impacts, events):
            ax.text(xi, yi + 0.0003, f'{int(ev)}', ha='center',
                    va='bottom', fontsize=10)

    # X-axis settings
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(df['aa_substitution'], rotation=45, ha='right')


    # Labels and legend
    ax.set_ylabel('Estimated Impact', fontsize=14)
    ax.set_xlabel('Amino Acid Substitution', fontsize=14)

    ax.set_title('Impact of Amino Acid Substitutions per Phase of the Pandemic', fontsize=16)

    # Add custom legend entry for n_events annotation explanation
    handles = [Patch(color=colors[i], label=labels[i]) for i in range(4)]
    handles.append(Patch(edgecolor='black', facecolor='none', label='1: Number of amino acid substitution events'))

    ax.legend(handles=handles, title='Clade Groups', loc='upper right', fontsize = 12, title_fontsize=13)

    plt.tight_layout()
    plt.show()

def plot_loss_different_lambda(objective_vals, n_zero_aa_impact, lambda_values, n_aa_subst, add_inf_lambda=False, inf_lambda=1e6):
    """
    Plot the objective values and number of near-zero AA impacts for different lambda values,
    with a second y-axis for the sparsity level.
    """

    # Add infinite lambda point if needed
    if add_inf_lambda:
        lambda_plot = lambda_values + [inf_lambda]
        objective_plot = objective_vals + [640.6815785668837]  # max_lambda_loss
        n_zero_plot = n_zero_aa_impact + [n_aa_subst]  # assume all zero at λ=∞
    else:
        lambda_plot = lambda_values
        objective_plot = objective_vals
        n_zero_plot = n_zero_aa_impact

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot objective function
    ax1.plot(lambda_plot, objective_plot, marker='o', label='Objective value', color="#397398")
    ax1.set_xlabel('Lambda (log scale)', fontsize = 13)
    ax1.set_ylabel('Final value of objective function', color="#397398", fontsize = 13)
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor="#397398")
    ax1.grid(True)

    # Plot annotation for λ = ∞
    if add_inf_lambda:
        ax1.annotate("λ = ∞", 
                     xy=(inf_lambda, objective_plot[-1]), 
                     xytext=(inf_lambda, objective_plot[-1] * 1.1),
                     arrowprops=dict(arrowstyle='->'), fontsize = 13)

    # Add second y-axis for number of near-zero impacts
    ax2 = ax1.twinx()
    ax2.plot(lambda_plot, n_zero_plot, marker='s', linestyle='--', color="#a6444f", label='Zero impact aa substitutions')
    ax2.set_ylabel('Number of zero impact aa substitutions', color="#a6444f", fontsize = 13)
    ax2.tick_params(axis='y', labelcolor="#a6444f")

    # Add legends
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)
    plt.title('Effect of L1 Regularization on Loss and Sparsity', fontsize = 14)
    plt.tight_layout()
    plt.show()






### HELPERS

def plot_heatmap(matrix, title, x_label, y_label, cmap='custom_cmap', log_coloring=False):

    if cmap == 'custom_cmap':
        base_colors = [
            "#80557e",  # purple
            "#397398",  # dark blue
            "#57a8b8",  # teal
            "#7394c2",  # mid blue
            "#b5d2f2",  # light blue
        ][::-1]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", base_colors, N=256)
    plt.figure(figsize=(8, 6.5))  

    mask = (matrix == 0)

    if (log_coloring):
        sns.heatmap(
            matrix,
            cmap=cmap,
            norm=LogNorm(1, vmax=np.nanmax(matrix)),
            mask=mask
        )
    else:
        sns.heatmap(matrix, cmap=cmap, mask=mask)

    plt.title(title, pad=20, size = 20, loc='center')  # Increase pad for more space above heatmap
    plt.xlabel(x_label, size = 16)
    plt.ylabel(y_label, size = 16)

    #plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()