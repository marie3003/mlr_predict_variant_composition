import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

def plot_viral_composition(counts, freq, n_samples, composition_estimate = None, y_range = (1e-5, 2), logscale = True):

    plt.figure(figsize=(freq.shape[0] // 50, 6))
    
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

    for i in range(counts.shape[1]):

        plt.scatter(np.arange(counts.shape[0]), counts[:, i] / n_samples, s= 10, alpha = 0.2, label = f'Variant {i+1}', color = base_colors[i % 10])
        plt.plot(np.arange(freq.shape[0]), freq[:, i], color = base_colors[i % 10])
        if composition_estimate is not None:
            plt.plot(np.arange(composition_estimate.shape[0]), composition_estimate[:, i], color = brighter_colors[i % 10])

    plt.xlabel("Days")
    plt.ylabel("Abundancy [%]")
    if(logscale):
        plt.yscale('log')
        plt.ylim(y_range)
    plt.title("Disease Variants in Population")

    if(freq.shape[1] > 10):
        plt.legend(ncol = freq.shape[1] // 10, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    else:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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


def plot_confidence_intervals_deviation(param_df):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
    gr_df = param_df[param_df.parameter_type == 'growth_rate']
    lif_df = param_df[param_df.parameter_type == 'log_initial_freq']

    # Plot for s_i (top left)
    max_gr = np.max(gr_df.parameter_estimate)
    growth_rate_range = (np.min(gr_df.parameter_estimate) - 0.5 * max_gr, max_gr + 0.5 * max_gr)

    for i in range(len(gr_df)):
        color = "green" if gr_df.ci_lower.iloc[i] <= gr_df.true_parameter.iloc[i] <= gr_df.ci_upper.iloc[i] else "red"
        axes[0, 0].errorbar(i, gr_df.parameter_estimate.iloc[i], yerr=gr_df.standard_error.iloc[i], fmt='o', capsize=5, color='black')
        axes[0, 0].plot(i, gr_df.true_parameter.iloc[i], 'x', color=color, markersize=10, label='True' if i == 0 else "")

    axes[0, 0].set_title("Growth rate (s_i)")
    axes[0, 0].set_ylim(growth_rate_range)
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Estimated value")
    axes[0, 0].grid(True)

    # Plot for o_i (top right)
    min_lif = np.min(lif_df.parameter_estimate)
    freq_range = (min_lif + 0.5 * min_lif, np.max(lif_df.parameter_estimate) - 0.5 * min_lif)

    for i in range(len(lif_df)):
        color = "green" if lif_df.ci_lower.iloc[i] <= lif_df.true_parameter.iloc[i] <= lif_df.ci_upper.iloc[i] else "red"
        axes[0, 1].errorbar(i, lif_df.parameter_estimate.iloc[i], yerr=lif_df.standard_error.iloc[i], fmt='o', capsize=5, color='black')
        axes[0, 1].plot(i, lif_df.true_parameter.iloc[i], 'x', color=color, markersize=10, label='True' if i == 0 else "")

    axes[0, 1].set_title("Log. initial frequencies (o_i)")
    axes[0, 1].set_xlabel("Index")
    axes[0, 1].set_ylim(freq_range)
    axes[0, 1].set_ylabel("Estimated value")
    axes[0, 1].grid(True)

    # Deviation subplot for s_i (bottom left)
    for i in range(len(gr_df)):
        deviation = gr_df.deviation.iloc[i]
        within_ci = gr_df.ci_lower.iloc[i] <= gr_df.true_parameter.iloc[i] <= gr_df.ci_upper.iloc[i]
        point_color = "green" if within_ci else "red"

        # Plot black error bar (lower zorder)
        axes[1, 0].errorbar(i, 0, yerr=gr_df.standard_error.iloc[i], capsize=5, color='black', zorder=1)
        # Plot colored point *after*, with higher zorder so it appears on top
        axes[1, 0].plot(i, deviation, 'o', color=point_color, markersize=6, zorder=2)

    axes[1, 0].axhline(0, linestyle='--', color='gray')
    axes[1, 0].set_title("Deviation: Estimated - True (s_i)")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Deviation")
    axes[1, 0].grid(True)

    # same for o_i (bottom right)
    for i in range(len(lif_df)):
        deviation = lif_df.deviation.iloc[i]
        within_ci = lif_df.ci_lower.iloc[i] <= lif_df.true_parameter.iloc[i] <= lif_df.ci_upper.iloc[i]
        point_color = "green" if within_ci else "red"

        axes[1, 1].errorbar(i, 0, yerr=lif_df.standard_error.iloc[i], capsize=5, color='black', zorder=1)
        axes[1, 1].plot(i, deviation, 'o', color=point_color, markersize=6, zorder=2)

    axes[1, 1].axhline(0, linestyle='--', color='gray')
    axes[1, 1].set_title("Deviation: Estimated - True (o_i)")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Deviation")
    axes[1, 1].grid(True)



    # Legend for markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', label='Estimate inside CI'),
        Line2D([0], [0], marker='x', color='green', linestyle='None', label='True parameter inside CI', markersize=10),
        Line2D([0], [0], marker='x', color='red', linestyle='None', label='True parameter outside CI', markersize=10),
        Line2D([0], [0], marker='o', color='green', linestyle='None', label='Deviation inside CI'),
        Line2D([0], [0], marker='o', color='red', linestyle='None', label='Deviation outside CI'),

    ]
    fig.legend(handles=legend_elements, loc="upper right", ncol=4)
    plt.suptitle("95% Confidence Intervals and Deviations of parameter estimates", fontsize=16)


def plot_heatmap(matrix, title, x_label, y_label):

    mask = (matrix == 0)

    sns.heatmap(matrix, cmap='viridis', mask = mask)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()