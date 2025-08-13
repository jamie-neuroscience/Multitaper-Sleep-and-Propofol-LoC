import numpy as np
from scipy.stats import ttest_rel, ttest_ind, mannwhitneyu, wilcoxon
import glob
import os
import pandas as pd
from scipy.stats import shapiro
from mne.stats import permutation_cluster_1samp_test
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import levene



# -----------------------------
# Paths and config
# -----------------------------
frequencies = np.load(os.path.join("Propofol_Spectrograms_group", "frequencies.npy"))
output_folder = 'Results'

# Define bands
freq_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Low Beta': (13, 20),
    'High Beta': (20, 30),
    'Gamma': (30, 49.5)
}

# Epoch windows: adjust as needed
# e.g., you have 240 epochs before and after LoC, total of 480 epochs
# So pre-LoC = first half, post-LoC = second half
n_epochs_total = 240
pre_indices = np.arange(0, n_epochs_total // 2)
post_indices = np.arange(n_epochs_total // 2, n_epochs_total)



print(f"Pre window: {pre_indices[0]}:{pre_indices[-1]}, Post window: {post_indices[0]}:{post_indices[-1]}")


# -----------------------------
# Load files into groups
# -----------------------------


sleep_relative_pattern = os.path.join("Sleep_Spectrograms_db", "*_db_spectrogram.npy")
propofol_relative_pattern = os.path.join("Propofol_Spectrograms_db", "*_db_spectrogram.npy")
times = os.path.join("Propofol_Spectrograms_db", "correct_times.npy")


# Load function
def load_group(pattern):
    files = glob.glob(pattern)
    print(f"Found {len(files)} files for pattern '{pattern}'")
    if not files:
        raise ValueError(f"No files matched pattern '{pattern}'!")
    arrays = [np.load(f) for f in files]
    return np.stack(arrays, axis=0)

# Format Band Power
def format_results_table(results, metric_key="Mean", alpha=0.05):
    """
    Format the results table for either absolute or relative power, including test statistics.

    Args:
        results (list): List of band dictionaries.
        metric_key (str): The key suffix, e.g. 'Mean' or 'Rel Mean'.
        alpha (float): Significance threshold.
    """
    rows = []
    for r in results:
        band = r['Band']

        # Compose keys dynamically
        prop_pre = r[f'Propofol Pre {metric_key}']
        prop_post = r[f'Propofol Post {metric_key}']
        prop_pre_std = r.get(f'Propofol Pre Std', None)
        prop_post_std = r.get(f'Propofol Post Std', None)

        prop_change = prop_post - prop_pre
        prop_direction = "Increase" if prop_change > 0 else "Decrease"
        prop_p = r['Propofol Paired p']
        prop_sig = "YES" if prop_p < alpha else "NO"
        prop_test = r.get("Propofol Test", "t-test")
        prop_stat = r.get("Propofol Paired stat", None)

        sleep_pre = r[f'Sleep Pre {metric_key}']
        sleep_post = r[f'Sleep Post {metric_key}']
        sleep_pre_std = r.get(f'Sleep Pre Std', None)
        sleep_post_std = r.get(f'Sleep Post Std', None)

        sleep_change = sleep_post - sleep_pre
        sleep_direction = "Increase" if sleep_change > 0 else "Decrease"
        sleep_p = r['Sleep Paired p']
        sleep_sig = "YES" if sleep_p < alpha else "NO"
        sleep_test = r.get("Sleep Test", "t-test")
        sleep_stat = r.get("Sleep Paired stat", None)

        between_p = r['Between Delta p']
        between_sig = "YES" if between_p < alpha else "NO"
        between_test = r.get("Between Test", "t-test")
        between_stat = r.get("Between Delta stat", None)

        rows.append({
            "Band": band,
            f"Prop Pre ({metric_key})": round(prop_pre, 3),
            f"Prop Pre (Std)": round(prop_pre_std, 3) if prop_pre_std is not None else None,
            f"Prop Post ({metric_key})": round(prop_post, 3),
            f"Prop Post (Std)": round(prop_post_std, 3) if prop_post_std is not None else None,
            "Prop Δ": round(prop_change, 3),
            "Prop Dir": prop_direction,
            "Prop stat": round(prop_stat, 3) if prop_stat is not None else None,
            "Prop p": round(prop_p, 5),
            "Prop Sig": prop_sig,
            "Prop Test": prop_test,

            f"Sleep Pre ({metric_key})": round(sleep_pre, 3),
            f"Sleep Pre (Std)": round(sleep_pre_std, 3) if sleep_pre_std is not None else None,
            f"Sleep Post ({metric_key})": round(sleep_post, 3),
            f"Sleep Post (Std)": round(sleep_post_std, 3) if sleep_post_std is not None else None,
            "Sleep Δ": round(sleep_change, 3),
            "Sleep Dir": sleep_direction,
            "Sleep stat": round(sleep_stat, 3) if sleep_stat is not None else None,
            "Sleep p": round(sleep_p, 5),
            "Sleep Sig": sleep_sig,
            "Sleep Test": sleep_test,

            "Between stat": round(between_stat, 3) if between_stat is not None else None,
            "Between p": round(between_p, 5),
            "Between Sig": between_sig,
            "Between Test": between_test
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

# Define band indices
def band_indices(freqs, band):
    return np.where((freqs >= band[0]) & (freqs < band[1]))[0]


# Load each group
sleep_group = load_group(sleep_relative_pattern)
propofol_group = load_group(propofol_relative_pattern)

print(f"Sleep group shape: {sleep_group.shape}")
print(f"Propofol group shape: {propofol_group.shape}")

epoch_length_sec = 2  # change to match your data
n_epochs = sleep_group.shape[-1]
mid_epoch = n_epochs // 2
time = (np.arange(n_epochs) - mid_epoch) * epoch_length_sec  # in seconds

# -----------------------------
# Compute per-subject band power
# -----------------------------
def mean_band_power(sleep_group, propofol_group):
    """

    Calculates the mean band power pre and post LoC, runs statistics, prints and saves them
    Arguments:
    :param sleep_group: the data for t
    :param propofol_group:
    :return:
    """

    print(sleep_group.shape)
    print(propofol_group.shape)

    mean_band_power_results = []
    normality_results = []

    # Mean Band Power before and after LoC
    for band_name, band_range in freq_bands.items():
        idx = band_indices(frequencies, band_range)
        print(f"{band_name}: {len(idx)} bins")

        # Mean over band freqs and epochs
        prop_pre = propofol_group[:, idx][:, :, pre_indices].mean(axis=(1, 2))
        prop_post = propofol_group[:, idx][:, :, post_indices].mean(axis=(1, 2))
        prop_delta = prop_post - prop_pre

        sleep_pre = sleep_group[:, idx][:, :, pre_indices].mean(axis=(1, 2))
        sleep_post = sleep_group[:, idx][:, :, post_indices].mean(axis=(1, 2))
        sleep_delta = sleep_post - sleep_pre

        # ------------------------
        # Statistics
        # ------------------------

        # Check normality
        _, shapiro_sleep_p = shapiro(sleep_delta)
        print(f'Shapiro-Wilk test p = {shapiro_sleep_p:.4f}')
        if shapiro_sleep_p < 0.05:
            shapiro_sleep_p_sig = "NOT Normally Distributed"
        else:
            shapiro_sleep_p_sig = "Normally Distributed"

        # Example: check normality of Propofol change scores
        _, shapiro_prop_p = shapiro(prop_delta)
        print(f'Shapiro-Wilk test p = {shapiro_prop_p:.4f}')
        if shapiro_prop_p < 0.05:
            shapiro_prop_p_sig = "NOT Normally Distributed"
        else:
            shapiro_prop_p_sig = "Normally Distributed"

        _, levene_p = levene(prop_delta, sleep_delta)
        print(f"Levene’s test p = {levene_p:.4f}")
        if levene_p < 0.05:
            levene_p_sig = "NOT Equal Variances"
        else:
            levene_p_sig = "Equal Variances"


        if shapiro_sleep_p < 0.05:
            sleep_result = wilcoxon(sleep_post, sleep_pre)
            sleep_stat = sleep_result.statistic
            sleep_p = sleep_result.pvalue
            sleep_p_test = "Wilcoxon"
        else:
            sleep_stat, sleep_p = ttest_rel(sleep_post, sleep_pre)
            sleep_p_test = "Paired t test"

        if shapiro_prop_p < 0.05:
            prop_result = wilcoxon(prop_post, prop_pre)
            prop_stat = prop_result.statistic
            prop_p = prop_result.pvalue
            prop_p_test = "Wilcoxon"
        else:
            prop_stat, prop_p = ttest_rel(prop_post, prop_pre)
            prop_p_test = "Paired t test"

        if levene_p < 0.05 or shapiro_sleep_p < 0.05 or shapiro_prop_p < 0.05:
            between_result = mannwhitneyu(prop_delta, sleep_delta, alternative='two-sided')
            between_stat = between_result.statistic
            between_p = between_result.pvalue
            between_p_test = "Mannwhitneyu"
        else:
            between_stat, between_p = ttest_ind(prop_delta, sleep_delta)
            between_p_test = "Independent t test"

        mean_band_power_results.append({
            "Band": band_name,
            "Propofol Pre Mean": np.mean(prop_pre),
            "Propofol Pre Std": np.std(prop_pre, ddof=1),  # <-- Added
            "Propofol Post Mean": np.mean(prop_post),
            "Propofol Post Std": np.std(prop_post, ddof=1),  # <-- Added
            "Sleep Pre Mean": np.mean(sleep_pre),
            "Sleep Pre Std": np.std(sleep_pre, ddof=1),  # <-- Added
            "Sleep Post Mean": np.mean(sleep_post),
            "Sleep Post Std": np.std(sleep_post, ddof=1),  # <-- Added
            "Propofol Paired stat": prop_stat,
            "Propofol Paired p": prop_p,
            "Propofol Test": prop_p_test,
            "Sleep Paired stat": sleep_stat,
            "Sleep Paired p": sleep_p,
            "Sleep Test": sleep_p_test,
            "Between Delta stat": between_stat,
            "Between Delta p": between_p,
            "Between Test": between_p_test,
        })




        normality_results.append({
            "Band": band_name,
            "Shapiro Sleep p": shapiro_sleep_p,
            "Shapiro Sleep Sig": shapiro_sleep_p_sig,
            "Shapiro Prop p": shapiro_prop_p,
            "Shapiro Prop Sig": shapiro_prop_p_sig,
            "Levene p": levene_p,
            "Levene Sig": levene_p_sig,

        })

    """print("\n---- Results ----")
    for r in mean_band_power_results:
        print(r)"""




    output_folder = 'Results'


    # Create the folder
    os.makedirs(output_folder, exist_ok=True)

    mean_band_power = format_results_table(mean_band_power_results, metric_key="Mean")
    print(mean_band_power)

    normality = pd.DataFrame(normality_results)
    pd.options.display.max_columns = 15
    print(normality)

    # Now save the CSV in that folder
    mean_band_power.to_csv(os.path.join(output_folder, 'Mean_Band_Power_Stats_Pre_Post_LoC_with_Std_Dev.csv'), index=False)
    normality.to_csv(os.path.join(output_folder, 'Normality_Band_Power_Stats_new.csv'), index=False)

    return mean_band_power_results
mean_band_power_results = mean_band_power(sleep_group, propofol_group)

import matplotlib.patches as mpatches

def pval_to_stars(p):
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""
def plot_mean_band_power_bar_with_significance(mean_band_power_results, alpha=0.05):
    bands = [r["Band"] for r in mean_band_power_results]

    sleep_pre_means = [r["Sleep Pre Mean"] for r in mean_band_power_results]
    sleep_pre_stds = [r["Sleep Pre Std"] for r in mean_band_power_results]
    sleep_post_means = [r["Sleep Post Mean"] for r in mean_band_power_results]
    sleep_post_stds = [r["Sleep Post Std"] for r in mean_band_power_results]

    prop_pre_means = [r["Propofol Pre Mean"] for r in mean_band_power_results]
    prop_pre_stds = [r["Propofol Pre Std"] for r in mean_band_power_results]
    prop_post_means = [r["Propofol Post Mean"] for r in mean_band_power_results]
    prop_post_stds = [r["Propofol Post Std"] for r in mean_band_power_results]

    sleep_ps = [r["Sleep Paired p"] for r in mean_band_power_results]
    prop_ps = [r["Propofol Paired p"] for r in mean_band_power_results]

    x = np.arange(len(bands))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create hatch patterns for striped bars
    hatch_pattern = '////'

    # Sleep Pre - solid blue
    rects1 = ax.bar(x - 1.5*width, sleep_pre_means, width, yerr=sleep_pre_stds,
                    label='Sleep Pre LoC', color='blue', alpha = 0.5, edgecolor='black', capsize=5)

    # Sleep Post - striped blue
    rects2 = ax.bar(x - 0.5*width, sleep_post_means, width, yerr=sleep_post_stds,
                    label='Sleep Post LoC', color='none', edgecolor='blue', hatch=hatch_pattern, capsize=5)

    # Propofol Pre - solid red
    rects3 = ax.bar(x + 0.5*width, prop_pre_means, width, yerr=prop_pre_stds,
                    label='Propofol Pre LoC', color='red', alpha = 0.5, edgecolor='black', capsize=5)

    # Propofol Post - striped red
    rects4 = ax.bar(x + 1.5*width, prop_post_means, width, yerr=prop_post_stds,
                    label='Propofol Post LoC', color='none', edgecolor='red', hatch=hatch_pattern, capsize=5)

    ax.set_xlabel('Frequency Bands')
    ax.set_ylabel('Mean Power (dB)')
    ax.set_title('Mean Band Power Pre and Post Loss of Consciousness')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Custom legend for hatch pattern
    solid_patch_blue = mpatches.Patch(facecolor='blue', edgecolor='black', alpha = 0.5, label='Sleep Pre LoC')
    striped_patch_blue = mpatches.Patch(facecolor='white', edgecolor='blue', alpha = 0.5, hatch=hatch_pattern, label='Sleep Post LoC')
    solid_patch_red = mpatches.Patch(facecolor='red', edgecolor='black', alpha = 0.5, label='Propofol Pre LoC')
    striped_patch_red = mpatches.Patch(facecolor='white', edgecolor='red', alpha = 0.5, hatch=hatch_pattern, label='Propofol Post LoC')



    ax.legend(handles=[solid_patch_blue, striped_patch_blue, solid_patch_red, striped_patch_red])

    # --- Add significance bars higher above the bars to avoid overlap ---
    # --- Improved significance bar height function ---
    def get_significance_y(pre_means, pre_stds, post_means, post_stds, i):
        # Get the tallest point (bar + error) of both bars
        y_pre = pre_means[i] + (pre_stds[i] if pre_means[i] >= 0 else -pre_stds[i])
        y_post = post_means[i] + (post_stds[i] if post_means[i] >= 0 else -post_stds[i])
        max_y = max(y_pre, y_post)
        base_y = max_y if max_y > 0 else 0  # If both are negative, base from zero

        return base_y + 0.5  # Push it far enough above bars

    cap_height = 0.1  # Taller caps
    text_offset = 0.5  # Move asterisk higher above the line
    bar_offset = 0.02  # Push the whole bar up a little more for clarity

    # --- Sleep significance markers ---
    for i, p in enumerate(sleep_ps):
        stars = pval_to_stars(p)
        if stars:
            x1 = x[i] - 1.5 * width
            x2 = x[i] - 0.5 * width
            y = get_significance_y(sleep_pre_means, sleep_pre_stds, sleep_post_means, sleep_post_stds, i)
            y += bar_offset  # Lift the bar

            # Horizontal line
            ax.plot([x1, x2], [y, y], color='black', linewidth=1.4)

            # Vertical caps
            ax.plot([x1, x1], [y, y - cap_height], color='black', linewidth=1.4)
            ax.plot([x2, x2], [y, y - cap_height], color='black', linewidth=1.4)

            # Stars
            ax.text((x1 + x2) / 2, y - text_offset, stars, ha='center', va='bottom', fontsize=18)

    # --- Propofol significance markers ---
    for i, p in enumerate(prop_ps):
        stars = pval_to_stars(p)
        if stars:
            x1 = x[i] + 0.5 * width
            x2 = x[i] + 1.5 * width
            y = get_significance_y(prop_pre_means, prop_pre_stds, prop_post_means, prop_post_stds, i)
            y += bar_offset  # Lift the bar

            ax.plot([x1, x2], [y, y], color='black', linewidth=1.4)
            ax.plot([x1, x1], [y, y - cap_height], color='black', linewidth=1.4)
            ax.plot([x2, x2], [y, y - cap_height], color='black', linewidth=1.4)

            ax.text((x1 + x2) / 2, y - text_offset, stars, ha='center', va='bottom', fontsize=18)

    plt.tight_layout()
    plt.savefig("mean_band_power_pre_post_barplot_new_new_new.png", dpi=300)
    plt.show()
plot_mean_band_power_bar_with_significance(mean_band_power_results)


# -----------------------------
# Compute cluster based permutation tests
# -----------------------------

"""def cluster_permutation_test(
        data,
        frequencies,
        n_permutations=1000,
        alpha=0.05,
        threshold=None,
        tail=0,
        plot=True,
        title='Cluster-based permutation test'):
    """
  #  Run cluster-based permutation test for data vs baseline (zero).
"""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import t
    from mne.stats import permutation_cluster_1samp_test

    X = data
    n_subjects = X.shape[0]

    # Reshape for MNE (n_samples, n_tests)
    X_reshaped = X.reshape(n_subjects, -1)

    # Use default threshold or compute from t-distribution
    if threshold is None:
        threshold = t.ppf(1 - (alpha / 2), df=n_subjects - 1)  # two-tailed

    # Run the test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X_reshaped,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=tail,
        out_type='mask',
        verbose=True
    )

    # Reshape T_obs back to (freqs, times)
    T_obs = T_obs.reshape(X.shape[1:])

    # Create significant cluster mask
    cluster_p_map = np.zeros_like(T_obs, dtype=bool)
    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] < alpha:
            cluster_p_map[c] = True

    if plot:
        plt.figure(figsize=(10, 5))
        extent = [0, X.shape[-1], frequencies[0], frequencies[-1]]

        # Option 1: Solid binary mask
        plt.imshow(cluster_p_map, aspect='auto', origin='lower', extent=extent,
                   cmap='Greys', alpha=1.0)
        plt.title(title + f'\nSignificant Clusters (p<{alpha})')
        plt.xlabel('Epoch')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Significant (1=True, 0=False)')
        plt.tight_layout()
        plt.show()

    return T_obs, clusters, cluster_p_values, cluster_p_map



def plot_stats_and_clusters(T_obs, cluster_p_map, frequencies, n_epochs, alpha=0.05, title='Cluster-based permutation test'):
    extent = [0, n_epochs, frequencies[0], frequencies[-1]]

    plt.figure(figsize=(14, 5))

    # Plot observed statistics (e.g. t-values)
    plt.subplot(1, 2, 1)
    vmax = np.max(np.abs(T_obs))  # symmetric color scale
    im1 = plt.imshow(T_obs, aspect='auto', origin='lower', extent=extent, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax)
    plt.colorbar(im1, label='Observed test statistic (e.g., t-value)')
    plt.xlabel('Epoch')
    plt.ylabel('Frequency (Hz)')
    plt.title(title + '\nObserved Statistic Map')

    # Plot significant cluster mask
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(cluster_p_map, aspect='auto', origin='lower', extent=extent,
                     cmap='Greys', alpha=1.0)
    plt.colorbar(im2, label='Significant (1=True, 0=False)')
    plt.xlabel('Epoch')
    plt.ylabel('Frequency (Hz)')
    plt.title(title + f'\nSignificant Clusters (p<{alpha})')

    plt.tight_layout()
    plt.show()

# Testing sleep against baseline
T_obs, clusters, cluster_p_values, cluster_p_map = cluster_permutation_test(
    sleep_group,
    frequencies,
    n_permutations=5000,
    alpha=0.05,
    plot=True,
    title='Sleep vs Baseline'
)
print(f'Number of clusters: {len(clusters)}')
print(f'Cluster p-values: {cluster_p_values}')
print(f'Shape of cluster mask: {cluster_p_map.shape}')
print(f'Number of significant clusters: {np.sum(cluster_p_values < 0.05)}')
print(f'Sum of cluster_p_map: {np.sum(cluster_p_map)}')

plot_stats_and_clusters(T_obs, cluster_p_map, frequencies, n_epochs=240 )

T_obs, clusters, cluster_p_values, cluster_p_map = cluster_permutation_test(
    propofol_group,
    frequencies,
    n_permutations=5000,
    alpha=0.05,
    plot=True,
    title='Propofol vs Baseline'
)
print(f'Number of clusters: {len(clusters)}')
print(f'Cluster p-values: {cluster_p_values}')
print(f'Shape of cluster mask: {cluster_p_map.shape}')
print(f'Number of significant clusters: {np.sum(cluster_p_values < 0.05)}')
print(f'Sum of cluster_p_map: {np.sum(cluster_p_map)}')

plot_stats_and_clusters(T_obs, cluster_p_map, frequencies, n_epochs=240)

diff_group = sleep_group - propofol_group
T_obs, clusters, cluster_p_values, cluster_p_map = cluster_permutation_test(
    diff_group,
    frequencies,
    n_permutations=5000,
    alpha=0.05,
    plot=True,
    title='Sleep vs Propofol Difference'
)
print(f'Number of clusters: {len(clusters)}')
print(f'Cluster p-values: {cluster_p_values}')
print(f'Shape of cluster mask: {cluster_p_map.shape}')
print(f'Number of significant clusters: {np.sum(cluster_p_values < 0.05)}')
print(f'Sum of cluster_p_map: {np.sum(cluster_p_map)}')

plot_stats_and_clusters(T_obs, cluster_p_map, frequencies, n_epochs=240)"""


# --------
# SPECTRAL SLOPE
# -------

def compute_spectral_slope_full_window(group_spectrograms, freqs, f_range=(0.5, 49.5)):
    """
    Compute spectral slope across the entire time window for each subject.

    Parameters:
    - group_spectrograms: ndarray, shape (n_subjects, n_freqs, n_times)
    - freqs: ndarray, frequency vector corresponding to spectrogram freqs
    - f_range: tuple, frequency range to use for slope calculation (default 2-40 Hz)

    Returns:
    - slopes: ndarray of shape (n_subjects,), spectral slope per subject
    """
    n_subjects = group_spectrograms.shape[0]
    slopes = []

    # Prepare frequency mask and log frequencies
    freq_mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    log_freqs = np.log10(freqs[freq_mask])

    for subj in range(n_subjects):
        # Average power across all time points for this subject
        power_spectrum = np.mean(group_spectrograms[subj][:, :], axis=1)

        # Select freq range, power is already log-scaled or not?
        # Assuming power is linear, take log10 power before fitting slope
        log_power = np.log10(power_spectrum[freq_mask])

        # Linear fit to log-log power spectrum (spectral slope)
        slope, intercept = np.polyfit(log_freqs, log_power, 1)
        slopes.append(slope)

    return np.array(slopes)


def compare_spectral_slopes_between_groups(slopes_group1, slopes_group2, group1_name='Group1', group2_name='Group2'):
    """
    Perform Mann-Whitney U test comparing spectral slopes between two independent groups.

    Parameters:
    - slopes_group1: ndarray, spectral slopes for group 1
    - slopes_group2: ndarray, spectral slopes for group 2
    - group1_name, group2_name: str, names for groups for labeling

    Returns:
    - results_df: pandas DataFrame with group means, stds, test statistic, p-value, and test name
    """
    stat, p_val = mannwhitneyu(slopes_group1, slopes_group2, alternative='two-sided')

    results = {
        'Group': [group1_name, group2_name],
        'Mean Slope': [np.mean(slopes_group1), np.mean(slopes_group2)],
        'Std Slope': [np.std(slopes_group1, ddof=1), np.std(slopes_group2, ddof=1)],
    }

    stats_summary = {
        'Comparison': [f'{group1_name} vs {group2_name} Spectral Slope'],
        'Test': ['Mann-Whitney U'],
        'Statistic': [stat],
        'p-value': [p_val]
    }

    results_df = pd.DataFrame(results)
    stats_df = pd.DataFrame(stats_summary)

    return results_df, stats_df

def compute_subject_slopes(group, freqs, pre_indices, post_indices, f_range=(2, 40)):
    n_subjects = group.shape[0]
    slopes_pre = []
    slopes_post = []

    # Only log frequencies — power already in log scale
    log_freqs = np.log10(freqs)
    freq_mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    log_f = log_freqs[freq_mask]

    for subj in range(n_subjects):
        spec = group[subj]  # shape: (freqs, epochs)

        pre_spec = np.mean(spec[:, pre_indices], axis=1)
        post_spec = np.mean(spec[:, post_indices], axis=1)

        # Do not log again
        log_pre = pre_spec[freq_mask]
        log_post = post_spec[freq_mask]

        slope_pre, _ = np.polyfit(log_f, log_pre, 1)
        slope_post, _ = np.polyfit(log_f, log_post, 1)

        slopes_pre.append(slope_pre)
        slopes_post.append(slope_post)

    return np.array(slopes_pre), np.array(slopes_post)

def compute_spectral_slope_stats(sleep_group, propofol_group, freqs, pre_indices, post_indices, output_folder='Results'):
    """
    Compute spectral slopes pre/post LoC for each subject in Sleep and Propofol groups,
    and perform within- and between-group statistical tests.

    Saves two CSVs: one for group-level slopes and one for between-group comparison.
    """

    # Use your external compute_subject_slopes function
    sleep_pre, sleep_post = compute_subject_slopes(sleep_group, freqs, pre_indices, post_indices)
    prop_pre, prop_post = compute_subject_slopes(propofol_group, freqs, pre_indices, post_indices)

    # Delta
    delta_sleep = sleep_post - sleep_pre
    delta_prop = prop_post - prop_pre

    # Normality & variance checks
    _, p_shapiro_sleep = shapiro(delta_sleep)
    _, p_shapiro_prop = shapiro(delta_prop)
    _, p_levene = levene(delta_sleep, delta_prop)

    print("NORMALITY")
    print(f"Slope Shapiro Sleep: {p_shapiro_sleep}")
    print(f"Slope Shapiro Prop: {p_shapiro_prop}")
    print(f"SLope Levene Both: {p_levene}")


    # Within-group stats
    if p_shapiro_sleep < 0.05:
        sleep_stat, sleep_p = wilcoxon(sleep_post, sleep_pre)
        sleep_test = "Wilcoxon"
    else:
        sleep_stat, sleep_p = ttest_rel(sleep_post, sleep_pre)
        sleep_test = "Paired t test"

    if p_shapiro_prop < 0.05:
        prop_stat, prop_p = wilcoxon(prop_post, prop_pre)
        prop_test = "Wilcoxon"
    else:
        prop_stat, prop_p = ttest_rel(prop_post, prop_pre)
        prop_test = "Paired t test"

    # Between-groups
    if p_levene < 0.05 or p_shapiro_sleep < 0.05 or p_shapiro_prop < 0.05:
        between_stat, between_p = mannwhitneyu(delta_prop, delta_sleep)
        between_test = "Mannwhitneyu"
    else:
        between_stat, between_p = ttest_ind(delta_prop, delta_sleep)
        between_test = "Independent t test"

    # Save group-level results
    slope_df = pd.DataFrame({
        "Group": ["Propofol", "Sleep"],
        "Pre Slope Mean": [np.mean(prop_pre), np.mean(sleep_pre)],
        "Pre Slope Std": [np.std(prop_pre, ddof=1), np.std(sleep_pre, ddof=1)],
        "Post Slope Mean": [np.mean(prop_post), np.mean(sleep_post)],
        "Post Slope Std": [np.std(prop_post, ddof=1), np.std(sleep_post, ddof=1)],
        "Slope Δ": [np.mean(delta_prop), np.mean(delta_sleep)],
        "Slope Δ Std": [np.std(delta_prop, ddof=1), np.std(delta_sleep, ddof=1)],
        "Stat": [prop_stat, sleep_stat],
        "p": [prop_p, sleep_p],
        "Test": [prop_test, sleep_test]
    })

    slope_between_df = pd.DataFrame({
        "Comparison": ["Propofol vs Sleep Δ Slope"],
        "Stat": [between_stat],
        "p": [between_p],
        "Test": [between_test]
    })

    # Output
    os.makedirs(output_folder, exist_ok=True)
    slope_df.to_csv(os.path.join(output_folder, 'Spectral_Slope_Pre_Post_LoC_full_range.csv'), index=False)
    slope_between_df.to_csv(os.path.join(output_folder, 'Spectral_Slope_Between_Group_Stats_full_range.csv'), index=False)

    print("\n--- Spectral Slope Results ---")
    print(slope_df)
    print("\n--- Between-Group Comparison ---")
    print(slope_between_df)

    # Δ slope per group (re-ordered: Sleep first, then Propofol)
    delta_data = [delta_sleep, delta_prop]
    group_labels = ['Sleep', 'Propofol']
    colors = ['blue', 'red']

    fig, ax = plt.subplots(figsize=(7, 6))

    # Boxplot with custom median line
    bp = ax.boxplot(delta_data, patch_artist=True, labels=group_labels, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor('black')

    # Overlay individual subject points
    for i, (group_deltas, color) in enumerate(zip(delta_data, colors)):
        x_vals = np.random.normal(loc=i + 1, scale=0.05, size=len(group_deltas))  # jitter
        ax.scatter(x_vals, group_deltas, color='black', alpha=0.7, zorder=3)

    # Labels and formatting
    ax.set_title('Change in Spectral Slope (Post - Pre) per Group', fontsize=14)
    ax.set_ylabel('Δ Slope', fontsize=12)
    ax.set_xlabel('')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout(pad=2)

    # Save (optional)
    plt.savefig('Spectral_Slope_Delta_Boxplot_matplotlib.png', dpi=300)
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Helper function to draw paired lines
    def paired_plot(ax, pre, post, color, title, ylabel=False):
        x = [0, 1]
        for i in range(len(pre)):
            ax.plot(x, [pre[i], post[i]], color=color, alpha=0.6, linewidth=1.5)
            ax.scatter(x, [pre[i], post[i]], color='black', zorder=3, s=40)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre', 'Post'])
        ax.set_title(title, fontsize=12)
        if ylabel:
            ax.set_ylabel('Spectral Slope')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Sleep (left) — with ylabel
    paired_plot(axs[0], sleep_pre, sleep_post, color='blue', title='Sleep Group', ylabel=True)

    # Propofol (right) — no ylabel, but keep y-ticks
    paired_plot(axs[1], prop_pre, prop_post, color='red', title='Propofol Group')

    plt.tight_layout(pad=2)

    # Optional save
    plt.savefig('Spectral_Slope_Paired_Pre_Post_full_range matplotlib.png', dpi=300)
    plt.show()


# Calculate slopes for each group:
slopes_sleep = compute_spectral_slope_full_window(sleep_group, frequencies)
slopes_propofol = compute_spectral_slope_full_window(propofol_group, frequencies)

# Run stats:
group_stats, test_stats = compare_spectral_slopes_between_groups(slopes_sleep, slopes_propofol, 'Sleep', 'Propofol')

print(group_stats)
print(test_stats)


compute_spectral_slope_stats(
    sleep_group=sleep_group,
    propofol_group=propofol_group,
    freqs=frequencies,
    pre_indices=pre_indices,
    post_indices=post_indices,
    output_folder='Results'
)




# ---------
# PEAK FREQUENCY
# ---------
from scipy.stats import mode


def get_peak_frequency(freqs, psd, f_range=None):
    """
    Returns frequency of max power in PSD (optionally within a range).
    """
    if f_range:
        mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    else:
        mask = np.ones_like(freqs, dtype=bool)

    peak_idx = np.argmax(psd[mask])
    return freqs[mask][peak_idx]


def extract_peak_frequencies(group_data, freqs, bands, aggregate='mean'):
    """
    Extract peak frequency per subject per band, then aggregate.

    Args:
        group_data: [subjects x freqs x epochs]
        freqs: 1D array of frequency bins
        bands: dict of {band_name: (low, high)}
        aggregate: 'mean' | 'median' | 'mode'

    Returns:
        dict of {band_name: list of peak freqs}
    """
    from scipy.stats import mode
    n_subjects = group_data.shape[0]
    peaks_by_band = {band: [] for band in bands}

    for subj in range(n_subjects):
        psd = group_data[subj].mean(axis=1)  # average over epochs

        for band, f_range in bands.items():
            peak_freq = get_peak_frequency(freqs, psd, f_range)
            peaks_by_band[band].append(peak_freq)

    # Aggregate across subjects
    aggregated_peaks = {}
    for band, values in peaks_by_band.items():
        if aggregate == 'mean':
            aggregated_peaks[band] = np.mean(values)
        elif aggregate == 'median':
            aggregated_peaks[band] = np.median(values)
        elif aggregate == 'mode':
            aggregated_peaks[band] = mode(values, keepdims=False).mode
        else:
            raise ValueError("aggregate must be 'mean', 'median', or 'mode'")

    return aggregated_peaks, peaks_by_band


def plot_subject_peaks(freqs, psd, peaks, bands, title=''):
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, psd, label='PSD')

    for band, freq in peaks.items():
        plt.axvline(freq, linestyle='--', label=f"{band} Peak: {freq:.2f} Hz")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Compute peaks
prop_peaks, all__prop_subject_peaks = extract_peak_frequencies(propofol_group, frequencies, freq_bands, aggregate='median')

# Plot first subject with peaks
psd_0 = propofol_group[0].mean(axis=1)
peaks_0 = {band: get_peak_frequency(frequencies, psd_0, f_range) for band, f_range in freq_bands.items()}

plot_subject_peaks(frequencies, psd_0, peaks_0, freq_bands, title='Subject 1 Peak Frequencies')




# ---------
# SPECTRAL PARAMETERS
# ---------


from specparam import SpectralModel
import numpy as np
import pandas as pd
import os

import numpy as np
import pandas as pd
import os
from specparam import SpectralModel
from mne.stats import permutation_cluster_test

def run_specparam_on_subjects(group_psd, frequencies, f_range=(1, 40), output_csv_prefix=None):
    
    """Run specparam fitting on subject-level average and per-epoch PSDs.

    Parameters:
    - group_psd: ndarray [n_subjects, n_freqs, n_epochs]
    - frequencies: ndarray [n_freqs], in Hz
    - f_range: tuple (f_min, f_max)
    - output_csv_prefix: str, optional base filename to save CSVs

    Returns:
    - summary_df: DataFrame with subject-level average fit
    - epoch_df: DataFrame with per-epoch fits"""
    

    # Remove 0 or negative frequencies



    valid_freqs = (frequencies >= 1)
    frequencies = frequencies[valid_freqs]

    print(valid_freqs)
    print(frequencies)

    if np.any(frequencies <= 0):
        raise ValueError("Frequency array contains 0 or negative values, which are invalid for log operations.")

    group_psd = group_psd[:, valid_freqs, :]

    n_subjects, n_freqs, n_epochs = group_psd.shape
    summary_results = []
    epoch_results = []

    for subj_idx in range(n_subjects):
        subj_psd = group_psd[subj_idx]  # shape: [n_freqs, n_epochs]
        subj_summary = {'Subject': subj_idx}

        try:
            mean_psd = np.mean(subj_psd, axis=1)
            sm = SpectralModel(aperiodic_mode='knee', verbose=False)
            sm.fit(frequencies, mean_psd, f_range)

            subj_summary.update({
                'Aperiodic_Offset': sm.aperiodic_params_[0],
                'Aperiodic_Knee': sm.aperiodic_params_[1],
                'Aperiodic_Exponent_Knee': sm.aperiodic_params_[2],
                'Num_Peaks': len(sm.peak_params_),
            })

            if sm.peak_params_.size > 0:
                peak = max(sm.peak_params_, key=lambda x: x[1])  # [CF, PW, BW]
                subj_summary.update({
                    'Peak_CF': peak[0],
                    'Peak_Power': peak[1],
                    'Peak_BW': peak[2],
                })
            else:
                subj_summary.update({
                    'Peak_CF': np.nan,
                    'Peak_Power': np.nan,
                    'Peak_BW': np.nan,
                })

        except Exception as e:
            print(f"[Subject-Average PSD] Subject {subj_idx} failed: {e}")
            subj_summary.update({
                'Aperiodic_Offset': np.nan,
                'Aperiodic_Slope': np.nan,
                'Num_Peaks': 0,
                'Peak_CF': np.nan,
                'Peak_Power': np.nan,
                'Peak_BW': np.nan,
            })

        summary_results.append(subj_summary)

        # === Epoch-level fitting ===
        for epoch_idx in range(n_epochs):
            epoch_result = {'Subject': subj_idx, 'Epoch': epoch_idx}
            try:
                epoch_psd = subj_psd[:, epoch_idx]
                sm_epoch = SpectralModel(aperiodic_mode='fixed', verbose=False)
                sm_epoch.fit(frequencies, epoch_psd, f_range)

                epoch_result.update({
                    'Aperiodic_Offset': sm_epoch.aperiodic_params_[0],
                    'Aperiodic_Slope': sm_epoch.aperiodic_params_[1],
                    'Num_Peaks': len(sm_epoch.peak_params_),
                })

                if sm_epoch.peak_params_.size > 0:
                    peak = max(sm_epoch.peak_params_, key=lambda x: x[1])
                    epoch_result.update({
                        'Peak_CF': peak[0],
                        'Peak_Power': peak[1],
                        'Peak_BW': peak[2],
                    })
                else:
                    epoch_result.update({
                        'Peak_CF': np.nan,
                        'Peak_Power': np.nan,
                        'Peak_BW': np.nan,
                    })

            except Exception as e:
                print(f"[Epoch PSD] Subject {subj_idx} Epoch {epoch_idx} failed: {e}")
                epoch_result.update({
                    'Aperiodic_Offset': np.nan,
                    'Aperiodic_Slope': np.nan,
                    'Num_Peaks': 0,
                    'Peak_CF': np.nan,
                    'Peak_Power': np.nan,
                    'Peak_BW': np.nan,
                })

            epoch_results.append(epoch_result)

    # Convert to DataFrames
    summary_df = pd.DataFrame(summary_results)
    epoch_df = pd.DataFrame(epoch_results)

    # Optional save
    if output_csv_prefix:
        os.makedirs(os.path.dirname(output_csv_prefix), exist_ok=True)
        summary_df.to_csv(f'{output_csv_prefix}_summary.csv', index=False)
        epoch_df.to_csv(f'{output_csv_prefix}_epochs.csv', index=False)
        print(f"Saved: {output_csv_prefix}_summary.csv and _epochs.csv")

    return summary_df, epoch_df


sleep_group_linear = 10 ** sleep_group  # Convert from log10(power) to linear

sleep_summary_df, sleep_epoch_df = run_specparam_on_subjects(
    sleep_group_linear,
    frequencies,
    f_range=(1, 49.5),
    output_csv_prefix='Results/sleep_specparam_new'
)

propofol_group_linear = 10 ** propofol_group  # Convert from log10(power) to linear

propofol_summary_df, propofol_epoch_df = run_specparam_on_subjects(
    propofol_group_linear,
    frequencies,
    f_range=(1, 49.5),
    output_csv_prefix='Results/propofol_specparam_new'
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# === Load per-epoch data ===
sleep_df = pd.read_csv("Results/sleep_specparam_new_epochs.csv")
prop_df = pd.read_csv("Results/propofol_specparam_new_epochs.csv")

# Add condition labels
sleep_df["Condition"] = "Sleep"
prop_df["Condition"] = "Propofol"

# Combine and clean
df = pd.concat([sleep_df, prop_df], ignore_index=True)
df["Subject"] = df["Subject"].astype(str)

# Smooth metrics
for col in ["Peak_CF", "Peak_Power"]:
    df[f"{col}_Smoothed"] = df.groupby(["Condition"])[col].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )

# Remove extreme values
df = df[(df["Aperiodic_Slope"] > -6) & (df["Aperiodic_Slope"] < 2)]
df = df[(df["Peak_CF"] > 1) & (df["Peak_CF"] < 40)]
df = df[df["Peak_Power"] < 50]

palette = {"Sleep": "blue", "Propofol": "red"}

import seaborn as sns
import numpy as np

# Define color palette
palette = {"Sleep": "blue", "Propofol": "red"}

epoch_duration = 2  # seconds per epoch
# Step 1: Create mapping from epoch index to time value
epoch_to_time = dict(zip(range(len(time)), time))

# Step 2: Apply to your DataFrame
df["Time"] = df["Epoch"].map(epoch_to_time)

# Plot loop
for metric in ["Peak_CF_Smoothed", "Peak_Power_Smoothed"]:
    plt.figure(figsize=(12, 5))

    sns.lineplot(
        data=df,
        x="Time",
        y=metric,
        hue="Condition",
        estimator="median",
        errorbar=("ci", 95),
        palette=palette
    )

    plt.axvline(0, color='black', linestyle='--', linewidth=1, label=None)


    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(f"{metric.replace('_', ' ')} Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel(metric.replace("_", " "))
    plt.legend()
    plt.xlim(df["Time"].min(), df["Time"].max())
    plt.xticks(np.arange(df["Time"].min(), df["Time"].max() + 3, 60))  # ticks every 60s
    plt.tight_layout()
    plt.savefig(f"{metric}_specparam_over_time_median.png", dpi=300)
    plt.show()





# === Cluster Permutation Tests (Unpaired, Independent Groups) ===

def stat_fun(x, y):
    t_vals, _ = ttest_ind(x, y, axis=0, equal_var=False, nan_policy="omit")
    return t_vals

def run_cluster_test(metric_name, df, epoch_duration=2):
    # Pivot data: rows = subjects, columns = epochs
    sleep_pivot = df[df["Condition"] == "Sleep"].pivot(index="Subject", columns="Epoch", values=metric_name)
    prop_pivot = df[df["Condition"] == "Propofol"].pivot(index="Subject", columns="Epoch", values=metric_name)

    # Keep only shared epochs
    shared_epochs = sorted(set(sleep_pivot.columns).intersection(prop_pivot.columns))
    if not shared_epochs:
        print(f"Skipping {metric_name} — no shared epochs.")
        return None

    sleep_data = sleep_pivot[shared_epochs].to_numpy(dtype=np.float64)
    prop_data = prop_pivot[shared_epochs].to_numpy(dtype=np.float64)

    # Drop subjects with too few valid epochs (e.g., < 10 non-NaNs)
    def clean_subjects(data, min_valid=10):
        return data[np.sum(~np.isnan(data), axis=1) >= min_valid]

    sleep_data = clean_subjects(sleep_data)
    prop_data = clean_subjects(prop_data)

    print(f"\nMetric: {metric_name}")
    print(f"Sleep shape: {sleep_data.shape} | Propofol shape: {prop_data.shape}")

    if sleep_data.size == 0 or prop_data.size == 0:
        print("Skipping — insufficient data.")
        return None

    # Time axis (epoch index × duration)
    time = np.array(shared_epochs) * epoch_duration

    # Perform cluster-based permutation test (independent two-sided t-test)
    T_obs, clusters, cluster_pv, _ = permutation_cluster_test(
        [sleep_data, prop_data],
        stat_fun=stat_fun,
        n_permutations=1000,
        tail=0,
        threshold=None,
        out_type="mask",
        n_jobs=1,
        verbose=True
    )

    # Extract significant clusters
    cluster_info = []
    for i, (mask, p_val) in enumerate(zip(clusters, cluster_pv)):
        if p_val < 0.05:
            time_points = time[mask]
            cluster_info.append({
                "Metric": metric_name,
                "Cluster": i + 1,
                "Start_Time": time_points[0],
                "End_Time": time_points[-1],
                "p-value": p_val,
                "Max_t": np.nanmax(T_obs[mask]),
                "Min_t": np.nanmin(T_obs[mask]),
                "Cluster_Size": np.sum(mask)
            })

    # === Plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(time, np.nanmedian(sleep_data, axis=0), label="Sleep", color="blue")
    plt.plot(time, np.nanmedian(prop_data, axis=0), label="Propofol", color="red")

    for i_c, (mask, p_val) in enumerate(zip(clusters, cluster_pv)):
        if p_val < 0.05:
            plt.axvspan(time[mask][0], time[mask][-1], color="orange", alpha=0.3,
                        label="Significant cluster" if i_c == 0 else None)

    plt.xlabel("Time (s)")
    plt.ylabel(metric_name)
    plt.title(f"Cluster Permutation Test: {metric_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(cluster_info)


# === Run Tests on Selected Metrics ===
metrics_to_test = ["Aperiodic_Slope", "Peak_CF", "Peak_Power"]
cluster_dfs = []

for metric in metrics_to_test:
    df_cluster = run_cluster_test(metric, df)
    if df_cluster is not None and not df_cluster.empty:
        cluster_dfs.append(df_cluster)

# Save results
if cluster_dfs:
    all_clusters_df = pd.concat(cluster_dfs, ignore_index=True)
    all_clusters_df.to_csv("Results/cluster_results.csv", index=False)
    print("Cluster results saved to Results/cluster_results.csv")
else:
    print("No significant clusters found.")


# === Subject-Level Mann–Whitney U Tests ===

# Load summary-level files
sleep_summary = pd.read_csv("Results/sleep_specparam_new_summary.csv")
prop_summary = pd.read_csv("Results/propofol_specparam_new_summary.csv")

sleep_summary["Condition"] = "Sleep"
prop_summary["Condition"] = "Propofol"

summary_df = pd.concat([sleep_summary, prop_summary], ignore_index=True)

metrics = ["Aperiodic_Offset", "Aperiodic_Knee", "Aperiodic_Exponent_Knee", "Num_Peaks", "Peak_CF", "Peak_Power", "Peak_BW"]
results = []

for metric in metrics:
    x = prop_summary[metric].dropna()
    y = sleep_summary[metric].dropna()

    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    results.append({
        "Metric": metric,
        "Propofol Median": x.median(),
        "Sleep Median": y.median(),
        "U Statistic": stat,
        "p-value": p
    })

# Save and print
results_df = pd.DataFrame(results)
results_df.to_csv("Results/mannwhitney_results_new.csv", index=False)
print("\nMann–Whitney U test results saved to Results/mannwhitney_results_new.csv")
print(results_df)


# -----------------------------
# Time-Resolved Band Power
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_ind
import seaborn as sns

def compute_mean_power_all_freqs(spectrogram, freqs, smooth=False, plot=True, label=None):
    """
    Compute mean power across all frequencies per epoch, for each subject.

    Parameters:
    - spectrogram: 3D array (subjects x freqs x epochs), in dB
    - freqs: 1D array of frequency values
    - smooth: whether to smooth the time series
    - plot: whether to plot
    - label: optional label for plotting

    Returns:
    - mean_power: list of arrays (one per subject), each shape (n_epochs,)
    """
    mean_power = []
    for subj in range(spectrogram.shape[0]):
        subj_spec = spectrogram[subj]  # shape: (freqs x epochs)
        subj_mean = np.mean(subj_spec, axis=0)  # mean over freqs per epoch

        if smooth:
            from scipy.ndimage import uniform_filter1d
            subj_mean = uniform_filter1d(subj_mean, size=5)

        mean_power.append(subj_mean)

    # Optional plot: group-level mean ± std
    if plot:
        mean_array = np.stack(mean_power)  # shape: (subjects x epochs)
        mean = np.mean(mean_array, axis=0)
        std = np.std(mean_array, axis=0)

        plt.figure(figsize=(12, 5))
        plt.plot(mean, label=label or 'Mean Power (All Freqs)', color='black')
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3, color='gray')
        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Power (dB)')
        plt.title('Mean Power Over Time (All Frequencies)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return mean_power

epoch_length_sec = 2  # change to match your data
n_epochs = sleep_group.shape[-1]
mid_epoch = n_epochs // 2
time = (np.arange(n_epochs) - mid_epoch) * epoch_length_sec  # in seconds

# Compute mean power per epoch
sleep_mean_power = compute_mean_power_all_freqs(sleep_group, frequencies, plot=False, label='Sleep')
prop_mean_power = compute_mean_power_all_freqs(propofol_group, frequencies, plot=False, label='Propofol')


# --- Convert lists to arrays: shape (n_subjects, n_epochs) ---
sleep_power_array = np.stack(sleep_mean_power)
prop_power_array = np.stack(prop_mean_power)

# --- Compute group means and stds per epoch ---
sleep_mean = np.mean(sleep_power_array, axis=0)
sleep_std = np.std(sleep_power_array, axis=0)
prop_mean = np.mean(prop_power_array, axis=0)
prop_std = np.std(prop_power_array, axis=0)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(time, sleep_mean, label='Sleep', color='blue')
plt.fill_between(time, sleep_mean - sleep_std, sleep_mean + sleep_std, alpha=0.3, color='blue')

plt.plot(time, prop_mean, label='Propofol', color='red')
plt.fill_between(time, prop_mean - prop_std, prop_mean + prop_std, alpha=0.3, color='red')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axvline(0, color='black', linestyle='--', linewidth=1)  # LoC marker
plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Baseline power
plt.title('Mean Power Over Time (All Frequencies)')
plt.xlabel('Time (s)')
plt.ylabel('Mean Power (dB)')
plt.legend()
plt.xlim(time[0], time[-1])
plt.xticks(np.arange(time[0], 241, 60))  # ticks every 60 seconds
plt.tight_layout()
plt.savefig('mean_power_over_time.png', dpi=300)
plt.show()





# --- Compute slope of mean power per subject ---
sleep_slopes = [linregress(time, subj_curve).slope for subj_curve in sleep_power_array]
prop_slopes = [linregress(time, subj_curve).slope for subj_curve in prop_power_array]

print("Sleep group:")
stat1, p1 = shapiro(sleep_slopes)
print(f"Shapiro-Wilk: W={stat1:.3f}, p={p1:.4f}")

print("\nPropofol group:")
stat2, p2 = shapiro(prop_slopes)
print(f"Shapiro-Wilk: W={stat2:.3f}, p={p2:.4f}")


# Mann-Whitney U test
u_stat, p_val = mannwhitneyu(sleep_slopes, prop_slopes, alternative='two-sided')
sig_label = '*' if p_val < 0.05 else 'n.s.'

plt.figure(figsize=(8, 6))

boxprops = dict(linewidth=2)
medianprops = dict(color='black', linewidth=2)


# Draw boxplots
bp = plt.boxplot([sleep_slopes, prop_slopes],
                 labels=['Sleep', 'Propofol'],
                 patch_artist=True,
                 boxprops=boxprops,
                 medianprops=medianprops,
                 )

# Colors for boxes
colors = ['blue', 'red']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

y_min = min(min(sleep_slopes), min(prop_slopes))
y_max = max(max(sleep_slopes), max(prop_slopes))
y_range = y_max - y_min

# Expand both limits by 10% of the data range
plt.ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)

# Add significance bar
x1, x2 = 1, 2
bar_height = y_max + 0.1 * y_range
bar_thickness = 0.02 * y_range
plt.plot([x1, x1, x2, x2],
         [bar_height, bar_height + bar_thickness, bar_height + bar_thickness, bar_height],
         lw=1.5, c='k')

# Place the asterisk slightly lower to avoid overlap
plt.text((x1 + x2) * 0.5, bar_height + bar_thickness + 0.01 * y_range, '**',
         ha='center', va='bottom', color='k', fontsize=20)


plt.ylabel('Slope of Mean Power Over Time')
plt.title('Mean Power Slope per Subject', y=1.07)

plt.axhline(0, linestyle='--', color='gray')


plt.tight_layout(pad=3)  # Add padding so nothing overlaps the title
plt.savefig('mean_power_slope_boxplot_new.png', dpi=300)
plt.show()


# --- Print slope stats ---
print("SLOPE STATS FOR MEAN POWER PER EPOCH WHOLE SPECTROGRAM")
print("Sleep slopes: mean =", np.mean(sleep_slopes), "std =", np.std(sleep_slopes))
print("Propofol slopes: mean =", np.mean(prop_slopes), "std =", np.std(prop_slopes))





def compute_band_power(spec, freqs, band, smooth=False, label=None, plot=False):
    """
    Compute and optionally plot band power (in dB) over epochs.

    Parameters:
    - spec: 2D array (freqs x epochs), in dB
    - freqs: 1D array of frequency values
    - band: tuple (low_freq, high_freq)
    - smooth: bool, whether to apply simple moving average
    - label: str, optional label for plotting
    - plot: bool, whether to plot the results

    Returns:
    - band_power: 1D array of band power per epoch
    """
    low, high = band
    band_mask = (freqs >= low) & (freqs < high)

    # Average power in the frequency band
    band_power = np.mean(spec[band_mask, :], axis=0)

    if smooth:
        from scipy.ndimage import uniform_filter1d
        band_power = uniform_filter1d(band_power, size=5)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(band_power, label=label or f'{low}-{high} Hz Band Power')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Power (dB relative to baseline)')
        plt.title(f'Time-Resolved Band Power: {low}-{high} Hz')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return band_power


from scipy.stats import sem
def compute_sliding_window_slopes(data, window_size=10):
    """
    Compute linear regression slope over sliding windows.

    Parameters:
    - data: np.array of shape (n_subjects, n_epochs)
    - window_size: number of epochs in each window

    Returns:
    - slopes: np.array of shape (n_subjects, n_windows)
    """
    n_subjects, n_epochs = data.shape
    n_windows = n_epochs - window_size + 1
    slopes = np.zeros((n_subjects, n_windows))

    for subj in range(n_subjects):
        for win_start in range(n_windows):
            win_end = win_start + window_size
            y = data[subj, win_start:win_end]
            x = np.arange(window_size)
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            slopes[subj, win_start] = slope

    return slopes


# Assuming you already have:
# - sleep_data: shape (n_subjects, n_freqs, n_epochs)
# - propofol_data: shape (n_subjects, n_freqs, n_epochs)
# - freqs: 1D array of frequency bins
# - freq_bands: your dictionary of frequency band ranges

sleep_band_powers = {}
propofol_band_powers = {}
# Store sliding window slopes
sleep_band_slopes = {}
propofol_band_slopes = {}
slopes_list = []
window_size = 5  # Adjust as needed

# Create array of bands
for band_name, band_range in freq_bands.items():
    # Each will be shape (n_subjects, n_epochs)
    sleep_band = np.array([
        compute_band_power(subject_data, frequencies, band_range, plot=False) for subject_data in sleep_group
    ])
    prop_band = np.array([
        compute_band_power(subject_data, frequencies, band_range, plot=False) for subject_data in propofol_group
    ])

    sleep_band_powers[band_name] = sleep_band
    propofol_band_powers[band_name] = prop_band

    # Compute sliding window slopes
    sleep_slopes = compute_sliding_window_slopes(sleep_band, window_size=window_size)  # shape: (n_subjects, n_windows)
    prop_slopes = compute_sliding_window_slopes(prop_band, window_size=window_size)

    sleep_band_slopes[band_name] = sleep_slopes
    propofol_band_slopes[band_name] = prop_slopes

    # Plotting mean ± SEM over windows
    epochs = np.arange(sleep_slopes.shape[1])
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, sleep_slopes.mean(axis=0), label='Sleep', color='blue')
    plt.fill_between(epochs,
                     sleep_slopes.mean(axis=0) - sem(sleep_slopes, axis=0),
                     sleep_slopes.mean(axis=0) + sem(sleep_slopes, axis=0),
                     color='blue', alpha=0.3)

    plt.plot(epochs, prop_slopes.mean(axis=0), label='Propofol', color='green')
    plt.fill_between(epochs,
                     prop_slopes.mean(axis=0) - sem(prop_slopes, axis=0),
                     prop_slopes.mean(axis=0) + sem(prop_slopes, axis=0),
                     color='green', alpha=0.3)

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(120, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Sliding Window Slope – {band_name} Band")
    plt.xlabel("Window Start Epoch")
    plt.ylabel("Slope (Power/epoch)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, prop_slopes.mean(axis=0), label='Propofol', color='green')
    plt.fill_between(epochs,
                     prop_slopes.mean(axis=0) - sem(prop_slopes, axis=0),
                     prop_slopes.mean(axis=0) + sem(prop_slopes, axis=0),
                     color='green', alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(120, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Sliding Window Propofol Slope – {band_name} Band")
    plt.xlabel("Window Start Epoch")
    plt.ylabel("Slope (Power/epoch)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, sleep_slopes.mean(axis=0), label='Sleep', color='blue')
    plt.fill_between(epochs,
                     sleep_slopes.mean(axis=0) - sem(sleep_slopes, axis=0),
                     sleep_slopes.mean(axis=0) + sem(sleep_slopes, axis=0),
                     color='blue', alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(120, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Sliding Window Propofol Slope – {band_name} Band")
    plt.xlabel("Window Start Epoch")
    plt.ylabel("Slope (Power/epoch)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Append to dataframe
    n_windows = sleep_slopes.shape[1]
    for subj_idx in range(sleep_slopes.shape[0]):
        for win in range(n_windows):
            slopes_list.append({
                "Subject": f"S{subj_idx + 1}",
                "Group": "Sleep",
                "Band": band_name,
                "Window": win,
                "Slope": sleep_slopes[subj_idx, win]
            })

    for subj_idx in range(prop_slopes.shape[0]):
        for win in range(n_windows):
            slopes_list.append({
                "Subject": f"P{subj_idx + 1}",
                "Group": "Propofol",
                "Band": band_name,
                "Window": win,
                "Slope": prop_slopes[subj_idx, win]
            })

# Convert to DataFrame
slopes_df = pd.DataFrame(slopes_list)

# Example: inspect or save
print(slopes_df.head())



print("LOOOOOK HERERERERERE")
print(f'Number of epochs: {n_epochs}')
print(f"mid_epoch: {mid_epoch}")
print(f"time: {time}")

#Plot time-resolved band power as the median with credible interval
for band_name in freq_bands.keys():
    sleep_band = sleep_band_powers[band_name]
    prop_band = propofol_band_powers[band_name]

    median_sleep = np.median(sleep_band, axis=0)
    median_prop = np.median(prop_band, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(time, median_sleep, label='Sleep (median)', color='blue', linewidth=2)
    plt.plot(time, median_prop, label='Propofol (median)', color='red', linewidth=2)
    plt.axvline(0, color='black', linestyle='--', linewidth=2)  # LoC marker
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Baseline power
    plt.title(f'{band_name} Band Power Over Time (relative to baseline) ')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (dB relative to baseline)')
    plt.legend(frameon=False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    lower = np.percentile(sleep_band, 2.5, axis=0)
    upper = np.percentile(sleep_band, 97.5, axis=0)
    plt.fill_between(time, lower, upper, color='blue', alpha=0.2, label='Sleep 95% CrI')
    lower = np.percentile(prop_band, 2.5, axis=0)
    upper = np.percentile(prop_band, 97.5, axis=0)
    plt.fill_between(time, lower, upper, color='red', alpha=0.2, label='Propofol 95% CrI')
    plt.tight_layout()
    plt.xlim(time[0], time[-1])  # explicitly set x limits to start/end of your time array
    plt.margins(x=0)
    # Set file path
    """output_folder = 'Results'
    filename = os.path.join(output_folder, f"{band_name} Sleep-Propofol.png")  # or .pdf, .svg, .jpg

    # Create the folder
    os.makedirs(output_folder, exist_ok=True)

    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # high resolution"""
    plt.show()

# Plot time-resolved band power as the mean with credible interval
for band_name in freq_bands.keys():
    sleep_band = sleep_band_powers[band_name]
    prop_band = propofol_band_powers[band_name]

    mean_sleep = np.mean(sleep_band, axis=0)
    mean_prop = np.mean(prop_band, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(time, mean_sleep, label='Sleep (mean)', color='blue', linewidth=2)
    plt.plot(time, mean_prop, label='Propofol (mean)', color='red', linewidth=2)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)  # LoC marker
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Baseline power
    plt.title(f'{band_name} Band Power Over Time (relative to baseline) ')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (dB relative to baseline)')
    plt.legend(frameon=False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    lower = np.percentile(sleep_band, 2.5, axis=0)
    upper = np.percentile(sleep_band, 97.5, axis=0)
    plt.fill_between(time, lower, upper, color='blue', alpha=0.2, label='Sleep 95% CrI')
    lower = np.percentile(prop_band, 2.5, axis=0)
    upper = np.percentile(prop_band, 97.5, axis=0)
    plt.fill_between(time, lower, upper, color='red', alpha=0.2, label='Propofol 95% CrI')
    plt.tight_layout()
    plt.xlim(time[0], time[-1])  # explicitly set x limits to start/end of your time array
    plt.margins(x=0)
    # Set file path
    """output_folder = 'Results'
    filename = os.path.join(output_folder, f"{band_name} Sleep-Propofol.png")  # or .pdf, .svg, .jpg
    # Create the folder
    os.makedirs(output_folder, exist_ok=True)
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # high resolution"""
    plt.show()



"""# PERMUTATION CLUSTER TESTS (NOT WRITTEN)
from mne.stats import permutation_cluster_test
def run_cluster_test(sleep_data, propofol_data, n_permutations=1000, threshold=None):
    
    Run cluster-based permutation test on time-resolved data between two conditions.
    Input shape: (n_subjects, n_epochs) for both sleep_data and propofol_data
    
    X = [sleep_data, propofol_data]  # Must be list of arrays: [group1, group2]
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        X, n_permutations=n_permutations, threshold=threshold, tail=0, out_type='mask'
    )
    return T_obs, clusters, cluster_p_values
def plot_cluster_results(T_obs, clusters, cluster_p_values, times, alpha=0.05):
    plt.figure(figsize=(12, 5))
    plt.plot(times, T_obs, label='Observed T-values')
    plt.axhline(0, color='black', linestyle='--')

    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] < alpha:
            plt.axvspan(times[c], times[c][-1], color='red', alpha=0.3,
                        label='Significant cluster' if i_c == 0 else None)

    plt.xlabel('Time (s)')
    plt.ylabel('T-statistic')
    plt.title(f'Cluster-Based Permutation Test (alpha={alpha})')
    plt.legend()
    plt.tight_layout()
    plt.show()
"""


# ----------
# Time-Resolved Band Power Metrics
# ----------


from scipy.stats import linregress








# Setup
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
from scipy.stats import linregress
import pandas as pd
import pingouin as pg

# Optional: store results
all_cluster_results = {}
all_trends = {}
all_anovas = {}

from mne.stats import permutation_cluster_test
cluster_results = {}


# Loop over each frequency band
for band_name in freq_bands.keys():
    print(f"\n=== Processing {band_name} Band ===")

    sleep_data = sleep_band_powers[band_name]  # shape: (n_sleep_subjects, n_epochs)
    propofol_data = propofol_band_powers[band_name]  # shape: (n_propofol_subjects, n_epochs)

    # ------------------------------------
    # 1. Cluster-Based Permutation Test
    # ------------------------------------
    data_diff = sleep_data - propofol_data  # shape: (n_subjects, n_times)

    # Run cluster-based permutation test
    T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
        [sleep_data, propofol_data],
        tail=0,  # two-tailed
        n_permutations=1000,
        threshold=None,
        out_type='indices',  # or 'mask' if your version supports it
        seed=42,
    )



    plt.figure(figsize=(10, 5))
    plt.plot(time, np.mean(sleep_data, axis=0), label='Sleep', color='blue')
    plt.plot(time, np.mean(propofol_data, axis=0), label='Propofol', color='red')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)  # LoC marker
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Baseline power

    for i_c, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
        if p_val <= 0.05:
            cluster_times = time[cluster]
            plt.axvspan(cluster_times[0], cluster_times[-1], color='orange', alpha=0.3,
                        label=None if i_c == 0 else "")

    plt.title(f'{band_name} Band Power (Significant Clusters Highlighted)')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (dB)')
    ax = plt.gca()




    plt.xlim(time[0], time[-1])
    plt.xticks(np.arange(time[0], 241, 60))  # ticks every 60 seconds
    plt.savefig(f'{band_name}_power_with_sig_clusters.png', dpi=300)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Store results
    cluster_results[band_name] = {
        'T_obs': T_obs,
        'clusters': clusters,
        'p_values': cluster_p_values,
        'times': time,
    }

    # ------------------------------------
    # 2. Trend Analysis (slope over time)
    # ------------------------------------
    print("Running trend analysis...")


    def run_trend_analysis(data, times, alpha=0.05):
        slopes, intercepts, pvals, direction, significant = [], [], [], [], []
        for subj_data in data:
            slope, intercept, r, p, stderr = linregress(times, subj_data)
            slopes.append(slope)
            intercepts.append(intercept)
            pvals.append(p)

            if slope > 0:
                direct = 'Increase'
            elif slope < 0:
                direct = 'Decrease'
            else:
                direct = 'Neutral'
            direction.append(direct)

            if p < alpha:
                sig = 'YES'
            else:
                sig = 'NO'
            significant.append(sig)

        return slopes, intercepts, pvals, direction, significant



    slopes_sleep, _, pvals_sleep, direction_sleep, significant_sleep = run_trend_analysis(sleep_data, time)
    slopes_prop, _, pvals_prop, direction_prop, significant_prop = run_trend_analysis(propofol_data, time)

    between_stat, between_p = mannwhitneyu(slopes_sleep, slopes_prop, alternative='two-sided')
    print(f"Slope comparison (Mann–Whitney U): p = {between_p:.4f}")
    if between_p < 0.05:
        print("SIGNIFICANT DIFFERENCE")
        between_significant = 'SIGNIFICANT DIFFERENCE'
    else:
        between_significant = 'NO SIGNIFICANT DIFFERENCE'

    all_trends[band_name] = {
        'sleep_slopes': slopes_sleep,
        'propofol_slopes': slopes_prop,
        'sleep_pvals': pvals_sleep,
        'propofol_pvals': pvals_prop,
        'direction_sleep': direction_sleep,
        'direction_prop': direction_prop,
        'sleep_significant': significant_sleep,
        'propofol_significant': significant_prop,
        'between_stat': between_stat,
        'between_p': between_p,
        'between_significant': between_significant,
    }

    df_slope_trends =pd.DataFrame(all_trends[band_name])
    pd.options.display.max_columns = 12
    print(df_slope_trends)

    output_folder = 'Results'
    df_slope_trends.to_csv(os.path.join(output_folder, f'Time-Resolved {band_name} Band Power Slope Trends.csv'), index=False)


    """import seaborn as sns

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=[slopes_sleep, slopes_prop], palette=['blue', 'red'])
    plt.xticks([0, 1], ['Sleep', 'Propofol'])
    plt.ylabel('Slope of Band Power Over Time')
    plt.title(f'{band_name} Band – Trend Comparison')
    plt.tight_layout()
    plt.show()
"""



    # ------------------------------------
    # 3. Mixed ANOVA
    # ------------------------------------
    print("Running mixed ANOVA...")
    from statsmodels.stats.multitest import multipletests


    def prepare_anova_df(sleep_data, propofol_data, times):
        df_list = []
        for cond, group_data in zip(['Sleep', 'Propofol'], [sleep_data, propofol_data]):
            for subj_idx, subj_data in enumerate(group_data):
                for t_idx, value in enumerate(subj_data):
                    df_list.append({
                        'Subject': f'{cond}_{subj_idx}',
                        'Condition': cond,
                        'Time': times[t_idx],
                        'Power': value
                    })
        return pd.DataFrame(df_list)


    # Run ANOVA per band and store
    df = prepare_anova_df(sleep_data, propofol_data, time)
    anova = pg.mixed_anova(data=df, dv='Power', within='Time', between='Condition', subject='Subject')
    all_anovas[band_name] = anova
    print(anova)

# Assuming all_anovas is a dict: {band_name: anova_df}
anova_summary_list = []

for band, df in all_anovas.items():
    df = df.copy()  # avoid modifying original
    df['Band'] = band
    anova_summary_list.append(df)

# Concatenate all results into one publication-style summary DataFrame
anova_summary_df = pd.concat(anova_summary_list, ignore_index=True)

for source in anova_summary_df['Source'].unique():
    mask = anova_summary_df['Source'] == source
    pvals = anova_summary_df.loc[mask, 'p-unc'].values
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    anova_summary_df.loc[mask, 'p-FDR'] = pvals_fdr

# Reorder columns for publication-style
columns_order = ['Band', 'Source', 'SS', 'DF1', 'DF2', 'MS', 'F', 'p-unc', 'p-FDR', 'np2']
anova_summary_df = anova_summary_df[columns_order]



# Save as CSV
anova_summary_df.to_csv(os.path.join(output_folder, "Time Resolved ANOVA Band Measure.csv"), index=False)

# (Optional) preview
print(anova_summary_df.head())



import matplotlib.patches as mpatches
#Box plot for slopes
# Prepare data for plotting
bands = list(all_trends.keys())
sleep_slopes_all = [all_trends[band]['sleep_slopes'] for band in bands]
prop_slopes_all = [all_trends[band]['propofol_slopes'] for band in bands]
between_ps = [all_trends[band]['between_p'] for band in bands]

# Plot settings (unchanged)
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.35
x = np.arange(len(bands))

for i in range(len(bands)):
    # Sleep box
    bp_sleep = ax.boxplot(sleep_slopes_all[i],
                          positions=[x[i] - width / 2],
                          widths=width * 0.9,
                          patch_artist=True,
                          boxprops=dict(facecolor='blue', alpha=0.5),
                          medianprops=dict(color='black', linewidth=2)
                          )

    # Propofol box
    bp_prop = ax.boxplot(prop_slopes_all[i],
                         positions=[x[i] + width / 2],
                         widths=width * 0.9,
                         patch_artist=True,
                         boxprops=dict(facecolor='red', alpha=0.5),
                         medianprops=dict(color='black', linewidth=2)
                         )

    # Significance bar if p < 0.05
    p = between_ps[i]
    if p < 0.05:
        y_max = max(max(sleep_slopes_all[i]), max(prop_slopes_all[i]))
        y = y_max + 0.005
        h = 0.003
        x1 = x[i] - width / 2
        x2 = x[i] + width / 2
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black')

        stars = pval_to_stars(p)
        ax.text(x[i], y + h + 0.0001, stars, ha='center', va='bottom', fontsize=16)

# Formatting (unchanged)
ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.set_ylabel('Slope of Band Power Over Time')
ax.set_title('Band Power Slope Comparison')
ax.axhline(0, linestyle='--', color='gray')
ax.set_xlim(-0.5, len(bands) - 0.5)

# Create legend handles manually
sleep_patch = mpatches.Patch(color='blue', alpha=0.5, label='Sleep')
prop_patch = mpatches.Patch(color='red', alpha=0.5, label='Propofol')

# Add legend to plot
ax.legend(handles=[sleep_patch, prop_patch], loc='upper right')
plt.tight_layout()

# Save figure
plt.savefig('All_Bands_Slope_Comparison_Boxplot_new_new.png', dpi=300)
plt.show()


print("CLUSTERS FOR MEAN BAND POWER")
summary_rows = []
for band, result in cluster_results.items():
    for i, (cluster_inds, p_val) in enumerate(zip(result['clusters'], result['p_values'])):
        if p_val <= 0.05:
            t_start = result['times'][cluster_inds][0]
            t_end = result['times'][cluster_inds][-1]
            summary_rows.append({
                'Band': band,
                'Cluster #': i + 1,
                'p-value': round(p_val, 4),
                'Start Time (s)': round(t_start, 2),
                'End Time (s)': round(t_end, 2),
                'Duration (s)': round(t_end - t_start, 2),
            })

summary_df = pd.DataFrame(summary_rows)
print("\n--- Significant Cluster Summary ---")
print(summary_df if not summary_df.empty else "No significant clusters found.")

summary_df.to_csv(os.path.join(output_folder, 'Time-Resolved Band Power Clusters.csv'), index=False)

bands = ['Delta', 'Theta', 'Alpha', 'Low Beta', 'High Beta', 'Gamma']
sources = ['Condition', 'Time', 'Interaction']
colors = ['blue', 'orange', 'green']

# Extract np2 and p-FDR values in the correct order
np2_vals = {src: [] for src in sources}
pvals = {src: [] for src in sources}
for band in bands:
    for src in sources:
        row = anova_summary_df[(anova_summary_df['Band'] == band) & (anova_summary_df['Source'] == src)]
        if not row.empty:
            np2_vals[src].append(row['np2'].values[0])
            pvals[src].append(row['p-FDR'].values[0])
        else:
            np2_vals[src].append(np.nan)
            pvals[src].append(np.nan)

fig, ax = plt.subplots(figsize=(12, 6))
width = 0.25
x = np.arange(len(bands))

for i, src in enumerate(sources):
    offset = (i - 1) * width  # center the bars
    bars = ax.bar(x + offset, np2_vals[src], width=width, label=src, color=colors[i], alpha=0.7, edgecolor='black')

    # Annotate bars with η² values and significance stars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,  # a little above the bar
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=9
            )
            # Add significance stars if p-FDR < 0.05
            star_text = pval_to_stars(pvals[src][j])
            if star_text:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.07,  # above the eta squared text
                    star_text,
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    color='black',
                    fontweight='bold'
                )

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.set_ylabel('Effect Size (η²)')
ax.set_title('Effect Sizes by Band and ANOVA Source')
ax.set_ylim(0, 1.0)
ax.legend(title='ANOVA Source')
plt.tight_layout()
plt.savefig("anova_effect_sizes_grouped_barplot_with_significance.png", dpi=300)
plt.show()

# ----------
# Calculating the Area Under Curve
# ---------

def compute_auc(y, x=None):
    """
    Compute area under curve using trapezoidal integration.
    y: power values (1D array)
    x: corresponding x-axis values (time or freq), optional (assumes uniform spacing if None)
    """
    return np.trapezoid(y, x=x)

# Entire period AUC
def auc_entire_period(spectrogram, freqs, times, bands=None):
    """
    Compute AUC over the entire period, for each subject.
    Returns a dict: band_name -> list of AUC values (one per subject)
    If bands is None, returns AUC for whole spectrogram.
    """
    n_subjects = spectrogram.shape[0]
    results = {}

    if bands is None:
        results['full_spectrogram'] = []
        for subj in range(n_subjects):
            total_power = spectrogram[subj].sum(axis=0)  # sum over freq axis
            auc_val = compute_auc(total_power, times)
            results['full_spectrogram'].append(auc_val)
    else:
        for band_name, band_range in bands.items():
            idx = band_indices(freqs, band_range)
            results[band_name] = []
            for subj in range(n_subjects):
                band_power = spectrogram[subj][idx, :].sum(axis=0)
                auc_val = compute_auc(band_power, times)
                results[band_name].append(auc_val)
    return results
def test_auc_entire_between_groups(sleep_auc, propofol_auc):
    """
    For each frequency band, test difference in AUC between sleep and propofol groups.
    Uses Mann-Whitney U test for all bands (non-parametric).

    Returns a pandas DataFrame summarizing results.
    """
    bands = list(sleep_auc.keys())
    results = []

    for band in bands:
        sleep_data = np.array(sleep_auc[band])
        prop_data = np.array(propofol_auc[band])

        # Mann-Whitney U test (two-sided)
        stat, pval = mannwhitneyu(sleep_data, prop_data, alternative='two-sided')
        test_name = 'Mann-Whitney U'

        # Collect results (normality p-values omitted)
        results.append({
            'Band': band,
            'Test': test_name,
            'Statistic': stat,
            'p-value': pval,
            'Sleep Mean AUC': np.mean(sleep_data),
            'Propofol Mean AUC': np.mean(prop_data)
        })

    return pd.DataFrame(results)
def plot_entire_auc(sleep_auc, propofol_auc, test_results):
    bands = list(sleep_auc.keys())
    sleep_means = [np.mean(sleep_auc[band]) for band in bands]
    sleep_stds = [np.std(sleep_auc[band]) for band in bands]
    prop_means = [np.mean(propofol_auc[band]) for band in bands]
    prop_stds = [np.std(propofol_auc[band]) for band in bands]

    x = np.arange(len(bands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_sleep = ax.bar(x - width / 2, sleep_means, width, yerr=sleep_stds,
                        label='Sleep', capsize=5,
                        color='blue', alpha=0.6,
                        edgecolor='black', linewidth=1)
    bars_prop = ax.bar(x + width / 2, prop_means, width, yerr=prop_stds,
                       label='Propofol', capsize=5,
                       color='red', alpha=0.6,
                       edgecolor='black', linewidth=1)

    # Horizontal line at 0 (lighter)
    ax.axhline(0, linestyle='--', color='gray', linewidth=0.8, alpha=0.7)

    max_y = max([m + s for m, s in zip(sleep_means, sleep_stds)] +
                [m + s for m, s in zip(prop_means, prop_stds)])
    min_y = min([m - s for m, s in zip(sleep_means, sleep_stds)] +
                [m - s for m, s in zip(prop_means, prop_stds)])

    y_lim_bottom = -200000
    y_lim_top = max_y * 1.4  # 40% padding

    ax.set_ylim(bottom=y_lim_bottom, top=y_lim_top)

    total_range = y_lim_top - y_lim_bottom  # full y-axis range

    for i, band in enumerate(bands):
        pval = test_results.loc[test_results['Band'] == band, 'p-value'].values[0]
        if pval < 0.05:
            stars = pval_to_stars(pval)
            y_max = max(sleep_means[i] + sleep_stds[i], prop_means[i] + prop_stds[i])

            # Base height: 30% padding above max_y
            base_height = y_max + (y_lim_top - max_y) * 0.3

            # Extra push for gamma band: add additional offset to avoid overlap with zero line
            if band.lower() == 'gamma':
                base_height += total_range * 0.15  # push up by 10% of total y range

            x1 = x[i] - width / 2
            x2 = x[i] + width / 2
            h = (y_lim_top - max_y) * 0.05

            ax.plot([x1, x2], [base_height, base_height], color='black', linewidth=1.5)
            ax.plot([x1, x1], [base_height, base_height - h], color='black', linewidth=1.5)
            ax.plot([x2, x2], [base_height, base_height - h], color='black', linewidth=1.5)

            ax.text(x[i], base_height + h * 1.2, stars, ha='center', va='bottom', fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel('AUC')
    ax.set_title('AUC Across Entire Period by Band')

    sleep_patch = mpatches.Patch(color='blue', alpha=0.6, label='Sleep', edgecolor='black')
    prop_patch = mpatches.Patch(color='red', alpha=0.6, label='Propofol', edgecolor='black')
    ax.legend(handles=[sleep_patch, prop_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig("auc_entire_period.png", dpi=300)
    plt.show()


def auc_pre_post_loc(spectrogram, freqs, times, loc_time, bands=None):
    """
    Compute AUC pre- and post-LoC for each subject.
    Returns dict with keys 'pre_LoC' and 'post_LoC', each containing dicts of band_name -> list of AUCs.
    """
    n_subjects = spectrogram.shape[0]
    pre_mask = times < loc_time
    post_mask = times >= loc_time

    results = {'pre_LoC': {}, 'post_LoC': {}}

    if bands is None:
        results['pre_LoC']['full_spectrogram'] = []
        results['post_LoC']['full_spectrogram'] = []
        for subj in range(n_subjects):
            total_power = spectrogram[subj].sum(axis=0)
            results['pre_LoC']['full_spectrogram'].append(compute_auc(total_power[pre_mask], times[pre_mask]))
            results['post_LoC']['full_spectrogram'].append(compute_auc(total_power[post_mask], times[post_mask]))
    else:
        for band_name, band_range in bands.items():
            idx = band_indices(freqs, band_range)
            results['pre_LoC'][band_name] = []
            results['post_LoC'][band_name] = []
            for subj in range(n_subjects):
                band_power = spectrogram[subj][idx, :].sum(axis=0)
                results['pre_LoC'][band_name].append(compute_auc(band_power[pre_mask], times[pre_mask]))
                results['post_LoC'][band_name].append(compute_auc(band_power[post_mask], times[post_mask]))

    return results
def test_auc_within_groups(auc_sleep, auc_propofol):
    """
    For each band, test if pre vs post LoC AUC differs within each group (Sleep and Propofol).
    Uses Wilcoxon signed-rank test (paired, non-parametric).

    Returns a DataFrame with columns:
    Group, Band, Test, Statistic, p-value, Pre Mean, Post Mean
    """
    groups = {'Sleep': auc_sleep, 'Propofol': auc_propofol}
    bands = list(auc_sleep['pre_LoC'].keys())
    results = []

    for group_name, group_data in groups.items():
        for band in bands:
            pre_data = np.array(group_data['pre_LoC'][band])
            post_data = np.array(group_data['post_LoC'][band])

            # Wilcoxon signed-rank test, paired samples
            try:
                stat, pval = wilcoxon(pre_data, post_data)
            except ValueError:
                # Wilcoxon may fail if data are constant; skip or assign NaN
                stat, pval = np.nan, np.nan

            results.append({
                'Group': group_name,
                'Band': band,
                'Test': 'Wilcoxon signed-rank',
                'Statistic': stat,
                'p-value': pval,
                'Pre Mean AUC': np.mean(pre_data),
                'Post Mean AUC': np.mean(post_data),
                'Pre Median AUC': np.median(pre_data),
                'Post Median AUC': np.median(post_data)
            })

    return pd.DataFrame(results)
def test_auc_between_groups(auc_sleep, auc_propofol):
    """
    Test whether the change in AUC (post - pre) differs between Sleep and Propofol groups.
    Uses Mann-Whitney U test for each band.

    Returns a DataFrame with columns:
    Band, Test, Statistic, p-value, Sleep ∆AUC Mean, Propofol ∆AUC Mean
    """
    bands = list(auc_sleep['pre_LoC'].keys())
    results = []

    for band in bands:
        # Compute ∆AUC per subject
        sleep_delta = np.array(auc_sleep['post_LoC'][band]) - np.array(auc_sleep['pre_LoC'][band])
        prop_delta = np.array(auc_propofol['post_LoC'][band]) - np.array(auc_propofol['pre_LoC'][band])

        try:
            stat, pval = mannwhitneyu(sleep_delta, prop_delta, alternative='two-sided')
        except ValueError:
            stat, pval = np.nan, np.nan

        results.append({
            'Band': band,
            'Test': 'Mann-Whitney U (∆AUC)',
            'Statistic': stat,
            'p-value': pval,
            'Sleep ∆ AUC Mean': np.mean(sleep_delta),
            'Propofol ∆ AUC Mean': np.mean(prop_delta),
            'Sleep ∆ AUC Median': np.median(sleep_delta),
            'Propofol ∆ AUC Median': np.median(prop_delta)
        })

    df = pd.DataFrame(results)
    print(df)
    return df
def plot_pre_post_auc_with_stats(auc_sleep, auc_propofol, within_results, alpha=0.05):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd

    bands = list(auc_sleep['pre_LoC'].keys())
    x = np.arange(len(bands))
    width = 0.2
    hatch_pattern = '////'

    def get_vals(group, period):
        return [np.mean(group[period][band]) for band in bands], [np.std(group[period][band]) for band in bands]

    sleep_pre_mean, sleep_pre_std = get_vals(auc_sleep, 'pre_LoC')
    sleep_post_mean, sleep_post_std = get_vals(auc_sleep, 'post_LoC')
    prop_pre_mean, prop_pre_std = get_vals(auc_propofol, 'pre_LoC')
    prop_post_mean, prop_post_std = get_vals(auc_propofol, 'post_LoC')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars
    ax.bar(x - 1.5 * width, sleep_pre_mean, width, yerr=sleep_pre_std,
           label='Sleep Pre-LoC', color='blue', alpha=0.5, edgecolor='black', capsize=5)
    ax.bar(x - 0.5 * width, sleep_post_mean, width, yerr=sleep_post_std,
           label='Sleep Post-LoC', color='none', edgecolor='blue', hatch=hatch_pattern, capsize=5)
    ax.bar(x + 0.5 * width, prop_pre_mean, width, yerr=prop_pre_std,
           label='Propofol Pre-LoC', color='red', alpha=0.5, edgecolor='black', capsize=5)
    ax.bar(x + 1.5 * width, prop_post_mean, width, yerr=prop_post_std,
           label='Propofol Post-LoC', color='none', edgecolor='red', hatch=hatch_pattern, capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel('AUC')
    ax.set_title('Pre vs Post LoC AUC by Band and Condition')
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)

    # Custom legend
    solid_patch_blue = mpatches.Patch(facecolor='blue', edgecolor='black', alpha=0.5, label='Sleep Pre-LoC')
    striped_patch_blue = mpatches.Patch(facecolor='white', edgecolor='blue', hatch=hatch_pattern, label='Sleep Post-LoC')
    solid_patch_red = mpatches.Patch(facecolor='red', edgecolor='black', alpha=0.5, label='Propofol Pre-LoC')
    striped_patch_red = mpatches.Patch(facecolor='white', edgecolor='red', hatch=hatch_pattern, label='Propofol Post-LoC')
    ax.legend(handles=[solid_patch_blue, striped_patch_blue, solid_patch_red, striped_patch_red])

    # Helper function to get bar top (top edge regardless of sign)
    def bar_top(mean, std):
        return mean + std  # always add std for top edge so significance bars go above

    # Convert p-values to numeric safely
    within_results['p-value'] = pd.to_numeric(within_results['p-value'], errors='coerce')

    # Calculate max bar heights (including error bars) per group and period
    all_means = sleep_pre_mean + sleep_post_mean + prop_pre_mean + prop_post_mean
    all_stds = sleep_pre_std + sleep_post_std + prop_pre_std + prop_post_std
    max_bar_height = max([m + s for m, s in zip(all_means, all_stds)])
    min_bar_height = min([m - s for m, s in zip(all_means, all_stds)])

    # Set initial y-limits with padding
    y_lim_bottom = min(-120000, min_bar_height * 1.1)  # as per your example or more negative
    y_lim_top = max_bar_height * 1.75  # 40% padding above max

    ax.set_ylim(bottom=y_lim_bottom, top=y_lim_top)

    total_range = y_lim_top - y_lim_bottom

    for i, band in enumerate(bands):
        for group_name, shift, pre, post, pre_std, post_std, color in [
            ('Sleep', -1, sleep_pre_mean, sleep_post_mean, sleep_pre_std, sleep_post_std, 'blue'),
            ('Propofol', +1, prop_pre_mean, prop_post_mean, prop_pre_std, prop_post_std, 'red')
        ]:
            row = within_results[(within_results['Group'] == group_name) & (within_results['Band'] == band)]
            if not row.empty:
                try:
                    pval_raw = row['p-value'].iat[0]
                    pval = float(pval_raw)
                except (IndexError, ValueError, TypeError):
                    pval = None

                if pval is not None and not pd.isna(pval) and pval < alpha:
                    stars = pval_to_stars(pval)

                    # X positions for the two bars (pre and post)
                    x1 = x[i] + (shift - 0.5) * width
                    x2 = x[i] + (shift + 0.5) * width

                    # Max height of the two bars (including error bars)
                    y_max = max(pre[i] + pre_std[i], post[i] + post_std[i])

                    # Base height: 30% of padding above max bar height
                    base_height = y_max + (y_lim_top - max_bar_height) * 0.3

                    # Adjust vertical position for Propofol in high beta and gamma
                    if group_name == 'Propofol' and band.lower() in ['high beta']:
                        base_height += total_range * 0.0001
                    elif group_name == 'Propofol' and band.lower() in ['gamma']:
                        base_height += total_range * 0.045
                    else:
                        # Push gamma and high beta bars up for other groups (or default)
                        if band.lower() == 'gamma':
                            base_height += total_range * 0.15
                        elif band.lower() == 'high beta':
                            base_height += total_range * 0.05



                    h = (y_lim_top - max_bar_height) * 0.1  # height of caps

                    # Draw horizontal significance line
                    ax.plot([x1, x2], [base_height, base_height], color='black', linewidth=1.5)
                    # Draw vertical caps
                    ax.plot([x1, x1], [base_height, base_height - h], color='black', linewidth=1.5)
                    ax.plot([x2, x2], [base_height, base_height - h], color='black', linewidth=1.5)
                    # Add stars
                    ax.text((x1 + x2) / 2, base_height + h * 1.2, stars, ha='center', va='bottom', fontsize=16)

    plt.tight_layout()
    plt.savefig('pre_post_AUC_new.png', dpi=300)
    plt.show()








def auc_sliding_window(spectrogram, freqs, times, bands=None, window_size=10, step_size=1):
    """
    Compute sliding window AUC for each subject.
    Returns dict: band_name -> list of lists (subjects x windows)
    """
    n_subjects = spectrogram.shape[0]

    # ✅ Check for alignment between time axis and spectrogram's epoch dimension
    assert spectrogram.shape[2] == len(times), "Mismatch: time axis does not match number of epochs"

    results = {}

    # Generate window start times
    start_times = np.arange(times[0], times[-1] - window_size + step_size, step_size)

    if bands is None:
        results['full_spectrogram'] = []
        for subj in range(n_subjects):
            subj_aucs = []
            total_power = spectrogram[subj].sum(axis=0)
            for start in start_times:
                mask = (times >= start) & (times < start + window_size)
                if np.sum(mask) < 2:
                    subj_aucs.append(np.nan)  # or continue
                    print(f"{start:.1f}–{start + window_size:.1f}: {mask.sum()} timepoints")
                    continue
                auc_val = compute_auc(total_power[mask], times[mask])
                subj_aucs.append(auc_val)
            results['full_spectrogram'].append(subj_aucs)
    else:
        for band_name, band_range in bands.items():
            idx = band_indices(freqs, band_range)
            results[band_name] = []
            for subj in range(n_subjects):
                subj_aucs = []
                band_power = spectrogram[subj][idx, :].sum(axis=0)
                for start in start_times:
                    mask = (times >= start) & (times < start + window_size)
                    if np.sum(mask) < 2:
                        subj_aucs.append(np.nan)  # or continue
                        continue
                    auc_val = compute_auc(band_power[mask], times[mask])
                    subj_aucs.append(auc_val)
                results[band_name].append(subj_aucs)

    return results

def summarize_auc_results(sleep_auc, propofol_auc):
    import numpy as np

    print(f"{'Band':<10} | {'Condition':<10} | {'Mean AUC':>12} | {'Std Dev':>10}")
    print("-" * 55)

    for band in sleep_auc.keys():
        # Get arrays
        sleep_vals = np.array(sleep_auc[band], dtype=np.float64)
        prop_vals = np.array(propofol_auc[band], dtype=np.float64)

        # Compute stats
        sleep_mean, sleep_std = np.mean(sleep_vals), np.std(sleep_vals)
        prop_mean, prop_std = np.mean(prop_vals), np.std(prop_vals)

        # Print
        print(f"{band:<10} | {'Sleep':<10} | {sleep_mean:12.2f} | {sleep_std:10.2f}")
        print(f"{'':<10} | {'Propofol':<10} | {prop_mean:12.2f} | {prop_std:10.2f}")
        print("-" * 55)

def print_pre_post_auc(auc_sleep, auc_propofol):
    print("MEDIANS NOW CALCULATED INSTEAD")
    print(f"{'Band':<10} | {'Group':<10} | {'Period':<8} | {'Mean AUC':>10} | {'Std Dev':>10}")
    print("-" * 60)

    for band in auc_sleep['pre_LoC'].keys():
        # Sleep group
        for period in ['pre_LoC', 'post_LoC']:
            auc_vals = np.array(auc_sleep[period][band], dtype=np.float64)
            print(f"{band:<10} | {'Sleep':<10} | {period:<8} | {np.median(auc_vals):10.2f} | {np.std(auc_vals):10.2f}")
        # Propofol group
        for period in ['pre_LoC', 'post_LoC']:
            auc_vals = np.array(auc_propofol[period][band], dtype=np.float64)
            print(f"{band:<10} | {'Propofol':<10} | {period:<8} | {np.median(auc_vals):10.2f} | {np.std(auc_vals):10.2f}")
        print("-" * 60)
def print_sliding_auc(auc_sleep_sliding, auc_propofol_sliding):
    print(f"{'Band':<10} | {'Group':<10} | {'Mean AUC':>10} | {'Std Dev':>10} | {'Num Windows':>12}")
    print("-" * 65)
    for band in auc_sleep_sliding.keys():
        # Sleep group
        sleep_vals = np.array(auc_sleep_sliding[band])  # shape: (subjects, windows)
        mean_vals = sleep_vals.mean(axis=1)  # average across windows for each subject
        print(f"{band:<10} | {'Sleep':<10} | {mean_vals.mean():10.2f} | {mean_vals.std():10.2f} | {sleep_vals.shape[1]:12}")
        # Propofol group
        prop_vals = np.array(auc_propofol_sliding[band])
        mean_vals = prop_vals.mean(axis=1)
        print(f"{band:<10} | {'Propofol':<10} | {mean_vals.mean():10.2f} | {mean_vals.std():10.2f} | {prop_vals.shape[1]:12}")
        print("-" * 65)



# WHOLE SPECTROGRAM
# Compute AUC for sleep group
sleep_auc_full = auc_entire_period(sleep_group, frequencies, time, bands=freq_bands)
# Compute AUC for propofol group
propofol_auc_full = auc_entire_period(propofol_group, frequencies, time, bands=freq_bands)
auc_test_results = test_auc_entire_between_groups(sleep_auc_full, propofol_auc_full)
print("AUC ENTIRE PERIOD TEST RESULTS")
print(auc_test_results)
test_results = test_auc_entire_between_groups(sleep_auc_full, propofol_auc_full)
plot_entire_auc(sleep_auc_full, propofol_auc_full, test_results)


# Example: Print delta band AUC
summarize_auc_results(sleep_auc_full, propofol_auc_full)


# Calculate pre/post LoC AUCs
auc_sleep_prepost = auc_pre_post_loc(sleep_group, frequencies, time, loc_time=0, bands=freq_bands)
auc_propofol_prepost = auc_pre_post_loc(propofol_group, frequencies, time, loc_time=0, bands=freq_bands)
print_pre_post_auc(auc_sleep_prepost, auc_propofol_prepost)
within_df = test_auc_within_groups(auc_sleep_prepost, auc_propofol_prepost)
between_df = test_auc_between_groups(auc_sleep_prepost, auc_propofol_prepost)


print ("PRE POST LOC STATS")
print(within_df)
print("PRE POST LOC STATS BETWEEN GROUPS")
print(between_df)
# Plot
plot_pre_post_auc_with_stats(auc_sleep_prepost, auc_propofol_prepost, within_df)


window_size = 5  # seconds
step_size = 1   # seconds

auc_sleep_sliding = auc_sliding_window(sleep_group, frequencies, time, bands=freq_bands,
                                       window_size=window_size, step_size=step_size)
auc_propofol_sliding = auc_sliding_window(propofol_group, frequencies, time, bands=freq_bands,
                                          window_size=window_size, step_size=step_size)
print_sliding_auc(auc_sleep_sliding, auc_propofol_sliding)






def plot_sliding_auc(auc_sleep_sliding, auc_propofol_sliding, times, window_size, step_size):
    bands = list(auc_sleep_sliding.keys())
    start_times = np.arange(times[0], times[-1] - window_size + step_size, step_size)

    for band in bands:
        sleep_vals = np.array(auc_sleep_sliding[band])  # (subjects, windows)
        prop_vals = np.array(auc_propofol_sliding[band])

        sleep_mean = sleep_vals.mean(axis=0)
        sleep_std = sleep_vals.std(axis=0)
        prop_mean = prop_vals.mean(axis=0)
        prop_std = prop_vals.std(axis=0)

        plt.figure(figsize=(12, 5))
        plt.plot(start_times, sleep_mean, label='Sleep', color='blue')
        plt.fill_between(start_times, sleep_mean - sleep_std, sleep_mean + sleep_std, alpha=0.3, color='blue')
        plt.plot(start_times, prop_mean, label='Propofol', color='red')
        plt.fill_between(start_times, prop_mean - prop_std, prop_mean + prop_std, alpha=0.3, color='red')

        plt.title(f'Sliding Window AUC: {band}')
        plt.xlabel('Time (s)')
        plt.ylabel('AUC')
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_individual_sliding_auc(auc_data, start_times, band, group_name):
    plt.figure(figsize=(12, 5))
    for subj_vals in auc_data[band]:
        plt.plot(start_times, subj_vals, alpha=0.3)
    plt.title(f'Individual Sliding AUC: {band} - {group_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('AUC')
    plt.tight_layout()
    plt.show()

start_times = np.arange(time[0], time[-1] - 5 + 1, 1)

"""plot_entire_auc(sleep_auc_full, propofol_auc_full)
plot_pre_post_auc(auc_sleep_prepost, auc_propofol_prepost)"""
plot_sliding_auc(auc_sleep_sliding, auc_propofol_sliding, time, window_size, step_size)
"""plot_individual_sliding_auc(auc_sleep_sliding, start_times=start_times, band='Alpha', group_name='Sleep')
plot_individual_sliding_auc(auc_propofol_sliding, start_times=start_times,band='Alpha', group_name='Propofol')"""





def stats_on_sliding_auc(sleep_sliding, prop_sliding, start_times, alpha=0.05):
    """
    Perform pointwise paired t-tests between Sleep and Propofol AUC across sliding windows.
    Mark clusters of consecutive significant windows.
    """
    sleep_arr = np.array(sleep_sliding)  # shape: (subjects, n_windows)
    prop_arr = np.array(prop_sliding)
    n_windows = sleep_arr.shape[1]

    pvals = np.ones(n_windows)
    tstats = np.zeros(n_windows)
    for w in range(n_windows):
        t, p = ttest_rel(sleep_arr[:, w], prop_arr[:, w])
        tstats[w], pvals[w] = t, p

    # Simple cluster detection
    sig = pvals < alpha
    clusters = []
    current = None
    for i, s in enumerate(sig):
        if s and current is None:
            current = [i, i]
        elif not s and current is not None:
            clusters.append(tuple(current))
            current = None
    if current is not None:
        clusters.append(tuple(current))

    return tstats, pvals, clusters

# Use this to run stats:
"""for band in freq_bands:
    print(f"\n--- Stats for Sliding AUC: {band} ---")
    tstats, pvals, clusters = stats_on_sliding_auc(
        auc_sleep_sliding[band],
        auc_propofol_sliding[band],
        start_times
    )
    print("Significant clusters of windows (indices):", clusters)

    # Plot with markings
    plt.figure(figsize=(10,4))
    mean_s = np.mean(auc_sleep_sliding[band], axis=0)
    mean_p = np.mean(auc_propofol_sliding[band], axis=0)
    plt.plot(start_times, mean_s, label='Sleep', color='blue')
    plt.plot(start_times, mean_p, label='Propofol', color='red')
    for (a, b) in clusters:
        plt.axvspan(start_times[a], start_times[b], color='orange', alpha=0.9)
    plt.title(f'{band} Sliding AUC (clusters marked)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.show()"""




def auc_slope_over_time(auc_vals, times):
    slopes = []
    for subj_curve in auc_vals:
        mask = ~np.isnan(subj_curve)
        if mask.sum() > 1:
            slope, *_ = linregress(times[mask], subj_curve[mask])
            slopes.append(slope)
        else:
            slopes.append(np.nan)
    return np.array(slopes)

"""sleep_slopes = auc_slope_over_time(np.array(auc_sleep_sliding['Alpha']), start_times)
prop_slopes = auc_slope_over_time(np.array(auc_propofol_sliding['Alpha']), start_times)"""

# -----------------------------
# 5. Alternative window sizes & step sizes
# -----------------------------
"""for ws in [3, 5, 7]:             # test window lengths
    for ss in [0.5, 1, 2]:        # test step sizes
        print(f"\nWindow {ws}s, Step {ss}s")
        soa = auc_sliding_window(sleep_group, frequencies, times, bands=freq_bands,
                                  window_size=ws, step_size=ss)
        poa = auc_sliding_window(propofol_group, frequencies, times, bands=freq_bands,
                                  window_size=ws, step_size=ss)
        # Compute and display mean trajectories
        for band in freq_bands:
            mean_s = np.nanmean(soa[band], axis=0)
            mean_p = np.nanmean(poa[band], axis=0)
            gap_size = np.nanmax(mean_p - mean_s) - np.nanmin(mean_p - mean_s)
            print(f"{band}: max gap ≈ {gap_size:.1f}")"""

# -----------------------------
# 6. Cluster-permutation on sliding AUC
# -----------------------------
from mne.stats import permutation_cluster_test
"""for band in freq_bands:
    print(f"\n=== Permutation Cluster Test on {band} Sliding AUC ===")
    data = [np.array(auc_sleep_sliding[band]), np.array(auc_propofol_sliding[band])]
    T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
        data, n_permutations=1000, tail=0, threshold=None, seed=0)
    print("Clusters & p-values:", list(zip(clusters, cluster_p_values)))"""

def test_sliding_auc_clusters(
    sleep_sliding, propofol_sliding, start_times, loc_time,
    window=5, alpha=0.05, band='Alpha', plot=True
):
    """
    Tests for significant differences in AUC between Sleep and Propofol in sliding windows.
    Focuses on the region around LoC.

    Parameters:
    - sleep_sliding, propofol_sliding: dict of {band: list of subject x timepoint AUCs}
    - start_times: list of sliding window start times
    - loc_time: time of Loss of Consciousness (s)
    - window: size of analysis window around LoC (s)
    - alpha: significance threshold
    - band: frequency band to test
    - plot: whether to plot the results

    Returns:
    - tvals: t-values at each window
    - pvals: p-values at each window
    - clusters: list of (start_idx, end_idx) for significant clusters
    """

    # Extract arrays: shape (n_subjects, n_windows)
    sleep_data = np.array(sleep_sliding[band])
    prop_data = np.array(propofol_sliding[band])
    n_windows = sleep_data.shape[1]

    # Restrict to region around LoC
    mask = (start_times >= loc_time - window/2) & (start_times <= loc_time + window/2)
    masked_times = start_times[mask]
    sleep_masked = sleep_data[:, mask]
    prop_masked = prop_data[:, mask]

    # Perform independent t-tests (or ttest_rel if within-subjects)
    tvals, pvals = [], []
    for i in range(len(masked_times)):
        t, p = ttest_ind(sleep_masked[:, i], prop_masked[:, i])
        tvals.append(t)
        pvals.append(p)
    tvals = np.array(tvals)
    pvals = np.array(pvals)

    # Find significant clusters
    sig = pvals < alpha
    clusters = []
    current = None
    for i, s in enumerate(sig):
        if s and current is None:
            current = [i, i]
        elif s:
            current[1] = i
        elif not s and current is not None:
            clusters.append((current[0], current[1]))
            current = None
    if current:
        clusters.append((current[0], current[1]))

    # Optional plot
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(masked_times, tvals, label='t-values', color='black')
        plt.axhline(y=0, color='gray', linestyle='--')
        for cluster in clusters:
            plt.axvspan(masked_times[cluster[0]], masked_times[cluster[1]], color='orange', alpha=0.3)
        plt.title(f"T-test Results Around LoC ({band})")
        plt.xlabel("Time (s)")
        plt.ylabel("T-value")
        plt.tight_layout()
        plt.show()

    return tvals, pvals, clusters

"""tvals, pvals, clusters = test_sliding_auc_clusters(
    sleep_sliding=auc_sleep_sliding,
    propofol_sliding=auc_propofol_sliding,
    start_times=start_times,
    loc_time=mid_epoch,  # LoC time
    window=10,           # Analyze ±5 sec around LoC
    band='Alpha',
    plot=True
)"""

"""print(tvals)
print(pvals)
print(clusters)
"""

"""print("SANITY CHECKS")
np.isnan(auc_sleep_sliding['Alpha']).sum()
print("Start times:", len(start_times))
print("Sleep AUC windows:", np.array(auc_sleep_sliding['Alpha']).shape)"""


"""for i, subj_auc in enumerate(auc_sleep_sliding['Delta']):
    plt.plot(subj_auc, label=f"Subject {i+1}")
plt.title("Sliding Window AUC – Sleep – Delta Band")
plt.xlabel("Window index")
plt.ylabel("AUC")
plt.legend()
plt.show()"""


def auc_per_epoch_all_freqs(spectrogram, freqs):
    """
    Compute total power (AUC) per epoch across all frequencies.
    Returns a list of arrays (one per subject) with AUC per epoch.
    """
    n_subjects = spectrogram.shape[0]
    epoch_auc_per_subject = []

    for subj in range(n_subjects):
        # shape: (n_freqs, n_epochs)
        subj_spec = spectrogram[subj]
        aucs = []
        for epoch in range(subj_spec.shape[1]):
            power_spectrum = subj_spec[:, epoch]  # shape: (n_freqs,)
            auc = compute_auc(power_spectrum, x=freqs)
            aucs.append(auc)
        epoch_auc_per_subject.append(np.array(aucs))

    return epoch_auc_per_subject

sleep_epoch_auc = auc_per_epoch_all_freqs(sleep_group, frequencies)
propofol_epoch_auc = auc_per_epoch_all_freqs(propofol_group, frequencies)

# Example: print AUCs for subject 0
print("Sleep subject 0 epoch AUCs:", sleep_epoch_auc[0])

# --- Convert to arrays: shape (n_subjects, n_epochs) ---
sleep_auc_array = np.stack(sleep_epoch_auc)   # shape: (n_subjects, n_epochs)
prop_auc_array = np.stack(propofol_epoch_auc)

# --- Compute mean & std across subjects per epoch ---
sleep_mean = np.mean(sleep_auc_array, axis=0)
sleep_std = np.std(sleep_auc_array, axis=0)
prop_mean = np.mean(prop_auc_array, axis=0)
prop_std = np.std(prop_auc_array, axis=0)

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.plot(time, sleep_mean, label='Sleep', color='blue')
plt.fill_between(time, sleep_mean - sleep_std, sleep_mean + sleep_std, alpha=0.3, color='blue')

plt.plot(time, prop_mean, label='Propofol', color='red')
plt.fill_between(time, prop_mean - prop_std, prop_mean + prop_std, alpha=0.3, color='red')

plt.title('Total Power per Epoch (All Frequencies)')
plt.xlabel('Time (s)')
plt.ylabel('Total Power (AUC) per Epoch')
plt.legend()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axvline(0, color='black', linestyle='--', linewidth=1)  # LoC marker
plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Baseline power

plt.xticks(np.arange(time[0], 241, 60))  # ticks every 60 seconds
plt.xlim(time[0], time[-1])

plt.savefig('total_power_over_time.png', dpi=300)
plt.tight_layout()
plt.show()


# --- Compute median & std across subjects per epoch ---
sleep_median = np.median(sleep_auc_array, axis=0)
sleep_std = np.std(sleep_auc_array, axis=0)
prop_median = np.median(prop_auc_array, axis=0)
prop_std = np.std(prop_auc_array, axis=0)

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.plot(time, sleep_median, label='Sleep', color='blue')
plt.fill_between(time, sleep_median - sleep_std, sleep_median + sleep_std, alpha=0.3, color='blue')

plt.plot(time, prop_median, label='Propofol', color='red')
plt.fill_between(time, prop_median - prop_std, prop_median + prop_std, alpha=0.3, color='red')

plt.title('Total median AUC per Epoch (All Frequencies)')
plt.xlabel('Time (s)')
plt.ylabel('Total Power (AUC)')
plt.legend()
plt.tight_layout()
plt.show()

sleep_slopes = [linregress(time, subj_auc).slope for subj_auc in sleep_auc_array]
prop_slopes = [linregress(time, subj_auc).slope for subj_auc in prop_auc_array]

# Compare average slopes
print("Mean slope (Sleep):", np.mean(sleep_slopes))
print("Mean slope (Propofol):", np.mean(prop_slopes))

# Optional: t-test on slopes

print("TOTAL POWER SLOPES")

print("Sleep group:")
stat1, p1 = shapiro(sleep_slopes)
print(f"Shapiro-Wilk: W={stat1:.3f}, p={p1:.4f}")

print("\nPropofol group:")
stat2, p2 = shapiro(prop_slopes)
print(f"Shapiro-Wilk: W={stat2:.3f}, p={p2:.4f}")


# Mann-Whitney U test
u_stat, p_val = mannwhitneyu(sleep_slopes, prop_slopes, alternative='two-sided')
sig_label = '*' if p_val < 0.05 else 'n.s.'
print("MANN WHITNEY U TEST")
print(f"P-value is {p_val}")
print(f"U-value is {u_stat}")

plt.figure(figsize=(8, 6))

boxprops = dict(linewidth=2)
medianprops = dict(color='black', linewidth=2)


# Draw boxplots
bp = plt.boxplot([sleep_slopes, prop_slopes],
                 labels=['Sleep', 'Propofol'],
                 patch_artist=True,
                 boxprops=boxprops,
                 medianprops=medianprops,
                 )

# Colors for boxes
colors = ['blue', 'red']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

y_min = min(min(sleep_slopes), min(prop_slopes))
y_max = max(max(sleep_slopes), max(prop_slopes))
y_range = y_max - y_min

# Expand both limits by 10% of the data range
plt.ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)

# Add significance bar
x1, x2 = 1, 2
bar_height = y_max + 0.1 * y_range
bar_thickness = 0.02 * y_range
plt.plot([x1, x1, x2, x2],
         [bar_height, bar_height + bar_thickness, bar_height + bar_thickness, bar_height],
         lw=1.5, c='k')

# Place the asterisk slightly lower to avoid overlap
plt.text((x1 + x2) * 0.5, bar_height + bar_thickness + 0.01 * y_range, '**',
         ha='center', va='bottom', color='k', fontsize=20)


plt.ylabel('Slope of Total Power Over Time')
plt.title('Total Power Slope per Subject', y=1.07)

plt.axhline(0, linestyle='--', color='gray')


plt.tight_layout(pad=3)  # Add padding so nothing overlaps the title
plt.savefig('total_power_slope_boxplot.png', dpi=300)
plt.show()


# Bandwidth - not meaningful
"""# ---------
# Bandwidth
# -------
sleep_bandwidth_results = {}
propofol_bandwidth_results = {}

# Loop over each frequency band
for band_name in freq_bands.keys():
    print(f"\n=== Processing {band_name} Band ===")


    sleep_bandwidth_results = {}


    def compute_fwhm(freqs, power_spectrum):
        peak_idx = np.argmax(power_spectrum)
        peak_freq = freqs[peak_idx]
        peak_power = power_spectrum[peak_idx]
        half_max = peak_power / 2

        # Find left crossing point (below half max before peak)
        left_crossings = np.where(power_spectrum[:peak_idx] < half_max)[0]
        if len(left_crossings) == 0:
            left_freq = freqs[0]  # No crossing, use start of band
        else:
            left_freq = freqs[left_crossings[-1]]

        # Find right crossing point (below half max after peak)
        right_crossings = np.where(power_spectrum[peak_idx:] < half_max)[0]
        if len(right_crossings) == 0:
            right_freq = freqs[-1]  # No crossing, use end of band
        else:
            right_freq = freqs[peak_idx + right_crossings[0]]

        fwhm = right_freq - left_freq
        if fwhm < 0:
            # This shouldn’t happen, but if it does, fallback to nan
            return np.nan
        return fwhm


    def plot_spectrum_with_fwhm(freqs, power, bandwidth, title=""):
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]
        peak_power = power[peak_idx]
        half_max = peak_power / 2

        plt.plot(freqs, power, label="Power Spectrum")
        plt.axvline(peak_freq, color='r', linestyle='--', label='Peak Frequency')
        plt.axhline(half_max, color='g', linestyle='--', label='Half Max')

        # Mark FWHM range if valid
        if bandwidth > 0:
            left_idx = np.where(power[:peak_idx] < half_max)[0]
            right_idx = np.where(power[peak_idx:] < half_max)[0]
            if len(left_idx) > 0 and len(right_idx) > 0:
                fwhm_start = freqs[left_idx[-1]]
                fwhm_end = freqs[peak_idx + right_idx[0]]
                plt.axvspan(fwhm_start, fwhm_end, color='y', alpha=0.3, label='FWHM Bandwidth')

        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.legend()
        plt.show()


    f_low, f_high = freq_bands[band_name]  # e.g., (0.5, 4) for Delta
    # Mask to select only frequencies inside current band
    band_mask = (frequencies >= f_low) & (frequencies <= f_high)

    sleep_bandwidths = []  # To collect FWHM per subject
    for subj_idx, subj_data in enumerate(sleep_group):
        avg_power_spectrum = np.mean(subj_data, axis=1)  # full spectrum

        # Restrict frequencies and power spectrum to band range
        band_freqs = frequencies[band_mask]
        band_power = avg_power_spectrum[band_mask]

        bw = compute_fwhm(band_freqs, band_power)
        sleep_bandwidths.append(bw)

        plot_spectrum_with_fwhm(band_freqs, band_power, bw, title=f"Sleep Subject {subj_idx + 1} - {band_name} Band")

    print("Sleep bandwidths per subject:", sleep_bandwidths)

    propofol_bandwidths = []  # To collect FWHM per subject
    for subj_idx, subj_data in enumerate(propofol_group):
        avg_power_spectrum = np.mean(subj_data, axis=1)  # full spectrum

        # Restrict frequencies and power spectrum to band range
        band_freqs = frequencies[band_mask]
        band_power = avg_power_spectrum[band_mask]

        bw = compute_fwhm(band_freqs, band_power)
        propofol_bandwidths.append(bw)

    print("Propofol bandwidths per subject:", propofol_bandwidths)
"""



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# ------
"""# 1. Define window around LoC (e.g., -10s to +10s)
# ------
loc_window = (time >= -10) & (time <= 10)  # Adjust this as needed
loc_idx = np.argmin(np.abs(time))  # Closest time point to 0

# ------
# 2. Flatten data for PCA/t-SNE: subjects x freqs x time -> subjects x (freqs*time)
# ------
def flatten(group):
    return group[:, :, loc_window].reshape(group.shape[0], -1)

sleep_flat = flatten(sleep_group)
prop_flat = flatten(propofol_group)
X = np.vstack([sleep_flat, prop_flat])
y = ['Sleep'] * sleep_flat.shape[0] + ['Propofol'] * prop_flat.shape[0]

# ------
# 3. PCA (at LoC window)
# ------
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ------
# 4. t-SNE (on same scaled data)
# ------
embedding = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(X_scaled)

# ------
# 5. Plot PCA & t-SNE
# ------
def plot_embedding(data, title, labels):
    plt.figure(figsize=(7, 6))
    for label, color in zip(['Sleep', 'Propofol'], ['blue', 'red']):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(data[idx, 0], data[idx, 1], label=label, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_embedding(X_pca, 'PCA at LoC Window', y)
plot_embedding(embedding, 't-SNE at LoC Window', y)

# ------
# 6. K-Means Clustering on t-SNE Embedding
# ------
kmeans = KMeans(n_clusters=2, random_state=42).fit(embedding)
cluster_labels = kmeans.labels_
sil_score = silhouette_score(embedding, cluster_labels)
print(f'Silhouette Score (k=2): {sil_score:.2f}')

plt.figure(figsize=(7, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='coolwarm', alpha=0.8)
plt.title('K-means Clustering (k=2) on t-SNE')
plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
plt.grid(True)
plt.tight_layout()
plt.show()
"""
# ------
# 7. PCA Trajectories over time (optional extension)
# ------
# Reshape all data: subjects x freqs x times -> (subjects * times) x freqs
sleep_reshaped = sleep_group.transpose(0, 2, 1).reshape(-1, sleep_group.shape[1])
prop_reshaped = propofol_group.transpose(0, 2, 1).reshape(-1, propofol_group.shape[1])

X_all = np.vstack([sleep_reshaped, prop_reshaped])
X_all_scaled = StandardScaler().fit_transform(X_all)
pca_model = PCA(n_components=2)
pca_all = pca_model.fit_transform(X_all_scaled)


n_sleep = sleep_group.shape[0]     # number of sleep subjects
n_prop = propofol_group.shape[0]   # number of propofol subjects
n_times = sleep_group.shape[2]     # number of timepoints
y = np.array([0] * (n_sleep * n_times) + [1] * (n_prop * n_times))  # 0 = Sleep, 1 = Propofol

n_sleep, n_times = sleep_group.shape[0], sleep_group.shape[2]
n_prop = propofol_group.shape[0]

sleep_pca = pca_all[:n_sleep * n_times].reshape(n_sleep, n_times, 2)
prop_pca = pca_all[n_sleep * n_times:].reshape(n_prop, n_times, 2)

# Flatten across subjects and timepoints to get all PCA points
sleep_pca_flat = sleep_pca.reshape(-1, 2)    # shape: (n_sleep * n_times, 2)
prop_pca_flat = prop_pca.reshape(-1, 2)      # shape: (n_prop * n_times, 2)

# Mean PCA trajectory for sleep group: shape (n_times, 2)
sleep_mean_traj = sleep_pca.mean(axis=0)

# Mean PCA trajectory for propofol group: shape (n_times, 2)
prop_mean_traj = prop_pca.mean(axis=0)

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.ticker as ticker

# Set color normalization explicitly from -240 to 240
norm = colors.Normalize(vmin=-240, vmax=240)
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(10, 7))

for i in range(n_times):
    color = cmap(norm(time[i]))
    ax.scatter(sleep_mean_traj[i, 0], sleep_mean_traj[i, 1], color=color, label='Sleep' if i == 0 else "", alpha=0.7)
    ax.scatter(prop_mean_traj[i, 0], prop_mean_traj[i, 1], color=color, marker='x', label='Propofol' if i == 0 else "", alpha=0.7)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Mean PCA over Time')

# ✅ Legend inside the plot (top-right corner)
ax.legend(loc='upper right')

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, label='Time progression (seconds)')
cbar.set_ticks(np.arange(-240, 241, 60))
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

ax.grid(True)
plt.tight_layout()
plt.savefig("mean_pca_over_time.png", dpi=300)
plt.show()




# Metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

print("\n--- PCA Metrics Summary ---")

# 1. Explained Variance Ratio
explained_var = pca_model.explained_variance_ratio_
print(f"Explained Variance - PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%}")

# 2. Trajectory Length (total path length) in PCA space
def compute_trajectory_length(traj):
    return np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1))

sleep_traj_len = compute_trajectory_length(sleep_mean_traj)
prop_traj_len = compute_trajectory_length(prop_mean_traj)
print(f"Trajectory Length - Sleep: {sleep_traj_len:.2f}, Propofol: {prop_traj_len:.2f}")

# 3. Mean Velocity (per time step)
mean_velocity_sleep = sleep_traj_len / (n_times - 1)
mean_velocity_prop = prop_traj_len / (n_times - 1)
print(f"Mean Velocity - Sleep: {mean_velocity_sleep:.4f}, Propofol: {mean_velocity_prop:.4f}")

# 4. Centroid Distance
sleep_centroid = np.mean(sleep_pca_flat, axis=0)
prop_centroid = np.mean(prop_pca_flat, axis=0)
centroid_dist = np.linalg.norm(sleep_centroid - prop_centroid)
print(f"Centroid Distance (Sleep vs Propofol): {centroid_dist:.2f}")





# 5. Separability in PCA space (simple logistic classifier)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Encode labels: Sleep=0, Propofol=1
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train logistic regression on PCA representation
clf = LogisticRegression().fit(pca_all, y_enc)
pred_probs = clf.predict_proba(pca_all)[:, 1]
y_pred = clf.predict(pca_all)

# Evaluate
auc = roc_auc_score(y_enc, pred_probs)
acc = accuracy_score(y_enc, y_pred)
print(f"Logistic Classifier in PCA space - Accuracy: {acc:.2%}, AUC: {auc:.2f}")


# Create a meshgrid over PCA space
h = 0.05  # step size for the mesh
x_min, x_max = pca_all[:, 0].min() - 1, pca_all[:, 0].max() + 1
y_min, y_max = pca_all[:, 1].min() - 1, pca_all[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict probabilities for the grid points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot actual PCA points with clearer labels and lower alpha
for label, color in zip(le.classes_, ['blue', 'red']):
    idx = np.where(np.array(y) == label)[0]
    # Change label from numeric to string manually, lower alpha for points
    label_name = 'Sleep' if label == 0 else 'Propofol'
    plt.scatter(pca_all[idx, 0], pca_all[idx, 1], label=label_name, color=color, edgecolor='k', s=60, alpha=0.7)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Logistic Classifier Decision Boundary in PCA Space')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logistic_decision_boundary.png", dpi=300)
plt.show()



print("\n--- PCA Metrics Summary (Over Time) ---")

# 1. Explained Variance Ratio (from full time PCA)
pca_over_time = PCA(n_components=2)
pca_over_time.fit(X_all_scaled)
explained_var_over_time = pca_over_time.explained_variance_ratio_
print(f"Explained Variance - PC1: {explained_var_over_time[0]:.2%}, PC2: {explained_var_over_time[1]:.2%}")

# 2. Trajectory Length (already computed correctly)
sleep_traj_len = compute_trajectory_length(sleep_mean_traj)
prop_traj_len = compute_trajectory_length(prop_mean_traj)
print(f"Trajectory Length - Sleep: {sleep_traj_len:.2f}, Propofol: {prop_traj_len:.2f}")

# 3. Mean Velocity
mean_velocity_sleep = sleep_traj_len / (n_times - 1)
mean_velocity_prop = prop_traj_len / (n_times - 1)
print(f"Mean Velocity - Sleep: {mean_velocity_sleep:.4f}, Propofol: {mean_velocity_prop:.4f}")

# 4. Centroid Distance (flattened across all subjects and timepoints)
sleep_centroid = np.mean(sleep_pca.reshape(-1, 2), axis=0)
prop_centroid = np.mean(prop_pca.reshape(-1, 2), axis=0)
centroid_dist = np.linalg.norm(sleep_centroid - prop_centroid)
print(f"Centroid Distance (Sleep vs Propofol): {centroid_dist:.2f}")

# 5. Logistic Classification Accuracy and AUC over time
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

accs, aucs = [], []

for t in range(n_times):
    # At each time step, gather PCA points for that time
    X_t = np.vstack((sleep_pca[:, t, :], prop_pca[:, t, :]))
    y_t = np.array([0] * n_sleep + [1] * n_prop)  # Sleep=0, Propofol=1

    clf = LogisticRegression().fit(X_t, y_t)
    y_pred = clf.predict(X_t)
    y_prob = clf.predict_proba(X_t)[:, 1]

    accs.append(accuracy_score(y_t, y_pred))
    aucs.append(roc_auc_score(y_t, y_prob))

# Print mean accuracy & AUC over time
print(f"Mean Classifier Accuracy over Time: {np.mean(accs)*100:.2f}%")
print(f"Mean Classifier AUC over Time: {np.mean(aucs):.2f}")