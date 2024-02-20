import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_daly_loss_over_time(df):
    # Convert 'DALY_loss' to numeric, coercing any errors to NaN
    df['DALY_loss'] = pd.to_numeric(df['DALY_loss'], errors='coerce')

    # Separate data by simulation and aggregate
    aggregated_data = (
        df.groupby(['week_start', 'simulation'])
            .agg({'has_long_covid': 'sum', 'DALY_loss': 'sum'})
            .sort_values(by=['simulation', 'week_start'])
            .groupby('simulation')
            .cumsum(numeric_only=True)
            .reset_index()
        )

    # Calculate mean for each period
    mean_data = aggregated_data.groupby('week_start').mean().reset_index()

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Long COVID Cases and Associated DALY Loss Over Time')

    # Plot each simulation as faint lines
    for sim in df['simulation'].unique():
        sim_data = aggregated_data[aggregated_data['simulation'] == sim]
        axs[0].plot(sim_data['week_start'], sim_data['has_long_covid'], color='green', alpha=0.2)
        axs[1].plot(sim_data['week_start'], sim_data['DALY_loss'], color='orange', alpha=0.2)

    # Plot mean line
    axs[0].plot(mean_data['week_start'], mean_data['has_long_covid'], color='green', linewidth=2, label='Mean')
    axs[1].plot(mean_data['week_start'], mean_data['DALY_loss'], color='orange', linewidth=2, label='Mean')

    # Set titles and labels
    axs[0].set_title('Total Long COVID Cases')
    axs[0].set_ylabel('Cases')
    axs[1].set_title('Total DALY Loss')
    axs[1].set_ylabel('DALY Loss')

    # Format the x-axis
    for ax in axs:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        if mean_data['week_start'].max() - mean_data['week_start'].min() > pd.Timedelta('365 days'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True)
        ax.legend()

    plt.xticks(rotation=45)
    plt.xlabel('Week Start Date')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to fit the title
    plt.savefig('output/plots/daly_loss_over_time.png')


def plot_symptom_years_histograms(df, num_subplots, alpha=0.5, colors=None):
    """
    Plots histograms of years of symptom prevalence in subplots, with thresholds automatically calculated.

    Args:
    df (DataFrame): DataFrame containing symptom prevalence data.
    num_subplots (int): Number of subplots to create.
    alpha (float): Alpha value for histogram transparency.
    colors (list): List of colors for the histograms.

    Returns:
    matplotlib figure: Figure containing the histograms.
    """
    # Calculate quantile thresholds
    symptom_means = df.mean()
    quantiles = np.linspace(0, 1, num_subplots + 1)[1:-1]
    year_thresholds = symptom_means.quantile(quantiles).tolist()

    # If colors are not provided, generate a color palette
    if colors is None:
        colors = plt.cm.jet(np.linspace(0, 1, len(df.columns)))

    # Create figure and subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))
    if num_subplots == 1:
        axes = [axes]

    # Create thresholds for slicing the DataFrame
    thresholds = [0] + year_thresholds + [df.max().max()]

    for i, ax in enumerate(axes):
        # Identify symptoms within the current threshold range
        filtered_symptoms = symptom_means[
            (symptom_means >= thresholds[i]) & 
            (symptom_means < thresholds[i + 1])
        ].index.tolist()

        # Generate a color palette for these symptoms
        colors = plt.cm.jet(np.linspace(0, 1, len(filtered_symptoms)))

        # Plot histograms for each symptom in the current threshold range
        for col, color in zip(filtered_symptoms, colors):
            symptom_values = df[col]
            ax.hist(symptom_values, bins=50, alpha=alpha, color=color, label=col)

        ax.set_title(f'Symptoms prevalence for years between (mean) {thresholds[i]:.2f} and {thresholds[i+1]:.2f}')
        ax.set_xlabel('Years')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig('output/plots/symptom_histograms.png')
    

def plot_daly_adjustments(data_daly):
    """
    Plots DALY adjustments per symptom categorized by severity and mean DALY adjustment.

    Args:
    data_daly (DataFrame): DataFrame containing DALY adjustment data for symptoms.

    Returns:
    matplotlib figure: Figure containing the subplots.
    """
    # Calculate mean DALY adjustment for mild version of each symptom
    mean_daly_mild = data_daly[data_daly['mild'] == 1].groupby('symptom')['daly_adjustment'].mean()
    sorted_symptoms = mean_daly_mild.sort_values().index.tolist()

    # Categorize symptoms into 3 bins by mean DALY adjustment quantiles
    quantiles = mean_daly_mild.quantile([1/3, 2/3]).tolist()
    bins = pd.cut(mean_daly_mild, bins=[0] + quantiles + [float('inf')], labels=False, include_lowest=True)

    # Create a DataFrame to hold the binned symptom names
    symptom_bins = {bin_num: [] for bin_num in range(3)}
    for symptom, bin_num in bins.iteritems():
        symptom_bins[bin_num].append(symptom)

    # Find the maximum DALY adjustment for y-axis scaling
    max_daly = data_daly['daly_adjustment'].max()

    # Create 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    severity_columns = ['mild', 'moderate', 'severe']

    for i, severity in enumerate(severity_columns):
        for bin_num in range(3):
            ax = axes[bin_num, i]
            # Get the symptom names for this bin
            bin_symptoms = symptom_bins[bin_num]
            # Filter and plot symptoms in the current bin and severity
            for symptom in bin_symptoms:
                symptom_data = data_daly[(data_daly['symptom'] == symptom) & (data_daly[severity] == 1)]
                ax.bar(symptom, symptom_data['daly_adjustment'].values[0], label=symptom)

            ax.set_title(f'{severity.capitalize()} (Symptom bin {bin_num + 1})')
            ax.set_ylim(0, max_daly)  # Standardize y-axis scale
            ax.set_ylabel('DALY Adjustment')
            # Set the x-ticks for the symptoms in this bin
            ax.set_xticks(range(len(bin_symptoms)))
            ax.set_xticklabels(bin_symptoms, rotation=45)  # Rotate x-axis labels
            ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig('output/plots/daly_adjustments.png')

def plot_all_symptoms(trace, symptoms, time_points):
    n_symptoms = len(symptoms)
    n_cols = 3
    n_rows = int(np.ceil(n_symptoms / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten to simplify indexing

    for i, symptom in enumerate(symptoms):
        baseline_samples = trace.posterior['baseline'].values[:, :, i]
        decay_rate_samples = trace.posterior['decay_rate'].values[:, :, i]

        # Generate predictions for each sample in the trace
        prevalence_pred = np.array([[baseline * np.exp(-decay_rate * time) for time in time_points] 
                                    for baseline, decay_rate in zip(baseline_samples.flatten(), decay_rate_samples.flatten())])

        percentiles = np.percentile(prevalence_pred, [2.5, 97.5], axis=0)
        axes[i].fill_between(time_points, percentiles[0], percentiles[1], alpha=0.3)
        axes[i].plot(time_points, np.mean(prevalence_pred, axis=0))
        axes[i].set_title(symptom)
        axes[i].set_xlabel('Time (months)')
        axes[i].set_ylabel('Prevalence')

    for ax in axes[n_symptoms:]:  # Hide any unused subplots
        ax.set_visible(False)

    plt.suptitle('Decay of Symptoms Prevalence Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('output/plots/symptom_decay.png')