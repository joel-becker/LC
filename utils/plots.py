import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_daly_loss_over_time(df, population_size_deflator):
    # Convert 'DALY_loss' to numeric, coercing any errors to NaN
    df['DALY_loss'] = pd.to_numeric(df['DALY_loss'], errors='coerce')

    # Complete index of weeks
    #start_date = df['week_start'].min() 
    #end_date = df['week_start'].max()
    #all_weeks = pd.date_range(start=start_date, end=end_date, freq='W-SUN')


    date_range = pd.date_range(start=df['week_start'].min(), end=df['week_start'].max(), freq='W-THU')
    all_dates = pd.DataFrame(date_range, columns=['week_start'])
    simulations = df['simulation'].unique()
    all_combinations = pd.MultiIndex.from_product([date_range, simulations], names=['week_start', 'simulation'])
    complete_data = pd.DataFrame(index=all_combinations).reset_index()
    aggregated_data = (
        df.groupby(['week_start', 'simulation'])
        .agg({'has_long_covid': 'sum', 'DALY_loss': 'sum'})
        .sort_values(by=['simulation', 'week_start'])
        .groupby('simulation')
        .cumsum(numeric_only=True)
        .reset_index()
        #.merge(complete_data, on=['week_start', 'simulation'], how='right')
        #.sort_values(by=['simulation', 'week_start'])
    )
    #aggregated_data.fillna(method='ffill', inplace=True)
    #aggregated_data.fillna(0, inplace=True)

    aggregated_data['has_long_covid'] = aggregated_data['has_long_covid'] * population_size_deflator
    aggregated_data['DALY_loss'] = aggregated_data['DALY_loss'] * population_size_deflator
    ## Separate data by simulation and aggregate
    #aggregated_data = (
    #    df.groupby(['week_start', 'simulation'])
    #        .agg({'has_long_covid': 'sum', 'DALY_loss': 'sum'})
    #        .sort_values(by=['simulation', 'week_start'])
    #        #.groupby('simulation')
    #        #.cumsum(numeric_only=True)
    #        #.reset_index()
    #    )
#
    ## Reset index to prepare for reindexing
    #aggregated_data = aggregated_data.reset_index()
    #
    ## Set the new index and unstack simulations to columns
    #aggregated_data.set_index(['week_start', 'simulation'], inplace=True)
    #aggregated_data = aggregated_data.unstack(level='simulation')
    #
    ## Reindex the DataFrame to include all weeks, filling missing weeks
    #aggregated_data = aggregated_data.reindex(all_weeks, method='ffill')

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

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_parameter_distributions(base_path):
    # Configure the aesthetics for seaborn plots
    sns.set(style="whitegrid")
    
    # Prepare to collect aggregated data
    total_daly_loss_by_file = []
    
    # List files in the directory
    for filename in os.listdir(base_path):
        if filename.endswith(".csv") and "parameters" not in filename:
            # Construct file path
            file_path = os.path.join(base_path, filename)
            # Read the data
            df = pd.read_csv(file_path)
            # Convert DALY_loss to numeric, handling errors
            df['DALY_loss'] = pd.to_numeric(df['DALY_loss'], errors='coerce')
            # Group by simulation and sum DALY_loss
            total_daly_loss = df.groupby('simulation')['DALY_loss'].sum().reset_index()
            # Extract parameter name and value from filename
            param_name_value = filename.split('_')
            param_name = '_'.join(param_name_value[:-1])  # Join all but the last part
            param_value = param_name_value[-1].replace('.csv', '')  # Remove the file extension
            total_daly_loss['Parameter'] = param_name
            total_daly_loss['Value'] = param_value
            total_daly_loss_by_file.append(total_daly_loss)
    
    # Combine all data frames
    combined_data = pd.concat(total_daly_loss_by_file)
    
    # Plot settings
    plt.figure(figsize=(12, 8))
    
    # Group data by the parameter to create subplots
    parameters = combined_data['Parameter'].unique()
    for i, param in enumerate(sorted(parameters)):
        ax = plt.subplot(len(parameters), 1, i + 1)
        
        # Filter data by parameter
        param_data = combined_data[combined_data['Parameter'] == param]
        
        # Sort values for coloring
        order = sorted(param_data['Value'].unique(), key=float)  # Sorting numerically
        palette = sns.color_palette("coolwarm", n_colors=len(order))
        
        # Draw the distribution plots
        sns.histplot(data=param_data, x="DALY_loss", hue="Value", element="step", fill=True,
                     palette=palette, common_norm=False, ax=ax, kde=True)
        
        # Aesthetics
        ax.set_title(f'Distribution of Total DALY_loss for {param}')
        ax.set_ylabel('Density')
        if i < len(parameters) - 1:
            ax.set_xlabel('')

    plt.tight_layout()
    plt.show()

# Usage
base_path = 'output/tables'  # Directory containing the CSV files
#plot_parameter_distributions(base_path)


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_parameter_distributions(base_path, population_size_deflator=30_000):
    # Configure the aesthetics for seaborn plots
    sns.set(style="whitegrid")
    
    # Prepare to collect aggregated data
    total_daly_loss_by_file = []
    
    # List files in the directory
    for filename in os.listdir(base_path):
        if filename.endswith(".csv") and "parameters" not in filename:
            # Construct file path
            file_path = os.path.join(base_path, filename)
            # Read the data
            df = pd.read_csv(file_path)
            # Convert DALY_loss to numeric, handling errors
            df['DALY_loss'] = pd.to_numeric(df['DALY_loss'], errors='coerce') * population_size_deflator
            # Group by simulation and sum DALY_loss
            total_daly_loss = df.groupby('simulation')['DALY_loss'].sum().reset_index()
            # Extract parameter name and value from filename
            param_info = filename.replace('.csv', '').split('_')
            param_name = param_info[0]
            param_value = param_info[2]  # Assuming the value is the third part after splitting
            total_daly_loss['Parameter'] = param_name
            total_daly_loss['Value'] = float(param_value)
            total_daly_loss_by_file.append(total_daly_loss)
    
    # Combine all data frames
    combined_data = pd.concat(total_daly_loss_by_file)
    
    # Plot settings
    plt.figure(figsize=(12, 8))
    
    # Group data by the parameter to create subplots
    parameters = combined_data['Parameter'].unique()
    for i, param in enumerate(sorted(parameters)):
        ax = plt.subplot(len(parameters), 1, i + 1)
        
        # Filter data by parameter
        param_data = combined_data[combined_data['Parameter'] == param]
        
        # Sort values for coloring and legend ordering
        sorted_values = sorted(param_data['Value'].unique(), key=float)  # Sorting numerically
        palette = ['green', 'orange', 'red'][:len(sorted_values)]
        
        # Draw the distribution plots
        sns.histplot(data=param_data, x="DALY_loss", hue="Value", element="step", fill=True,
                     palette=palette, common_norm=False, ax=ax, kde=True, hue_order=sorted_values)
        
        # Aesthetics
        ax.set_ylabel(param)
        ax.set_title('')  # Remove subplot title
        if i < len(parameters) - 1:
            ax.set_xlabel('')

    plt.tight_layout()
    plt.show()
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def plot_parameter_distributions(base_path, population_size_deflator=30_000, years=2, use_symlog=False, linthresh=1000):
    sns.set(style="whitegrid")
    
    total_daly_loss_by_file = []
    
    for filename in os.listdir(base_path):
        if filename.endswith(".csv") and "parameters" not in filename:
            file_path = os.path.join(base_path, filename)
            df = pd.read_csv(file_path)
            df['DALY_loss'] = pd.to_numeric(df['DALY_loss'], errors='coerce') * (population_size_deflator / years)
            total_daly_loss = df.groupby('simulation')['DALY_loss'].sum().reset_index()
            
            param_info = filename.replace('.csv', '').split('_')
            param_name = param_info[0] + '_' + param_info[1]
            param_value = param_info[2]
            
            total_daly_loss['Parameter'] = param_name
            total_daly_loss['Value'] = float(param_value)
            total_daly_loss_by_file.append(total_daly_loss)
    
    combined_data = pd.concat(total_daly_loss_by_file)
    plt.figure(figsize=(12, 8))
    
    parameters = combined_data['Parameter'].unique()
    global_max = combined_data['DALY_loss'].max()
    
    for i, param in enumerate(sorted(parameters)):
        ax = plt.subplot(len(parameters), 1, i + 1)
        
        param_data = combined_data[combined_data['Parameter'] == param]
        sorted_values = sorted(param_data['Value'].unique(), key=float)
        palette = ['green', 'orange', 'red'][:len(sorted_values)]
        
        sns.histplot(data=param_data, x="DALY_loss", hue="Value", element="step", fill=True,
                     palette=palette, common_norm=False, ax=ax, kde=True, hue_order=sorted_values)
        
        ax.set_xlim(left=linthresh, right=global_max)
        ax.set_ylabel(param)
        ax.grid(False, axis='y')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # Format x-axis with commas
        
        if use_symlog:
            ax.set_xscale('symlog', linthresh=linthresh)  # Set x-axis to symmetrical logarithmic scale

        if i < len(parameters) - 1:
            ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig('output/plots/robustness_distribution.png')

# Usage
base_path = 'output/tables'  # Directory containing the CSV files
#plot_parameter_distributions(base_path)

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_tornado_diagram(base_path, population_size_deflator=30_000, years=2):
    sns.set(style="whitegrid")

    results = []

    for filename in os.listdir(base_path):
        if filename.endswith(".csv") and "parameters" not in filename:
            file_path = os.path.join(base_path, filename)
            df = pd.read_csv(file_path)
            df['Adjusted_DALY_loss'] = pd.to_numeric(df['DALY_loss'], errors='coerce') * (population_size_deflator / years)
            
            # Assume that 'simulation' uniquely identifies a scenario
            total_daly_loss = df.groupby('simulation')['Adjusted_DALY_loss'].sum().reset_index()

            param_info = filename.replace('.csv', '').split('_')
            param_name = param_info[0] + '_' + param_info[1]
            param_value = param_info[2]

            # Record parameter, value, and total DALY loss
            for index, row in total_daly_loss.iterrows():
                results.append({'Parameter': param_name, 'Value': float(param_value), 'DALY_loss': row['Adjusted_DALY_loss'], 'Simulation': row['simulation']})

    # Create DataFrame from results
    df_results = pd.concat([pd.DataFrame([i]) for i in results], ignore_index=True)
    
    # Calculate mean DALY_loss for each parameter-value pair
    mean_daly_losses = df_results.groupby(['Parameter', 'Value'])['DALY_loss'].mean().reset_index()

    # Determine baseline values - assuming it's the median of values for simplicity
    baseline_values = mean_daly_losses.groupby('Parameter')['Value'].median().reset_index()
    baseline_values = baseline_values.rename(columns={'Value': 'Baseline_Value'})

    # Merge to find baseline DALY_loss
    mean_daly_losses = mean_daly_losses.merge(baseline_values, on='Parameter', how='left')
    baseline_daly = mean_daly_losses[mean_daly_losses['Value'] == mean_daly_losses['Baseline_Value']]
    baseline_daly = baseline_daly.rename(columns={'DALY_loss': 'Baseline_DALY'})

    # Calculate deviations from baseline
    mean_daly_losses['Delta'] = mean_daly_losses.apply(lambda row: row['DALY_loss'] - baseline_daly[baseline_daly['Parameter'] == row['Parameter']]['Baseline_DALY'].values[0], axis=1)

    # Plotting
    plt.figure(figsize=(10, 6))
    for param in parameters:
        param_data = mean_daly_losses[mean_daly_losses['Parameter'] == param]
        for _, row in param_data.iterrows():
            # Determine the starting point of the bar
            if row['Delta'] < 0:
                # Start from the negative value and extend to 0
                bar_start = row['Delta']
            else:
                # Start from 0 and extend to the positive value
                bar_start = 0
            # Determine the color of the bar
            color = 'red' if row['Delta'] > 0 else 'green'
            # Plot the bar
            plt.barh(param, abs(row['Delta']), color=color, left=bar_start, align='center')

    plt.xlabel('Change in DALY_loss')
    plt.title('Tornado Diagram of Parameter Impact')
    plt.axvline(x=0, color='blue', linestyle='--')  # Line for baseline
    plt.grid(True, axis='x')
    plt.show()

# Usage
base_path = 'output/tables'  # Directory containing the CSV files
#plot_tornado_diagram(base_path)
