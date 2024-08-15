import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_daly_loss_over_time(df, population_size_deflator, 
                             output_file=None, 
                             line_color='#7570b3', 
                             grid_color='#E0E0E0', 
                             fig_size=(15, 10),
                             font_size=12,
                             dpi=300,
                             y_axis_margin=0.01):
    """
    Generate a plot of DALY loss and Long COVID cases over time.
    
    Parameters:
    - df (pandas.DataFrame): DataFrame containing the data.
    - population_size_deflator (float): Factor to adjust the population size.
    - output_file (str, optional): Path to save the output plot. If None, the plot is displayed instead.
    - line_color (str): Color of the plot lines in hex format.
    - grid_color (str): Color of the grid lines in hex format.
    - fig_size (tuple): Figure size in inches (width, height).
    - font_size (int): Font size for subplot titles and labels.
    - dpi (int): DPI for the output file (if saving).
    - y_axis_margin (float): Margin for y-axis limits.
    
    Returns:
    - fig, axes: The generated figure and axes objects.
    """
    # Convert 'DALY_loss' to numeric, coercing any errors to NaN
    df['DALY_loss'] = pd.to_numeric(df['DALY_loss'], errors='coerce')

    # Create a complete date range
    date_range = pd.date_range(start=df['week_start'].min(), end=df['week_start'].max(), freq='W-THU')
    
    # Check if 'simulation' column exists, if not, create a dummy one
    if 'simulation' not in df.columns:
        df['simulation'] = 0
    simulations = df['simulation'].unique()
    
    # Create all combinations of dates and simulations
    all_combinations = pd.MultiIndex.from_product([date_range, simulations], names=['week_start', 'simulation'])
    complete_data = pd.DataFrame(index=all_combinations).reset_index()
    
    # Aggregate the data
    aggregated_data = (
        df.groupby(['week_start', 'simulation'])
        .agg({'has_long_covid': 'sum', 'DALY_loss': 'sum'})
        .reset_index()
    )
    
    # Merge aggregated data with complete date range
    merged_data = complete_data.merge(aggregated_data, on=['week_start', 'simulation'], how='left')
    
    # Sort the data
    merged_data = merged_data.sort_values(['simulation', 'week_start'])
    
    # Forward fill the data within each simulation group
    merged_data['has_long_covid'] = merged_data.groupby('simulation')['has_long_covid'].ffill()
    merged_data['DALY_loss'] = merged_data.groupby('simulation')['DALY_loss'].ffill()
    
    # Fill NaN values with 0
    merged_data = merged_data.fillna(0)
    
    # Calculate cumulative sum for each simulation
    merged_data['has_long_covid'] = merged_data.groupby('simulation')['has_long_covid'].cumsum()
    merged_data['DALY_loss'] = merged_data.groupby('simulation')['DALY_loss'].cumsum()
    
    # Apply population size deflator
    merged_data['has_long_covid'] = merged_data['has_long_covid'] * population_size_deflator
    merged_data['DALY_loss'] = merged_data['DALY_loss'] * population_size_deflator
    
    # Calculate mean for each period
    mean_data = merged_data.groupby('week_start').mean().reset_index()

    # Set up the plot
    fig, axes = plt.subplots(2, 1, figsize=fig_size, sharex=True)
    fig.suptitle('Long COVID Cases and Associated DALY Loss Over Time', fontsize=font_size+2)

    titles = ['Total Long COVID Cases', 'Total DALY Loss']
    y_labels = ['Cases', 'DALY Loss']
    data_cols = ['has_long_covid', 'DALY_loss']

    for i, (ax, title, ylabel, data_col) in enumerate(zip(axes, titles, y_labels, data_cols)):
        # Plot each simulation as faint lines
        for sim in simulations:
            sim_data = merged_data[merged_data['simulation'] == sim]
            ax.plot(sim_data['week_start'], sim_data[data_col], color=line_color, alpha=0.2)

        # Plot mean line
        ax.plot(mean_data['week_start'], mean_data[data_col], color=line_color, linewidth=2, label='Mean')

        # Set title and labels
        ax.set_title(title, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)

        # Remove box
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Set grid
        ax.grid(True, which='major', linestyle='--', color=grid_color, alpha=0.7)

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Remove tick marks
        ax.tick_params(axis='both', which='both', length=0)

    # Format x-axis
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Set common x-label
    fig.text(0.5, 0.02, 'Week Start Date', ha='center', va='center', fontsize=font_size)

    # Add a common legend
    legend_elements = [plt.Line2D([0], [0], color=line_color, alpha=1, label='Mean'),
                       plt.Line2D([0], [0], color=line_color, alpha=0.2, label='Individual Simulations')]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=2, frameon=False, fontsize=font_size)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.9)  # Increased bottom margin for legend

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

    return fig, axes


def plot_symptom_prevalence_over_time(data, 
                             output_file=None, 
                             line_color='#7570b3', 
                             grid_color='#E0E0E0', 
                             fig_size=(15, 30),
                             font_size=12,
                             dpi=300,
                             y_axis_margin=0.01):
    """
    Generate a plot of Long COVID symptom prevalence over time.
    
    Parameters:
    - data (str or pandas.DataFrame): Path to the CSV file containing the symptom data,
      or a pandas DataFrame containing the data.
    - output_file (str, optional): Path to save the output plot. If None, the plot is displayed instead.
    - line_color (str): Color of the plot lines in hex format.
    - grid_color (str): Color of the grid lines in hex format.
    - fig_size (tuple): Figure size in inches (width, height).
    - font_size (int): Font size for subplot titles and labels.
    - dpi (int): DPI for the output file (if saving).
    - y_axis_margin (float): Margin for y-axis limits.
    
    Returns:
    - fig, axes: The generated figure and axes objects.
    """
    
    # Read the data if a file path is provided, otherwise use the DataFrame directly
    if isinstance(data, str):
        data = pd.read_csv(data, sep='\s{2,}', engine='python')
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be either a file path (str) or a pandas DataFrame")
    
    # Calculate number of rows needed
    n_rows = -(-len(data) // 3)  # Ceiling division
    
    # Set up the plot
    fig, axes = plt.subplots(n_rows, 3, figsize=fig_size, sharex=True, sharey=True)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Define the x-axis points for detailed plotting
    x_detailed = np.arange(0, 24.01, 0.01)
    
    # Plot each symptom
    for i, (_, row) in enumerate(data.iterrows()):
        ax = axes[i]
        
        # Create detailed DataFrame for plotting
        y_adjusted = np.concatenate([
            np.full(600, row['prevalence_diff_6m']/100),
            np.full(600, row['prevalence_diff_12m']/100),
            np.full(600, row['prevalence_diff_18m']/100),
            np.full(601, 0)
        ])
        y_raw = np.concatenate([
            np.full(600, row['prevalence_diff_6m']/100),
            np.full(600, row['prevalence_diff_12m']/100),
            np.full(600, row['previous_prevalence_diff_18m']/100),
            np.full(601, 0)
        ])
        
        # Plot lines
        ax.plot(x_detailed, y_adjusted, color=line_color, alpha=1, linewidth=2)
        ax.plot(x_detailed, y_raw, color=line_color, alpha=0.3, linewidth=2)
        
        # Set title and remove box
        ax.set_title(row['symptom'], fontsize=font_size)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Set x-axis ticks and grid
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(['0', '6', '12', '18', ''])
        ax.set_xlim(0, 24)
        ax.xaxis.grid(True, which='major', linestyle='--', color=grid_color, alpha=0.7)
        
        # Set y-axis ticks, labels, and grid
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.set_yticklabels(['0%', '5%', '10%', '15%', ''])
        ax.set_ylim(0-y_axis_margin, 0.20+y_axis_margin)
        ax.yaxis.grid(True, which='major', linestyle='--', color=grid_color, alpha=0.7)
        
        # Remove tick marks
        ax.tick_params(axis='both', which='both', length=0)
            
    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Set common labels with more space
    fig.text(0.5, 0.02, 'Months Since Infection', ha='center', va='center', fontsize=font_size)
    fig.text(0.01, 0.5, 'Prevalence Difference', ha='center', va='center', rotation='vertical', fontsize=font_size)
    
    # Add a common legend
    legend_elements = [plt.Line2D([0], [0], color=line_color, alpha=1, label='Adjusted Prevalence Difference'),
                       plt.Line2D([0], [0], color=line_color, alpha=0.3, label='Raw Prevalence Difference')]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), 
               ncol=2, frameon=False, fontsize=font_size)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, left=0.06, top=0.95, right=0.98)  # Adjusted margins
    
    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig, axes


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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tornado_diagram(data_path, 
                         output_file=None, 
                         population_size_deflator=30_000, 
                         years=3,
                         central_param_value='median',
                         lower_color='#4daf4a',
                         higher_color='#e41a1c',
                         baseline_color='#377eb8',
                         grid_color='#E0E0E0',
                         fig_size=(12, 10),
                         font_size=12,
                         dpi=300):
    """
    Generate a tornado diagram of parameter impact on DALY loss.
    
    Parameters:
    - data_path (str): Path to the directory containing CSV files with simulation results.
    - output_file (str, optional): Path to save the output plot. If None, the plot is displayed instead.
    - population_size_deflator (int): Factor to adjust DALY loss calculations.
    - years (int): Number of years for DALY loss calculation.
    - central_param_value (str): Method to calculate central value ('median' or 'mean').
    - lower_color (str): Color for lower parameter values in hex format.
    - higher_color (str): Color for higher parameter values in hex format.
    - baseline_color (str): Color for the baseline line in hex format.
    - grid_color (str): Color of the grid lines in hex format.
    - fig_size (tuple): Figure size in inches (width, height).
    - font_size (int): Font size for labels and title.
    - dpi (int): DPI for the output file (if saving).
    
    Returns:
    - fig, ax: The generated figure and axis objects.
    """
    
    # Set the style for the plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=fig_size)

    # Function to process each CSV file
    def process_file(file_path):
        df = pd.read_csv(file_path)
        df['Adjusted_DALY_loss'] = pd.to_numeric(df['DALY_loss'], errors='coerce') * (population_size_deflator / years)
        
        param_info = os.path.basename(file_path).replace('.csv', '').split('_')
        param_name = '_'.join(param_info[:-1])  # Join all parts except the last one
        param_value = float(param_info[-1])  # The last part is the parameter value
        
        total_daly_loss = df.groupby('simulation')['Adjusted_DALY_loss'].sum().mean()
        
        return pd.DataFrame({'Parameter': [param_name], 'Value': [param_value], 'DALY_loss': [total_daly_loss]})

    # Process all CSV files
    results = pd.concat([process_file(os.path.join(data_path, f)) for f in os.listdir(data_path) 
                         if f.endswith('.csv') and 'parameters' not in f], ignore_index=True)

    # Calculate central values for each parameter
    if central_param_value == 'median':
        central_values = results.groupby('Parameter')['Value'].median().reset_index()
    elif central_param_value == 'mean':
        central_values = results.groupby('Parameter')['Value'].mean().reset_index()
    else:
        raise ValueError("central_param_value must be either 'median' or 'mean'")

    central_values = central_values.rename(columns={'Value': 'Central_Value'})
    
    # Merge to find central DALY_loss
    results = results.merge(central_values, on='Parameter', how='left')
    central_daly = results[results['Value'] == results['Central_Value']]
    central_daly = central_daly.rename(columns={'DALY_loss': 'Central_DALY'})

    # Calculate deviations from central value
    results = results.merge(central_daly[['Parameter', 'Central_DALY']], on='Parameter', how='left')
    results['Delta'] = results['DALY_loss'] - results['Central_DALY']

    # Sort parameters by their maximum absolute impact
    param_order = results.groupby('Parameter')['Delta'].agg(lambda x: x.abs().max()).sort_values(ascending=False).index

    # Plotting
    for i, param in enumerate(param_order):
        param_data = results[results['Parameter'] == param]
        central_value = central_values[central_values['Parameter'] == param]['Central_Value'].values[0]
        
        lower_value = param_data[param_data['Value'] < central_value]
        higher_value = param_data[param_data['Value'] > central_value]
        
        if not lower_value.empty:
            ax.barh(i, lower_value['Delta'].values[0], height=0.6, color=lower_color, alpha=0.7)
        if not higher_value.empty:
            ax.barh(i, higher_value['Delta'].values[0], height=0.6, color=higher_color, alpha=0.7)

    # Customize the plot
    ax.axvline(x=0, color=baseline_color, linestyle='--', linewidth=1)
    ax.set_yticks(range(len(param_order)))
    ax.set_yticklabels(param_order)
    ax.set_xlabel('Change in DALY Loss', fontsize=font_size)
    ax.set_title('Tornado Diagram of Parameter Impact on DALY Loss', fontsize=font_size+2, fontweight='bold')
    
    # Set grid
    ax.xaxis.grid(True, linestyle='--', color=grid_color, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=lower_color, alpha=0.7, label='Lower Value'),
                       plt.Rectangle((0,0),1,1, facecolor=higher_color, alpha=0.7, label='Higher Value'),
                       plt.Line2D([0], [0], color=baseline_color, linestyle='--', label='Baseline')]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=font_size)

    # Adjust layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig, ax

# Example usage:
fig, ax = plot_tornado_diagram("output/tables/robustness", output_file="output/plots/robustness_tornado_diagram.png")