import squigglepy as sq
import pandas as pd

param_descriptions = {
    'size': 'Total population size',
    'baseline_risk': 'Baseline risk of long COVID',
    'infection_rate': 'Weekly infection rate',
    'strain_reduction_factor': 'Reduction factor for new strains',
    'total_strains': 'Number of distinct strains', 
    'current_strain': 'Current strain', 
    'strain_decay': 'Half-life of strain relative prevalence',
    'initial_vaccination_distribution': 'Initial distribution of number of vaccinations per person',
    'vaccination_reduction': 'Peak effectiveness of vaccination (against Long COVID, conditional on COVID)', 
    'vaccination_interval': 'Minimum interval between vaccinations', 
    'vaccination_effectiveness_halflife': 'Half-life of vaccination effectiveness', 
    'aor_value': 'AOR value'
}

default_params = {
    'size': 330_000_000,
    'baseline_risk': sq.beta(100*0.15, 100*(1-0.15)), 
    'infection_rate': (sq.to(6687391, 40968288) / 330_000_000) / 52,
    'vaccination_reduction': sq.beta(100*0.25, 100*(1-0.25)), 
    'vaccination_interval': 180, 
    'vaccination_effectiveness_halflife': sq.norm(mean=148, sd=30), 
    'p_never_vaccinated': 0.2,
    'p_two_shots': 0.5,
    'p_boosted_yearly': 0.3,
    'covid_risk_reduction_factor': 0.7
}

robustness_params = {
    'baseline_risk': [0.1, 0.15, 0.2],
    'infection_rate': [x / (330_000_000 * 52) for x in [6687391, 19202639, 40968288]],
    'vaccination_reduction': [0.15, 0.25, 0.35],
    'vaccination_effectiveness_halflife': [75, 148, 300],
    'covid_risk_reduction_factor' : [0.5, 0.7, 0.9]
}

def merge_and_describe_parameters(default_params, scenario_params, descriptions):
    # Merge default and scenario-specific parameters
    merged_params = {**default_params, **scenario_params}
    
    # Prepare data for DataFrame
    data = []
    for param, value in merged_params.items():
        description = descriptions.get(param, "No description available")
        data.append([param, description, value])
    
    return pd.DataFrame(data, columns=['Parameter', 'Description', 'Value'])


def is_number(value):
    """Check if the value is a number."""
    return isinstance(value, (int, float))

def format_distribution(value):
    """Format distribution objects based on numerical output."""
    # Attempt to access and format mean and standard deviation if applicable
    try:
        if is_number(value.mean) and is_number(value.sd):
            formatted_mean = f"{value.mean:.3g}"
            formatted_std = f"{value.sd:.3g}"
            return f"Normal({formatted_mean}, {formatted_std})"
    except AttributeError:
        pass

    # Attempt to format based on 'a' and 'b' attributes for Beta distributions
    try:
        if is_number(value.a) and is_number(value.b):
            formatted_a = f"{value.a:.3g}"
            formatted_b = f"{value.b:.3g}"
            return f"Beta({formatted_a}, {formatted_b})"
    except AttributeError:
        pass

    # If none of the above, return a generic representation
    return "Distribution"

def format_value(value):
    """Format numbers, distributions, and dictionaries appropriately."""
    if is_number(value):  # Direct numerical values
        return f"{value:.3g}"
    elif isinstance(value, dict):  # Dictionaries
        return {k: format_value(v) for k, v in value.items()}
    else:  # Attempt to format as distribution
        return format_distribution(value)

def generate_comparison_table(default_params, scenario_params, descriptions):
    # Create an empty DataFrame for the comparison table
    table = pd.DataFrame(columns=['Parameter', 'Description', 'Mainline', 'Pessimistic'])
    
    # Iterate over all parameters in the descriptions
    rows = []
    for param, description in descriptions.items():
        mainline_value = format_value(default_params.get(param, 'N/A'))
        pessimistic_value = format_value(scenario_params.get(param, default_params.get(param, 'N/A')))
        rows.append({'Parameter': param, 'Description': description, 'Mainline': mainline_value, 'Pessimistic': pessimistic_value})
    
    # Use pd.concat instead of append to avoid FutureWarning
    table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    
    return table
