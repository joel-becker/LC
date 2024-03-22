import os

robustness_params = {
    'baseline_risk': [0.1, 0.15, 0.2],
    'infection_rate': [0.01, 0.03, 0.05],
}

def run_simulation(params, save_path, df_symptom_integrals, data_daly):
    lcs = LongCovidSimulator(
        params=params, 
        years=5, 
        n_simulations=30, 
        verbose=False,
        save_path=save_path
    )
    results = lcs.run_many_simulations()
    wlc = DataSimulationsMerger(results, df_symptom_integrals, data_daly)
    df_merged = wlc.calculate_welfare_loss()
    return df_merged

def run_robustness_checks(
        default_params, output_dir, long_covid_simulator, robustness_params=robustness_params
        ):
    for param_name, param_values in robustness_params.items():
        for param_value in param_values:
            # Create a copy of the default parameters
            params = default_params.copy()
            
            # Update the specific parameter value
            params[param_name] = param_value
            
            # Create a unique save path for each parameter configuration
            save_path = os.path.join(output_dir, f"{param_name}_{param_value}.csv")
            
            # Run the simulation with the updated parameters
            df_merged = long_covid_simulator.run_simulation(params, save_path)
            
            # Save the merged DataFrame to a file
            df_merged.to_csv(save_path, index=False)