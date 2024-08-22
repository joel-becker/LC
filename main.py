from utils.process_daly_adjustments import DalyDataProcessor
from utils.process_symptom_prevalence import SymptomPrevalenceDataProcessor
from utils.estimate_symptom_prevalence_decay import SymptomPrevalenceEstimator, PiecewiseConstantIntegralEstimator
import utils.parameters as params
from utils.simulate_long_covid_cases import LongCovidSimulator
from utils.merge_data_with_simulations import DataSimulationsMerger
import utils.plots as plots

import pickle
import numpy as np
import logging
import time

# Setup logging
logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Data processing started.')


def main():
    years = 20
    n_simulations = 3
    population_size_deflator = 300_000

    # Setup logging
    logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Data processing started.')

    # Process data
    ddp = DalyDataProcessor('data/daly.csv')
    data_daly = ddp.process_data()

    spdp = SymptomPrevalenceDataProcessor('data/prevalence_and_symptoms.csv')
    data_symptom_prevalence = spdp.process_data()

    # Estimation
    try:
        spe = PiecewiseConstantIntegralEstimator(data_symptom_prevalence, num_draws = int(round(330_000_000 / population_size_deflator)))
        normalized_integrals_df = spe.calculate_normalized_piecewise_constant_integrals_with_uncertainty()    
        df_symptom_integrals = normalized_integrals_df

        if df_symptom_integrals is not None:
            with open('temp/df_symptom_integrals.pkl', 'wb') as f:
                pickle.dump(df_symptom_integrals, f)
            
        logging.info('Successfully estimated symptom prevalence decay.')
    except Exception as e:
        logging.error('Error processing decay curves: %s', e)

    # Simulation
    try:
        comparison_table = params.generate_comparison_table(
            params.default_params, 
            params.robustness_params, 
            params.param_descriptions
            )
        with open('output/tables/parameters.txt', 'w') as f:
            # Write headers
            f.write('\t'.join(comparison_table.columns) + '\n')

            # Write each row
            for index, row in comparison_table.iterrows():
                row_str = '\t'.join(str(x) for x in row.values)
                f.write(row_str + '\n')
    except Exception as e:
        logging.error('Error generating comparison table: %s', e)

    try:
        # Modify population size for speed
        params.default_params['size'] = int(round(params.default_params['size'] / population_size_deflator))

        # Run the simulation
        save_path = 'temp/results.pkl'
        lcs = LongCovidSimulator(
            params=params.default_params, 
            years=years, 
            n_simulations=n_simulations, 
            verbose=False,
            save_path=save_path
            )
        
        start_time = time.time()  # Start timing
        
        #df_simulation, df_weekly_stats = lcs.run_one_simulation()
        results = lcs.run_many_simulations()
        
        end_time = time.time()  # End timing
        total_time = end_time - start_time  # Calculate total time taken
        
        print(results)

        if results is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)

        logging.info('Successfully ran simulations.')
        
        # Calculate and log time per simulation-year
        time_per_simulation_year = total_time / (lcs.years * lcs.n_simulations)
        logging.info('Time per simulation-year: %f seconds', time_per_simulation_year)
    except Exception as e:  # Modified to catch and log the exception properly
        logging.error('Error running simulations: %s', e)

    # Merge DALY and symptom prevalence data with simulation data
    try:
        wlc = DataSimulationsMerger(results, df_symptom_integrals, data_daly)
        df_merged = wlc.calculate_welfare_loss()

        if df_merged is not None:
            with open('temp/df_merged.pkl', 'wb') as f:
                pickle.dump(df_merged, f)

        logging.info('Successfully merged data.')
    except Exception as e:
        logging.error('Error merging data: %s', e)

    # Tables and plots
    plots.plot_daly_adjustments(data_daly) # DALYs per symptom
    #plots.plot_all_symptoms(
    #    spe.trace, data_symptom_prevalence['symptom'].unique().tolist(), time_points=np.linspace(0, 18, 100)
    #    ) # Symptom prevalence, decay over time
    plots.plot_symptom_prevalence_over_time(df_symptom_integrals)
    plots.plot_symptom_years_histograms(df_symptom_integrals, num_subplots=3) # Symptom prevalence, total years
    # # Internal simulation outcomes over time
    plots.plot_daly_loss_over_time(df_merged, population_size_deflator) # Total welfare loss, over time

    # Robustness checks
    try:
        for param_name, param_values in params.robustness_params.items():
            for param_value in param_values:
                logging.info('Running robustness check for parameter %s with value %s', param_name, param_value)

                # Create a copy of the default parameters
                params_copy = params.default_params.copy()
                
                # Update the specific parameter value
                params_copy[param_name] = param_value

                # Modify population size for speed
                params_copy['size'] = int(round(params_copy['size'] / population_size_deflator))
                
                # Create a unique save path for each parameter configuration
                save_path = f"output/tables/robustness/{param_name}_{param_value}.csv"
                
                # Run the simulation with the updated parameters
                lcs = LongCovidSimulator(
                    params=params_copy, 
                    years=years, 
                    n_simulations=int(n_simulations/3), 
                    verbose=False,
                    save_path=save_path
                )
                results = lcs.run_many_simulations()
                print(results)

                wlc = DataSimulationsMerger(results, df_symptom_integrals, data_daly)
                df_merged = wlc.calculate_welfare_loss()
                
                # Save the merged DataFrame to a file
                df_merged.to_csv(save_path, index=False)
        
        logging.info('Successfully ran robustness checks.')
    except Exception as e:
        logging.error('Error running robustness checks: %s', e)

    logging.info('Data processing completed.')


if __name__ == '__main__':
    main()

