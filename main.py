from utils.process_daly_adjustments import DalyDataProcessor
from utils.process_symptom_prevalence import SymptomPrevalenceDataProcessor
from utils.estimate_symptom_prevalence_decay import SymptomPrevalenceEstimator
import utils.parameters as params
from utils.simulate_long_covid_cases import LongCovidSimulator
from utils.merge_data_with_simulations import DataSimulationsMerger
import utils.plots as plots

import pickle
import numpy as np
import logging

# Setup logging
logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Data processing started.')

def main():
    # Setup logging
    logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Data processing started.')

    # Process data
    ddp = DalyDataProcessor('data/daly.csv')
    data_daly = ddp.process_data()

    spdp = SymptomPrevalenceDataProcessor('data/prevalence_and_symptoms.csv')
    data_symptom_prevalence = spdp.process_data(adjustment_method='conservative')

    # Estimation
    try:
        spe = SymptomPrevalenceEstimator(data_symptom_prevalence)
        spe.trace = spe.setup_and_sample_model()
        df_symptom_integrals = spe.calculate_symptom_integrals()

        if spe.trace is not None:
            with open('temp/trace.pkl', 'wb') as f:
                pickle.dump(spe.trace, f)
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
            params.pessimistic_params, 
            params.param_descriptions
            )
        with open('output/tables/parameters.txt', 'w') as f:
            # Write headers
            f.write('\t'.join(comparison_table.columns) + '\n')

            # Write each row
            for index, row in comparison_table.iterrows():
                row_str = '\t'.join(str(x) for x in row.values)
                f.write(row_str + '\n')
    except:
        logging.error('Error generating comparison table: %s', e)
    try:
        save_path = 'temp/results.pkl'
        lcs = LongCovidSimulator(
            params=params.default_params, 
            years=5, 
            n_simulations=10, 
            verbose=False,
            save_path=save_path
            )
        #df_simulation, df_weekly_stats = lcs.run_one_simulation()
        results = lcs.run_many_simulations()
        print(results)

        if results is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)

        logging.info('Successfully ran simulations.')
    except:
        logging.error('Error running simulations: %s', e)

    # Merge DALY and symptom prevalence data with simulation data
    try:
        wlc = DataSimulationsMerger(results, df_symptom_integrals, data_daly)
        df_merged = wlc.calculate_welfare_loss()

        if df_merged is not None:
            with open('temp/df_merged.pkl', 'wb') as f:
                pickle.dump(df_merged, f)

        logging.info('Successfully merged data.')
    except:
        logging.error('Error merging data: %s', e)

    # Tables and plots
    plots.plot_daly_adjustments(data_daly) # DALYs per symptom
    plots.plot_all_symptoms(
        spe.trace, data_symptom_prevalence['symptom'].unique().tolist(), time_points=np.linspace(0, 18, 100)
        ) # Symptom prevalence, decay over time
    plots.plot_symptom_years_histograms(df_symptom_integrals, num_subplots=3) # Symptom prevalence, total years
    # # Internal simulation outcomes over time
    plots.plot_daly_loss_over_time(df_merged) # Total welfare loss, over time

    logging.info('Data processing completed.')


if __name__ == '__main__':
    main()

