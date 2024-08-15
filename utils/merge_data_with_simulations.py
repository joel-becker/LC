import pandas as pd
import numpy as np

class DataSimulationsMerger:
    def __init__(self, df_simulation, df_symptom_integrals, data_daly):
        self.df_simulation = df_simulation
        self.df_symptom_integrals = df_symptom_integrals
        self.data_daly = data_daly

    def calculate_individual_severity_proportions(self, individual_characteristics):
        # Placeholder for actual logic to determine severity based on individual characteristics
        # For example, this might return {'mild': 0.5, 'moderate': 0.3, 'severe': 0.2}
        # Adjust this method based on your specific requirements and data
        return {'mild': 1, 'moderate': 0, 'severe': 0}

    def calculate_welfare_loss(self):
        long_covid_cases = self.df_simulation[self.df_simulation['has_long_covid']].copy()
        # Reset the index of long_covid_cases
        long_covid_cases.reset_index(drop=True, inplace=True)

        # Assuming calculate_individual_severity_proportions can be vectorized or simplified
        # Assuming a placeholder here that should be replaced with actual logic
        severity_df = pd.DataFrame([self.calculate_individual_severity_proportions(row) for index, row in long_covid_cases.iterrows()])
        
        # Prepare DALY matrix
        symptoms = self.data_daly['symptom'].unique()

        # Melt the DataFrame to reshape it for pivoting
        data_daly_melted = self.data_daly.melt(id_vars=['symptom', 'daly_adjustment'], value_vars=['mild', 'moderate', 'severe'], 
                                          var_name='severity', value_name='is_severity')

        # Filter out the rows where the severity indicator is 0
        data_daly_melted = data_daly_melted[data_daly_melted['is_severity'] == 1]

        # Drop the 'is_severity' as it's no longer needed
        data_daly_melted.drop('is_severity', axis=1, inplace=True)

        # Now pivot this DataFrame to create the severity columns with DALY adjustment values
        daly_matrix = data_daly_melted.pivot_table(index='symptom', columns='severity', values='daly_adjustment', aggfunc='first')

        # Sample symptom prevalences for each individual
        sample_indices = np.random.randint(0, self.df_symptom_integrals.shape[0], size=long_covid_cases.shape[0])
        sampled_symptoms = self.df_symptom_integrals.iloc[sample_indices].set_index(long_covid_cases.index)

        # Compute welfare loss
        for symptom in symptoms:
            # Calculate DALY adjustments for each symptom
            daly_adjustment = daly_matrix.loc[symptom].dot(severity_df[['mild', 'moderate', 'severe']].T)

            # Ensure indices are aligned
            daly_adjustment = daly_adjustment.reindex(long_covid_cases.index)

            # Calculate the welfare factor for the symptom
            welfare_factor = 1 - (daly_adjustment * sampled_symptoms[symptom])

            # Ensure indices are aligned
            welfare_factor = welfare_factor.reindex(long_covid_cases.index)

            # Aggregate welfare factors
            long_covid_cases['remaining_welfare'] = long_covid_cases.get('remaining_welfare', 1) * welfare_factor

        # Compute the total welfare loss and adjust by long COVID risk
        long_covid_cases['total_welfare_loss'] = 1 - long_covid_cases['remaining_welfare']
        long_covid_cases['DALY_loss'] = long_covid_cases['total_welfare_loss'] * long_covid_cases['long_covid_risk']
        
        return long_covid_cases

#    def calculate_welfare_loss(self):
#        # Filter for individuals with long COVID
#        long_covid_cases = self.df_simulation[self.df_simulation['has_long_covid']]
#
#        # Create a DataFrame to store the welfare loss for each individual
#        welfare_loss_df = pd.DataFrame(index=long_covid_cases.index)
#
#        processed_cases = 0  # Initialize a counter for the number of processed cases
#        for index, row in long_covid_cases.iterrows():
#            processed_cases += 1  # Increment the counter for each case processed
#            if processed_cases % 1000 == 0:
#                print(f"Processing {processed_cases} of {len(long_covid_cases)}")
#            remaining_welfare = 1  # Start with 100% welfare
#
#            # Get symptom prevalence integrals for this individual by random sampling
#            symptom_prevalences = self.df_symptom_integrals.sample(n=1).iloc[0]
#
#            # Calculate severity proportions for this individual
#            severity_proportions = self.calculate_individual_severity_proportions(row)
#
#            # Iterate over symptoms
#            for symptom in self.data_daly['symptom'].unique():
#                daly_adjustments = self.data_daly[self.data_daly['symptom'] == symptom]
#
#                # Calculate the weighted DALY adjustment for each symptom as a percentage
#                weighted_daly_percentage = sum(daly_adjustments['daly_adjustment'] * 
#                                    daly_adjustments[severity_proportions.keys()].mul(list(severity_proportions.values()), axis=0).sum(axis=1))
#
#                # Apply the DALY percentage loss to the remaining welfare
#                symptom_welfare_loss_factor = 1 - (weighted_daly_percentage * symptom_prevalences[symptom])
#                remaining_welfare *= symptom_welfare_loss_factor
#
#            # Calculate the total welfare loss, adjusted for long COVID risk
#            total_welfare_loss = 1 - remaining_welfare
#            long_covid_cases.at[index, 'DALY_loss'] = total_welfare_loss * long_covid_cases.at[index, 'long_covid_risk']
#
#        return long_covid_cases
#