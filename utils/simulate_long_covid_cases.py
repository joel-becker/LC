import pandas as pd
import numpy as np
from datetime import datetime
import squigglepy as sq

class Population:
    def __init__(
            self, 
            params = {
                'size': 330_000, 
                'baseline_risk': sq.norm(mean = 0.15, sd = 0.01), 
                'infection_rate': (sq.to(6687391, 40968288) / 330_000_000) / 52,
                'vaccination_reduction': sq.beta(100*0.25, 100*(1-0.25)), 
                'vaccination_interval': 180, 
                'vaccination_effectiveness_halflife': 1/365, 
                'covid_risk_reduction_factor': 0.7
            },
            verbose=True
            ):
        """
        Initialize the population DataFrame.
        """
        param_values = self.get_param_values(params)

        self.size = param_values['size']
        self.baseline_risk = param_values['baseline_risk']
        self.infection_rate = param_values['infection_rate']
        self.vaccination_reduction = param_values['vaccination_reduction']
        self.vaccination_interval = param_values['vaccination_interval']
        self.vaccination_effectiveness_halflife = param_values['vaccination_effectiveness_halflife']
        self.covid_risk_reduction_factor = param_values['covid_risk_reduction_factor']
    
        # Assign vaccination types
        vaccination_types = ['never_vaccinated', 'two_shots', 'boosted_yearly']
        vaccination_type_probabilities = [
            params['p_never_vaccinated'],
            params['p_two_shots'],
            params['p_boosted_yearly']
        ]

        self.verbose = verbose

        self.current_date = pd.Timestamp(datetime.now().date())

        self.data = pd.DataFrame({
            'individual_id': range(self.size),
            'covid_infections': np.zeros(self.size),
            'last_vaccination_date': np.full(self.size, self.current_date - pd.Timedelta(days=self.vaccination_interval)),
            'vaccination_type': np.random.choice(vaccination_types, size=self.size, p=vaccination_type_probabilities),
            'long_covid_risk': np.full(self.size, self.baseline_risk),
            'has_long_covid': np.zeros(self.size, dtype=bool)
        })

    @staticmethod
    def get_param_values(params):
        """Draw parameter values from distributions, else use scalar values."""
        param_values = {}
        for key, value in params.items():
            try:
                # Attempt to draw from a distribution
                param_values[key] = value @ 1
            except (TypeError, ValueError):
                # If not a distribution, use the scalar value
                param_values[key] = value
        return param_values
    
    def calculate_long_covid_risk(self, week_data):
        # Calculate the COVID risk adjustment based on the number of years passed
        years_passed = (week_data['week_start'] - self.current_date).days // 365
        covid_risk_adjustment = self.covid_risk_reduction_factor ** years_passed

        vaccination_adjustment = self.calculate_vaccination_adjustment(week_data)

        # Add this line
        self.data['covid_risk_adjustment'] = covid_risk_adjustment

        self.data['vaccination_adjustment'] = vaccination_adjustment

        # Multiply the baseline risk by the COVID risk adjustment
        adjusted_risk = self.baseline_risk * covid_risk_adjustment * vaccination_adjustment
        self.data['long_covid_risk'] = adjusted_risk

        # Determine Long COVID cases
        current_infections = self.data['last_infection_date'] == week_data['week_start']
        new_long_covid_cases = (np.random.rand(self.size) < adjusted_risk) & current_infections
        self.data['has_long_covid'] = new_long_covid_cases

        if self.verbose:
            print(f"Current infections: {current_infections.sum()}")
            print(f"Adjusted risk: {adjusted_risk.mean()}")
            print(f"Adjusted risk (current infections): {self.data[current_infections]['long_covid_risk'].mean()}")
            print(f"Vaccination adjustment (current infections): {self.data[current_infections]['vaccination_adjustment'].mean()}")
            print(f"New long COVID cases: {new_long_covid_cases.sum()}")

    def reset_long_covid_status(self):
        self.data['has_long_covid'] = False
        
    def calculate_vaccination_adjustment(self, week_data):
        # Initialize adjustment array with 1 (no reduction in risk)
        vaccination_adjustment = np.ones(self.size)

        # Calculate adjustment for individuals with 'two_shots'
        two_shots_mask = self.data['vaccination_type'] == 'two_shots'
        last_vaccination_dates = self.data.loc[two_shots_mask, 'last_vaccination_date']
        time_since_vaccination = (week_data['week_start'] - last_vaccination_dates).dt.days
        vaccination_decayrate = np.log(2) / self.vaccination_effectiveness_halflife
        vaccination_effectiveness = np.exp(-vaccination_decayrate * time_since_vaccination) * self.vaccination_reduction
        vaccination_adjustment[two_shots_mask] = 1 - vaccination_effectiveness

        # Calculate adjustment for individuals with 'boosted_yearly'
        boosted_yearly_mask = self.data['vaccination_type'] == 'boosted_yearly'
        vaccination_adjustment[boosted_yearly_mask] = 1 - self.vaccination_reduction

        return vaccination_adjustment


class Simulation:
    def __init__(self, population, verbose=True):
        self.population = population
        self.weekly_data = population.data
        self.size = population.size
        self.data = []
        self.current_date = pd.Timestamp(datetime.now().date())
        self.weekly_summary = []
        self.verbose = verbose

    def simulate_week(self, week_data):
        self.population.reset_long_covid_status()
        self.population.update_infection_status(week_data)
        self.population.calculate_long_covid_risk(week_data)
        self.record_weekly_statistics(week_data)

        # Take a snapshot of the population's data for this week
        data = self.weekly_data.copy()
        data['week_start'] = week_data['week_start']
        self.data.append(data)

    def record_weekly_statistics(self, week_data):
        weekly_cases = self.weekly_data['has_long_covid'].sum()
        avg_infections = self.weekly_data['covid_infections'].mean()
        avg_days_since_vaccination = (week_data['week_start'] - self.weekly_data['last_vaccination_date']).dt.days.mean()
        avg_long_covid_risk = self.weekly_data['long_covid_risk'].mean()

        #avg_aor_adjustment = self.weekly_data['aor_adjustment'].mean()
        avg_covid_risk_adjustment = self.weekly_data['covid_risk_adjustment'].mean()
        avg_vaccination_adjustment = self.weekly_data['vaccination_adjustment'].mean()

        self.weekly_summary.append({
            'week': week_data['week_start'],
            'new_long_covid_cases': weekly_cases,
            'average_infections': avg_infections,
            'average_days_since_last_vaccination': avg_days_since_vaccination,
            'average_long_covid_risk': avg_long_covid_risk,
            'average_covid_risk_adjustment': avg_covid_risk_adjustment,
            'average_vaccination_adjustment': avg_vaccination_adjustment,
        })

    def run(self, duration):
        for week in range(duration):
            week_start = self.current_date + pd.Timedelta(weeks=week)

            year = week // 52
            week = week % 52
            if self.verbose:
                print(f"Simulating week {week}, year {year}")

            week_data = {'week_start': week_start}
            self.simulate_week(week_data)
        
        self.data = pd.concat(self.data)

class LongCovidSimulator:
    def __init__(
            self, 
            params=None, 
            years=10, 
            n_simulations=300, 
            verbose=True,
            save_path = None
            ):
        self.params = params
        self.weeks_in_year = 52
        self.years = years
        self.n_simulations = n_simulations
        self.verbose = verbose
        self.save_path = save_path

    def run_one_simulation(self, summary=False):
        if self.params is not None:
            population = Population(params=self.params, verbose=self.verbose)
        else:
            population = Population(verbose=self.verbose)
        simulation = Simulation(population, verbose=self.verbose)
        simulation.run(self.weeks_in_year * self.years)
        df_weekly_summary = pd.DataFrame(simulation.weekly_summary)
        result = df_weekly_summary if summary else simulation.data

        # Save the result to a file
        if self.save_path is not None:
            with open(self.save_path, 'a') as f:
                result.to_csv(f, index=False)

        return result

    def run_many_simulations(self):
        results = []
        for simulation in range(self.n_simulations):
            if simulation % 1 == 0:
                print(f"Running simulation {simulation}")
            results.append(self.run_one_simulation())
        print("Done running simulations.")

        # Initialize an empty list to store the modified DataFrames
        modified_dataframes = []

        # Iterate over the results, adding a 'simulation' column
        for i, df in enumerate(results):
            df['simulation'] = i  # Add a simulation identifier column
            modified_dataframes.append(df)

        # Concatenate all DataFrames into one, while keeping the simulation identifier
        combined_dataframe = pd.concat(modified_dataframes)
        print("Done combining DataFrames.")

        return combined_dataframe