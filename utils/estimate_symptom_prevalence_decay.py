import pymc as pm
import pandas as pd
import numpy as np
from scipy.integrate import quad

class SymptomPrevalenceEstimator:
    def __init__(
            self, 
            data_symptom_prevalence, 
            hyperparameters={
                'baseline_alpha_hyperprior_mean': 1, 
                'baseline_alpha_hyperprior_var': 3, 
                'baseline_beta_hyperprior_mean': 50, 
                'baseline_beta_hyperprior_var': 10,
                'decayrate_mean_hyperprior_mean': 1, 
                'decayrate_mean_hyperprior_var': 10, 
                'decayrate_prior_var': 10
            },
            non_centered=False):
        self.data_symptom_prevalence = data_symptom_prevalence
        self.n_symptoms = len(data_symptom_prevalence['symptom'].unique())
        self.hyperparameters = hyperparameters
        self.non_centered = non_centered
        self.model = None
        self.trace = None

    @staticmethod
    def _calculate_gamma_params(mean, variance):
        alpha = mean ** 2 / variance
        theta = variance / mean
        return alpha, theta

    def setup_model(self, time, prevalence, symptom_idx, hyperparameters):
        # Set up hyperparameters
        baseline_alpha_alpha, baseline_alpha_theta = self._calculate_gamma_params(
            hyperparameters['baseline_alpha_hyperprior_mean'], hyperparameters['baseline_alpha_hyperprior_var']
            )
        baseline_beta_alpha, baseline_beta_theta = self._calculate_gamma_params(
            hyperparameters['baseline_beta_hyperprior_mean'], hyperparameters['baseline_beta_hyperprior_var']
            )
        decayrate_alpha_alpha, decayrate_alpha_theta = self._calculate_gamma_params(
            hyperparameters['decayrate_mean_hyperprior_mean'], hyperparameters['decayrate_mean_hyperprior_var']
            )
        
        with pm.Model() as self.model:
            # Set up priors and hyperpriors
            if self.non_centered == True:
                # Non-centered priors for baseline parameters
                baseline_alpha_offset = pm.Normal('baseline_alpha_offset', mu=0, sigma=1, shape=self.n_symptoms)
                baseline_beta_offset = pm.Normal('baseline_beta_offset', mu=0, sigma=1, shape=self.n_symptoms)

                # Transform to Gamma distribution
                baseline_alpha = pm.Deterministic('baseline_alpha', baseline_alpha_alpha + baseline_alpha_theta * baseline_alpha_offset)
                baseline_beta = pm.Deterministic('baseline_beta', baseline_beta_alpha + baseline_beta_theta * baseline_beta_offset)

                # Non-centered priors for decay rate
                decay_rate_offset = pm.Normal('decay_rate_offset', mu=0, sigma=1, shape=self.n_symptoms)

                # Transform to Gamma distribution
                decay_rate_alpha = pm.Deterministic('decay_rate_alpha', decayrate_alpha_alpha + decayrate_alpha_theta * decay_rate_offset)
            else:
                # Priors and hyperpriors for baseline parameters
                baseline_alpha = pm.Gamma('baseline_alpha', alpha=baseline_alpha_alpha, beta=1/baseline_alpha_theta, shape=self.n_symptoms)
                baseline_beta = pm.Gamma('baseline_beta', alpha=baseline_beta_alpha, beta=1/baseline_beta_theta, shape=self.n_symptoms)

                # Priors and hyperpriors for decay rate
                decay_rate_alpha = pm.Gamma('decay_rate_alpha', alpha=decayrate_alpha_alpha, beta=1/decayrate_alpha_theta, shape=self.n_symptoms)

            baseline = pm.Beta('baseline', alpha=baseline_alpha, beta=baseline_beta, shape=self.n_symptoms)
            decay_rate = pm.Gamma('decay_rate', alpha=decay_rate_alpha, beta=1/hyperparameters['decayrate_prior_var'], shape=self.n_symptoms)

            # Model for prevalence and Likelihood of observations
            prevalence_est = baseline[symptom_idx] * pm.math.exp(-decay_rate[symptom_idx] * time)
            Y_obs = pm.Normal('Y_obs', mu=prevalence_est, sigma=0.01, observed=prevalence)

    def sample_model(self, draws=1000, tune=500, chains=4, target_accept=0.99):
        with self.model:
            self.trace = pm.sample(draws, tune=tune, chains=chains, target_accept=target_accept)

        return self.trace
        
    @staticmethod
    def prepare_data(df):
        # Reshape data and convert to proportions
        melted_df = df.melt(id_vars=['symptom'], var_name='time', value_name='prevalence')
        melted_df['time'] = melted_df['time'].replace({'prevalence_diff_6m': 6, 'prevalence_diff_12m': 12, 'prevalence_diff_18m': 18})
        melted_df['prevalence'] = pd.to_numeric(melted_df['prevalence'], errors='coerce')
        melted_df['prevalence'] /= 100  # Convert to proportion
        symptom_idx = pd.Categorical(melted_df['symptom']).codes
        return melted_df['time'].values, melted_df['prevalence'].values, symptom_idx

    def setup_and_sample_model(self):
        time, prevalence, symptom_idx = self.prepare_data(self.data_symptom_prevalence)
        with pm.Model() as self.model:
            self.setup_model(time, prevalence, symptom_idx, self.hyperparameters)
            self.trace = self.sample_model()
        
        return self.trace

    def calculate_symptom_integrals(self, max_time=18):
        if self.trace is None:
            self.trace = self.setup_and_sample_model()

        # Flatten the samples from different chains into a single dimension
        baseline_samples = self.trace.posterior['baseline'].values.reshape(-1, self.n_symptoms)
        decay_rate_samples = self.trace.posterior['decay_rate'].values.reshape(-1, self.n_symptoms)

        highest_baseline_prevalence = np.max(baseline_samples, axis=0)

        symptom_integrals_list = []
        for i in range(len(baseline_samples)):
            row = []
            for j in range(self.n_symptoms):
                baseline = baseline_samples[i, j] / highest_baseline_prevalence[j]
                decay_rate = decay_rate_samples[i, j]
                integral, _ = quad(self._decay_func, 0, max_time, args=(baseline, decay_rate))

                # Convert integral to annual basis
                integral /= 12
                row.append(integral)
            symptom_integrals_list.append(row)

        # Convert list to DataFrame
        symptom_names = self.data_symptom_prevalence['symptom'].unique()
        symptom_integrals = pd.DataFrame(symptom_integrals_list, columns=symptom_names)
        return symptom_integrals

    @staticmethod
    def _decay_func(t, baseline, decay_rate):
        return baseline * np.exp(-decay_rate * t)
    

