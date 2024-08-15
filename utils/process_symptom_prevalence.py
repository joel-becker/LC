import pandas as pd

class SymptomPrevalenceDataPreparer:
    def __init__(self, file_path):
        """
        Initializes the DataPreparer with the path to the prevalence and symptoms data file.

        Parameters:
        file_path (str): Path to the CSV file containing prevalence and symptoms data.
        """
        self.file_path = file_path

    def prepare_data(self):
        """
        Prepares the raw data by cleaning, calculating, merging, and collapsing it.

        Returns:
        pd.DataFrame: Collapsed and prepared data.
        """
        data = self._load_data()
        cleaned_data = self._clean_and_subset_data(data)
        prevalence_diff_6m, prevalence_diff_2nd = self._calculate_prevalence_differences(cleaned_data)
        merged_data = self._merge_prevalence_data(prevalence_diff_6m, prevalence_diff_2nd)
        collapsed_data = self._collapse_prevalence_data(merged_data)
        return collapsed_data

    def _load_data(self):
        """
        Loads data from the CSV file.

        Returns:
        pd.DataFrame: Data loaded from the CSV file.
        """
        return pd.read_csv(self.file_path)

    def _clean_and_subset_data(self, data):
        """
        Cleans and subsets the data for relevant columns, adjusts pvalue column values, and converts it to float.

        Parameters:
        data (pd.DataFrame): The raw data.

        Returns:
        pd.DataFrame: Cleaned and subsetted data.
        """
        columns = ['symptomatic', 'cohort_period', 'symptom', 'percentage_1st_period', 'percentage_2nd_period', 'pvalue']
        subset_data = data[columns].dropna()
        subset_data['pvalue'] = subset_data['pvalue'].replace('<0.001', '0.001').astype(float)
        return subset_data

    def _calculate_prevalence_differences(self, cleaned_data):
        """
        Calculates prevalence differences for different periods.
    
        Parameters:
        cleaned_data (pd.DataFrame): The cleaned data.
    
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Prevalence differences for 6 months and the second period.
        """
        grouped = cleaned_data.groupby(['symptom', 'cohort_period'])
        prevalence_diff_6m = grouped.apply(lambda x: self._calculate_mean_diff(x, 'percentage_1st_period', 'pvalue')).reset_index()
        prevalence_diff_2nd = grouped.apply(lambda x: self._calculate_mean_diff(x, 'percentage_2nd_period', 'pvalue')).reset_index()
        return prevalence_diff_6m, prevalence_diff_2nd
    
    def _calculate_mean_diff(self, group, column, pvalue_column):
        """
        Calculates the mean difference for a given group and column.
    
        Parameters:
        group (pd.DataFrame): Grouped data.
        column (str): The column to calculate the mean difference for.
        pvalue_column (str): The column containing p-values.
    
        Returns:
        pd.Series: Mean difference and p-value.
        """
        symptomatic_mean = group[group['symptomatic'] == 1][column].mean()
        asymptomatic_mean = group[group['symptomatic'] == 0][column].mean()
        pvalue = group[group['symptomatic'] == 1][pvalue_column].values[0]
        return pd.Series({'prevalence_diff': symptomatic_mean - asymptomatic_mean, 'pvalue': pvalue})

    def _merge_prevalence_data(self, prevalence_diff_6m, prevalence_diff_2nd):
        """
        Merges the 6-month and second period prevalence data.
    
        Parameters:
        prevalence_diff_6m (pd.DataFrame): Prevalence differences for 6 months.
        prevalence_diff_2nd (pd.DataFrame): Prevalence differences for the second period.
    
        Returns:
        pd.DataFrame: Merged prevalence data.
        """
        merged = pd.merge(prevalence_diff_6m, prevalence_diff_2nd, on=['symptom', 'cohort_period'], suffixes=('_6m', '_2nd'))
        merged['months_2nd_period'] = merged['cohort_period'].apply(lambda x: 18 if '18' in x else 12)
        merged.rename(columns={'prevalence_diff_6m': 'prevalence_diff_6m', 'prevalence_diff_2nd': 'prevalence_diff_2nd', 'pvalue_6m': 'pvalue'}, inplace=True)
        return merged

    def _collapse_prevalence_data(self, merged_data):
        """
        Collapses the prevalence data to a single row per symptom.

        Parameters:
        merged_data (pd.DataFrame): Merged prevalence data.

        Returns:
        pd.DataFrame: Collapsed prevalence data.
        """
        mean_prevalence_diff_6m = merged_data.groupby('symptom')['prevalence_diff_6m'].mean().reset_index()
        merged_data = merged_data.merge(mean_prevalence_diff_6m, on='symptom', suffixes=('', '_mean'))
        merged_data['prevalence_diff_6m'] = merged_data['prevalence_diff_6m_mean']
        merged_data = self._separate_and_drop_columns(merged_data)
        collapsed = merged_data.groupby('symptom').agg({
            'prevalence_diff_6m': 'first', 'prevalence_diff_12m': 'first', 'prevalence_diff_18m': 'first', 'pvalue': 'first'
            }).reset_index()
        return collapsed

    def _separate_and_drop_columns(self, merged_data):
        """
        Separates and drops unnecessary columns from the merged data.

        Parameters:
        merged_data (pd.DataFrame): Merged prevalence data.

        Returns:
        pd.DataFrame: Data with separated and dropped columns.
        """
        merged_data['prevalence_diff_12m'] = merged_data.apply(lambda row: row['prevalence_diff_2nd'] if row['months_2nd_period'] == 12 else None, axis=1)
        merged_data['prevalence_diff_18m'] = merged_data.apply(lambda row: row['prevalence_diff_2nd'] if row['months_2nd_period'] == 18 else None, axis=1)
        return merged_data.drop(columns=['cohort_period', 'months_2nd_period', 'prevalence_diff_2nd', 'prevalence_diff_6m_mean'])

class SymptomPrevalenceDataAdjuster:
    def __init__(self, prevalence_data):
        """
        Initializes the PrevalenceDataAdjuster with the given data.

        Parameters:
        prevalence_data (pd.DataFrame): DataFrame containing prevalence data.
        """
        self.prevalence_data = prevalence_data

    def adjust_data(self):
        """
        Adjusts the prevalence data based on the revised approach.

        Returns:
        pd.DataFrame: Adjusted prevalence data with steady-state values.
        """
        adjusted_data = self.prevalence_data.copy()
        adjusted_data['steady_state_value'] = adjusted_data.apply(self._determine_steady_state, axis=1)
        return adjusted_data

    def _determine_steady_state(self, row):
        """
        Determines the steady-state value for a given symptom based on the revised approach.

        Parameters:
        row (pd.Series): A row of the prevalence data DataFrame.

        Returns:
        float: The steady-state value for the symptom.
        """
        if row['prevalence_diff_18m'] > row['prevalence_diff_6m'] and row['pvalue'] < 0.05:
            return row['prevalence_diff_18m']
        else:
            return row['prevalence_diff_18m']
        
class SymptomPrevalenceDataProcessor:
    def __init__(self, file_path):
        """
        Initializes the DataWorkflowManager with the path to the data file.

        Parameters:
        file_path (str): Path to the CSV file containing the data.
        """
        self.file_path = file_path

    def process_data(self, adjustment_method='conservative'):
        """
        Processes the data by preparing and then adjusting it.

        Parameters:
        adjustment_method (str): The method to use for adjusting the data ('conservative' or 'moderate').

        Returns:
        pd.DataFrame: The processed data.
        """
        preparer = SymptomPrevalenceDataPreparer(self.file_path)
        prepared_data = preparer.prepare_data()

        adjuster = SymptomPrevalenceDataAdjuster(prepared_data)
        adjusted_data = adjuster.adjust_data(method=adjustment_method)

        return adjusted_data

class SymptomPrevalenceDataAdjuster:
    def __init__(self, prevalence_data):
        """
        Initializes the PrevalenceDataAdjuster with the given data.

        Parameters:
        prevalence_data (pd.DataFrame): DataFrame containing prevalence data.
        """
        self.prevalence_data = prevalence_data

    def adjust_data(self):
        """
        Adjusts the prevalence data based on the revised approach.

        Returns:
        pd.DataFrame: Adjusted prevalence data with steady-state values.
        """
        adjusted_data = self.prevalence_data.copy()
        adjusted_data['steady_state_value'] = adjusted_data.apply(self._determine_steady_state, axis=1)
        adjusted_data['prevalence_diff_18m'] = adjusted_data['steady_state_value']
        return adjusted_data

    def _determine_steady_state(self, row):
        """
        Determines the steady-state value for 18 months based on specific criteria.
        - Use the value for 18 months if it's higher than 6 months and p-value is less than 0.05,
          or if 18 months is lower than 6 months.
        - If 18 months is higher than 6 months but p-value is 0.05 or more, use the average of 6, 12, and 18 months values.
        
        Parameters:
        row (pd.Series): A row of the prevalence data DataFrame.
        
        Returns:
        float: The steady-state value for 18 months.
        """
        if row['prevalence_diff_18m'] > row['prevalence_diff_6m'] and row['pvalue'] < 0.05:
            return row['prevalence_diff_18m']
        elif row['prevalence_diff_18m'] < row['prevalence_diff_6m']:
            return row['prevalence_diff_18m']
        elif row['prevalence_diff_18m'] > row['prevalence_diff_6m'] and row['pvalue'] >= 0.05:
            return (row['prevalence_diff_6m'] + row['prevalence_diff_12m'] + row['prevalence_diff_18m']) / 3
        else:
            return row['prevalence_diff_18m']
class SymptomPrevalenceDataProcessor:
    def __init__(self, file_path):
        """
        Initializes the DataWorkflowManager with the path to the data file.

        Parameters:
        file_path (str): Path to the CSV file containing the data.
        """
        self.file_path = file_path

    def process_data(self):
        """
        Processes the data by preparing and then adjusting it.

        Returns:
        pd.DataFrame: The processed data with steady-state values.
        """
        preparer = SymptomPrevalenceDataPreparer(self.file_path)
        prepared_data = preparer.prepare_data()

        adjuster = SymptomPrevalenceDataAdjuster(prepared_data)
        adjusted_data = adjuster.adjust_data()

        return adjusted_data
    