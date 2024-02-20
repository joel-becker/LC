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
        Cleans and subsets the data for relevant columns.

        Parameters:
        data (pd.DataFrame): The raw data.

        Returns:
        pd.DataFrame: Cleaned and subsetted data.
        """
        columns = ['symptomatic', 'cohort_period', 'symptom', 'percentage_1st_period', 'percentage_2nd_period']
        return data[columns].dropna()

    def _calculate_prevalence_differences(self, cleaned_data):
        """
        Calculates prevalence differences for different periods.

        Parameters:
        cleaned_data (pd.DataFrame): The cleaned data.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Prevalence differences for 6 months and the second period.
        """
        grouped = cleaned_data.groupby(['symptom', 'cohort_period'])
        prevalence_diff_6m = grouped.apply(lambda x: self._calculate_mean_diff(x, 'percentage_1st_period')).reset_index(name='prevalence_diff_6m')
        prevalence_diff_2nd = grouped.apply(lambda x: self._calculate_mean_diff(x, 'percentage_2nd_period')).reset_index(name='prevalence_diff_2nd')
        return prevalence_diff_6m, prevalence_diff_2nd

    def _calculate_mean_diff(self, group, column):
        """
        Calculates the mean difference for a given group and column.

        Parameters:
        group (pd.DataFrame): Grouped data.
        column (str): The column to calculate the mean difference for.

        Returns:
        float: Mean difference.
        """
        symptomatic_mean = group[group['symptomatic'] == 1][column].mean()
        asymptomatic_mean = group[group['symptomatic'] == 0][column].mean()
        return symptomatic_mean - asymptomatic_mean

    def _merge_prevalence_data(self, prevalence_diff_6m, prevalence_diff_2nd):
        """
        Merges the 6-month and second period prevalence data.

        Parameters:
        prevalence_diff_6m (pd.DataFrame): Prevalence differences for 6 months.
        prevalence_diff_2nd (pd.DataFrame): Prevalence differences for the second period.

        Returns:
        pd.DataFrame: Merged prevalence data.
        """
        merged = pd.merge(prevalence_diff_6m, prevalence_diff_2nd, on=['symptom', 'cohort_period'])
        merged['months_2nd_period'] = merged['cohort_period'].apply(lambda x: 18 if '18' in x else 12)
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
        collapsed = merged_data.groupby('symptom').agg({'prevalence_diff_6m': 'first', 'prevalence_diff_12m': 'first', 'prevalence_diff_18m': 'first'}).reset_index()
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
        self.col_6m = 'prevalence_diff_6m'
        self.col_12m = 'prevalence_diff_12m'
        self.col_18m = 'prevalence_diff_18m'

    def adjust_data(self, method='conservative'):
        """
        Adjusts the prevalence data based on the chosen method.

        Parameters:
        method (str): Adjustment method, either 'conservative' or 'moderate'.

        Returns:
        pd.DataFrame: Adjusted prevalence data.
        """
        if method == 'conservative':
            return self._apply_conservative_adjustment()
        elif method == 'moderate':
            return self._apply_moderate_adjustment()
        else:
            raise ValueError("Method must be either 'conservative' or 'moderate'.")

    def _apply_conservative_adjustment(self):
        """
        Applies a conservative adjustment to the prevalence data.

        Returns:
        pd.DataFrame: Adjusted prevalence data.
        """
        adjusted = self.prevalence_data.copy()
        for col_higher, col_lower in [(self.col_18m, self.col_12m), 
                                      (self.col_12m, self.col_6m), 
                                      (self.col_18m, self.col_12m)]:
            adjusted[col_higher] = adjusted[[col_lower, col_higher]].min(axis=1)
        return adjusted

    def _apply_moderate_adjustment(self):
        """
        Applies a moderate adjustment to the prevalence data.

        Returns:
        pd.DataFrame: Adjusted prevalence data.
        """
        adjusted = self.prevalence_data.copy()
        adjusted = self._apply_mean_adjustment(adjusted)
        adjusted = self._ensure_non_decreasing_trend(adjusted)
        return adjusted.drop(columns=[
            'mean_all', 
            'mean_prevalence_diff_12m_prevalence_diff_18m', 
            'mean_prevalence_diff_6m_prevalence_diff_12m'
        ])

    def _apply_mean_adjustment(self, data):
        data['mean_all'] = data[[self.col_6m, self.col_12m, self.col_18m]].mean(axis=1)
        is_non_decreasing = (data[self.col_12m] >= data[self.col_6m]) & (data[self.col_18m] >= data[self.col_12m])
        data.loc[is_non_decreasing, [self.col_6m, self.col_12m, self.col_18m]] = data['mean_all']
        return data

    def _ensure_non_decreasing_trend(self, data):
        for pair in [(self.col_12m, self.col_18m), (self.col_6m, self.col_12m)]:
            mean_col = f'mean_{"_".join(pair)}'
            data[mean_col] = data[list(pair)].mean(axis=1)
            data.loc[data[pair[1]] >= data[pair[0]], pair] = data[mean_col]

        data.loc[data[self.col_18m] > data[self.col_12m], [self.col_6m, self.col_12m, self.col_18m]] = data['mean_all']
        return data

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

