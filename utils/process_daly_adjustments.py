import pandas as pd

class DalyDataProcessor:
    def __init__(self, file_path):
        """
        Initializes the DalyDataWrangler with the given DALY data.

        Parameters:
        data_daly (pd.DataFrame): DataFrame containing DALY data.
        """
        self.file_path = file_path
        self.data = None

    def process_data(self):
        """
        Processes the DALY data into the required format.

        Returns:
        pd.DataFrame: A processed DataFrame with selected columns.
        """
        self.data = self._load_data()
        self._rename_columns()
        self._validate_required_columns()
        self._fill_missing_daly_adjustments()
        self._add_missing_severity_rows()
        self._remove_invalid_rows()
        return self.data[['symptom', 'daly_adjustment', 'mild', 'moderate', 'severe']]

    def _load_data(self):
        """
        Loads data from the CSV file.

        Returns:
        pd.DataFrame: Data loaded from the CSV file.
        """
        return pd.read_csv(self.file_path)

    def _rename_columns(self):
        """
        Renames columns for consistency.
        """
        self.data.rename(columns={'name_merge_data': 'symptom', 'GHE2019': 'daly_adjustment', 'Health burden coefficient': 'health_burden_coefficient'}, inplace=True)

    def _validate_required_columns(self):
        """
        Validates if all required columns are present in the DataFrame.

        Raises:
        ValueError: If one or more required columns are missing.
        """
        required_columns = ['symptom', 'daly_adjustment', 'health_burden_coefficient', 'mild', 'moderate', 'severe']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError("One or more required columns are missing from the DALY data.")

    def _fill_missing_daly_adjustments(self):
        self.data['daly_adjustment'] = self.data['daly_adjustment'].fillna(self.data['health_burden_coefficient'])

    def _add_missing_severity_rows(self):
        new_rows = []
        for _, row in self.data.iterrows():
            existing_combinations = [(row['mild'], row['moderate'], row['severe'])]
            all_combinations = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

            missing_combinations = [comb for comb in all_combinations if comb not in existing_combinations]

            for comb in missing_combinations:
                new_row = row.copy()
                new_row['mild'], new_row['moderate'], new_row['severe'] = comb
                new_rows.append(new_row)

        self.data = pd.concat([self.data, pd.DataFrame(new_rows)], ignore_index=True)
        self.data = self.data.drop_duplicates(subset=['symptom', 'mild', 'moderate', 'severe'], keep='first')

    def _remove_invalid_rows(self):
        # Remove rows where daly_adjustment is NA
        self.data = self.data.dropna(subset=['daly_adjustment'])

        # Remove rows where mild, moderate, and severe are all zero
        self.data = self.data[~((self.data['mild'] == 0) & (self.data['moderate'] == 0) & (self.data['severe'] == 0))]

