import pandas as pd
import os

class DataProcessor:
    def __init__(self, folder_path):
        """
        Initialize the DataProcessor class with the folder path containing data files.

        Parameters:
        folder_path (str): The path to the folder containing the data files.
        """
        self.folder_path = folder_path

    def remove_unchanged(self, df):
        """
        Remove columns in the DataFrame where all values are the same.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with unchanged columns removed.
        """
        dropped = []
        for c in list(df):
            a = df[c].unique()
            if len(a) == 1:
                print(c, 'ALL', a, ': REMOVED')
                dropped.append(c)
        print('Total Deleted: ', len(dropped))
        print('Original shape:', df.shape)
        df = df.drop(columns=dropped)
        print('After shape:', df.shape)
        return df

    def combine_column(self, df_sd):
        """
        Combine date and time columns into a single datetime column.

        Parameters:
        df_sd (pd.DataFrame): The satellite data DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with combined datetime column.
        """
        df = pd.DataFrame({
            'year': df_sd['satellite/utctime/year/year/value'],
            'month': df_sd['satellite/utctime/month/month/value'],
            'day': df_sd['satellite/utctime/day/day/value'],
            'hour': df_sd['satellite/utctime/hour/hour/value'],
            'minute': df_sd['satellite/utctime/minute/minute/value'],
            'second': df_sd['satellite/utctime/sec/second/value']
        })
        df_sd['satellite/utctime'] = pd.to_datetime(df)
        df_sd = df_sd.drop(columns=[
            'satellite/utctime/year/year/value',
            'satellite/utctime/month/month/value',
            'satellite/utctime/day/day/value',
            'satellite/utctime/hour/hour/value',
            'satellite/utctime/minute/minute/value',
            'satellite/utctime/sec/second/value'
        ])
        return df_sd

    def read_file(self):
        """
        Read parquet files from the specified folder.

        Returns:
        Tuple[pd.DataFrame]: A tuple containing the satellite, vehicle control, and vehicle data DataFrames.
        """
        df_vcd = pd.read_parquet(os.path.join(self.folder_path, "vehicle_control_data.parquet"))
        df_vd = pd.read_parquet(os.path.join(self.folder_path, "vehicle_data.parquet"))
        df_sd = pd.read_parquet(os.path.join(self.folder_path, "satellite_data.parquet"))

        print('The shape of vehicle control data:', df_vcd.shape)
        print('The shape of vehicle data:', df_vd.shape)
        print('The shape of satellite data:', df_sd.shape)
        
        return self.combine_column(df_sd), df_vcd, df_vd

    def manual_clean(self, df_sd, df_vcd, df_vd):
        """
        Clean the data by removing specific columns and replacing values.

        Parameters:
        df_sd (pd.DataFrame): The satellite data DataFrame.
        df_vcd (pd.DataFrame): The vehicle control data DataFrame.
        df_vd (pd.DataFrame): The vehicle data DataFrame.

        Returns:
        Tuple[pd.DataFrame]: A tuple containing cleaned DataFrames.
        """
        # Drop uninterpretable satellite data columns
        df_sd = df_sd.drop(columns=[
            'satellite/altitude/meters/value',
            'satellite/dilution_of_precision/horizontal/unitless/value',
            'satellite/dilution_of_precision/position/unitless/value',
            'satellite/dilution_of_precision/time/unitless/value',
            'satellite/dilution_of_precision/vertical/unitless/value',
            'satellite/nrof_satellites/unitless/value',
            'satellite/heading/degrees/value'
        ])

        # Drop unnecessary vehicle control data columns
        df_vcd = df_vcd.drop(columns=[
            'ego_vehicle_controls/steering_wheel_angle/angle_rate/radians_per_second/value',
            'ego_vehicle_controls/turn_indicator_status/state',
            'ego_vehicle_controls/steer_wheel_torque/torque/newton_meters/value'
        ])

        # Replace year value in satellite data
        df_sd = df_sd.replace({'satellite/utctime/year/year/value': 22.0}, {'satellite/utctime/year/year/value': 2022}, regex=True)

        # Remove unchanged columns
        return self.remove_unchanged(df_sd), self.remove_unchanged(df_vcd), self.remove_unchanged(df_vd)

    def join_table(self, df_sd, df_vcd, df_vd):
        """
        Join the cleaned data tables on their timestamp columns.

        Parameters:
        df_sd (pd.DataFrame): The cleaned satellite data DataFrame.
        df_vcd (pd.DataFrame): The cleaned vehicle control data DataFrame.
        df_vd (pd.DataFrame): The cleaned vehicle data DataFrame.

        Returns:
        pd.DataFrame: The combined DataFrame.
        """
        # Rename timestamp columns for merging
        df_vcd = df_vcd.rename(columns={'ego_vehicle_controls/timestamp/nanoseconds/value': 'timestamps'})
        df_vd = df_vd.rename(columns={'ego_vehicle_data/timestamp/nanoseconds/value': 'timestamps'})
        df_sd = df_sd.rename(columns={'satellite/timestamp/nanoseconds/value': 'timestamps'})

        print('The shape of vehicle control data:', df_vcd.shape)
        print('The shape of vehicle data:', df_vd.shape)
        print('The shape of satellite data:', df_sd.shape)

        # Merge the dataframes based on timestamps
        combined = pd.merge_asof(df_sd, df_vd, on="timestamps", direction='backward')
        downsized_df = pd.merge_asof(combined, df_vcd, on="timestamps", direction='backward')
        
        print('The shape of combined table:', downsized_df.shape)
        
        return downsized_df

    def process_data(self):
        """
        Complete data processing pipeline: read, clean, and join the data.

        Returns:
        pd.DataFrame: The combined DataFrame after processing.
        """
        # Read files
        df_sd, df_vcd, df_vd = self.read_file()
        
        # Clean the data
        df_sd, df_vcd, df_vd = self.manual_clean(df_sd, df_vcd, df_vd)
        
        # Join the tables
        combined_df = self.join_table(df_sd, df_vcd, df_vd)
        
        return combined_df


