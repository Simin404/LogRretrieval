import pandas as pd

## Remove unchanged columns
def remove_unchanged(df):
    dropped = []
    for c in list(df):
        a = df[c].unique()
        if len(a) == 1:
            # print(c, 'ALL', a,': REMOVED')
            dropped.append(c)
    # print('Total Deleted: ', len(dropped))
    # print('Original shape:', df.shape)
    df = df.drop(columns=dropped)
    # print('After shape:', df.shape)
    return df


def combine_column(df_sd):
    ## those 6 columns can be interpreted by using timestamp
    # df = pd.DataFrame({'year': df_sd['satellite/utctime/year/year/value'],
    #                'month': df_sd['satellite/utctime/month/month/value'],
    #                'day': df_sd['satellite/utctime/day/day/value'],
    #                'hour':df_sd['satellite/utctime/hour/hour/value'],
    #                'minute':df_sd['satellite/utctime/minute/minute/value'],
    #                'second':df_sd['satellite/utctime/sec/second/value']
    #                })
    # df_sd['satellite/utctime'] = pd.to_datetime(df)
    df_sd = df_sd.drop(columns=['satellite/utctime/year/year/value', 'satellite/utctime/month/month/value','satellite/utctime/day/day/value', 'satellite/utctime/hour/hour/value', 'satellite/utctime/minute/minute/value', 'satellite/utctime/sec/second/value'])
    return df_sd

def read_file(folder_path):
    df_vcd = pd.read_parquet(folder_path + "/vehicle_control_data.parquet")
    df_vd = pd.read_parquet(folder_path + "/vehicle_data.parquet")
    df_sd = pd.read_parquet(folder_path + "/satellite_data.parquet")

    # print('The shape of vehicle control data:', df_vcd.shape)
    # print('The shape of vehicle data:', df_vd.shape)
    # print('The shape of satellite control data:', df_sd.shape)
    return df_sd, df_vcd, df_vd


def manual_clean(df_sd, df_vcd, df_vd):
    ## Manual drop some uninterpretable satellite data
    df_sd = df_sd.drop(columns=['satellite/altitude/meters/value', 'satellite/dilution_of_precision/horizontal/unitless/value', 'satellite/dilution_of_precision/position/unitless/value', 'satellite/dilution_of_precision/time/unitless/value', 'satellite/dilution_of_precision/vertical/unitless/value', 'satellite/nrof_satellites/unitless/value', 'satellite/heading/degrees/value'])
    df_vcd = df_vcd.drop(columns=['ego_vehicle_controls/steering_wheel_angle/angle_rate/radians_per_second/value', 'ego_vehicle_controls/turn_indicator_status/state', 'ego_vehicle_controls/steer_wheel_torque/torque/newton_meters/value'])
    return  remove_unchanged(df_sd), remove_unchanged(df_vcd), remove_unchanged(df_vd)

def join_table(df_sd, df_vcd, df_vd):
    df_vcd = df_vcd.rename(columns={'ego_vehicle_controls/timestamp/nanoseconds/value': 'timestamps'})
    df_vd = df_vd.rename(columns={'ego_vehicle_data/timestamp/nanoseconds/value': 'timestamps'})
    df_sd = df_sd.rename(columns={'satellite/timestamp/nanoseconds/value': 'timestamps'})

    # print('The shape of vehicle control data:', df_vcd.shape)
    # print('The shape of vehicle data:', df_vd.shape)
    # print('The shape of satellite control data:', df_sd.shape)
    combined = pd.merge_asof(df_sd, df_vd, on="timestamps", direction= 'backward')
    downsized_df = pd.merge_asof(combined, df_vcd, on="timestamps", direction= 'backward')
    # print('The shape of combined table:', downsized_df.shape) 
    return downsized_df

def process_data(folder_path):
    df_sd, df_vcd, df_vd = read_file(folder_path)
    # df_sd, df_vcd, df_vd = manual_clean(df_sd, df_vcd, df_vd)
    combined_df = join_table(df_sd, df_vcd, df_vd)
    
    return combined_df


