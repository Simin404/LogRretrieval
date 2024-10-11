import geopy
import pandas as pd

from geopy.geocoders import Nominatim

def get_location(latitude, longitude):
    geolocator = Nominatim(user_agent="my-app")
    location = geolocator.reverse((latitude.item(), longitude.item()))
    return location

def timestamp_to_datetime(nanosecond_timestamp):
    """
    Convert a timestamp in nanoseconds to a readable datetime.

    Parameters:
    nanosecond_timestamp (int): The timestamp in nanoseconds since the Unix epoch.

    Returns:
    pd.Timestamp: The readable datetime.
    """
    # Convert nanoseconds to seconds
    second_timestamp = nanosecond_timestamp / 1e9
    
    # Convert to a readable datetime
    readable_datetime = pd.to_datetime(second_timestamp, unit='s')
    
    return readable_datetime


def interpreter(df, path):
    columns_name = list(df.columns)
    text = ''
    for name in columns_name:
        if name == 'satellite/latposn/nanodegrees/value':
            start = get_location(df['satellite/latposn/nanodegrees/value'].iloc[0]/1000000000, df['satellite/longposn/nanodegrees/value'].iloc[0]/1000000000)
            end = get_location(df['satellite/latposn/nanodegrees/value'].iloc[-1]/1000000000, df['satellite/longposn/nanodegrees/value'].iloc[-1]/1000000000)
            text += 'Departure address is {}, Arrival address is {};'.format(start, end)
        if name == 'satellite/speed/meters_per_second/value':
            min, max = df[name].agg(['min', 'max'])
            a = 'Value range of satellite speed in meters per second is [{},{}];'.format(min, max)
            text += a
        if name == 'timestamp':
            text += 'The start time is {}, the end time is {};'.format(timestamp_to_datetime(df[name].iloc[0]), timestamp_to_datetime(df[name].iloc[-1]))
        if name == 'ego_vehicle_data/body_pitch/angle/radians/value':
            min, max = df[name].agg(['min', 'max'])
            if min < 0:
                text += 'deceleration, braking, or driving downhill'
            else:
                text += 'acceleration or driving uphill'
        if name == 'ego_vehicle_data/lat_acc_data/acceleration/meters_per_second2/value':
            min, max = df[name].agg(['min', 'max'])
            a = 'Value range of vehicle latitude acceleration in meters per second squared is [{0:.2f},{0:.2f}];'.format(min, max)
            text += a
        if name == 'ego_vehicle_data/lat_vel_data/velocity/meters_per_second/value':
            min, max = df[name].agg(['min', 'max'])
            a = 'Value range of vehicle latitude velocity in meters per second is [{0:.2f},{0:.2f}];'.format(min, max)
            text += a
        if name == 'ego_vehicle_data/lon_acc_data/acceleration/meters_per_second2/value':
            min, max = df[name].agg(['min', 'max'])
            a = 'Value range of vehicle longtitude acceleration in meters per second squared is [{0:.2f},{0:.2f}];'.format(min, max)
            text += a
        if name == 'ego_vehicle_data/lon_acc_data/velocity/meters_per_second/value':
            min, max = df[name].agg(['min', 'max'])
            a = 'Value range of vehicle longtitude velocity in meters per second is [{0:.2f},{0:.2f}];'.format(min, max)
            text += a
        if name == 'ego_vehicle_data/roll_rate_data/angle_rate/radians_per_second/value':
            min, max = df[name].agg(['min', 'max'])
            a = 'Value range of vehicle roll rate in radians per second is [{0:.2f},{0:.2f}];'.format(min, max)
            text += a
        if name == 'ego_vehicle_data/vertical_acc_data/acceleration//value':
            min, max = df[name].agg(['min', 'max'])
            a = 'Value range of vehicle vertical acceleration in meters per second squared is [{0:.2f},{0:.2f}];'.format(min, max)
            text += a   
        if name == 'ego_vehicle_controls/acceleration_pedal/ratio/unitless/value':
            min, max = df[name].agg(['min', 'max'])
            if min > 20:
                a = 'Vehicle acceleration pedal is pressed and vehicle is increasing acceleration;'
            text += a
        if name == 'ego_vehicle_controls/brake_pedal_pressed/is_brake_pedal_pressed/unitless/value':
            min, max = df[name].agg(['min', 'max'])
            if min > 10:
                a = 'Vehicle brake pedal is pressed and vehicle is increasing braking force;'
            text += a
        if name == 'ego_vehicle_controls/steering_wheel_angle/angle/radians/value':
            min, max = df[name].agg(['min', 'max'])
            if max < 0:
                text += 'left turn'
            else:
                text += 'right turn'
            text += a
    print(path)
    f = open(path, "w")
    f.write(text)
    f.close()