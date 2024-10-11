import os
import pandas as pd
from datetime import datetime

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


def timestamp_to_nanoseconds(readable_timestamp):
    # Parse the readable timestamp into a datetime object
    # dt = datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    
    # Convert datetime to timestamp in seconds, then multiply by 1e9 to get nanoseconds
    nanoseconds = int(readable_timestamp.timestamp() * 1e9)
    
    return nanoseconds


def extract_info(folder_path):
    """
    Extract IDs and start/end times from image filenames in a folder.

    Parameters:
    folder_path (str): The path to the folder containing the image files.

    Returns:
    Tuple[dict, pd.Timestamp, pd.Timestamp]: A tuple containing a dictionary of IDs and times, and the start and end times.
    """
    file_data = []
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            # Extract ID and datetime from the filename
            date_str, time_str = filename.split('T')
            time_str = time_str.replace('Z', '')  # Remove the trailing Z
            id = date_str.split('_')[0]
            date = date_str.split('_')[-1]
            time_series = time_str.split('_')
            hour = time_series[0]
            minute = time_series[1]
            second_series = time_series[2].replace('.jpg', '').split('.')
            second = second_series[0]
            micro_sec = second_series[1]
            timestamp_full = f"{date} {hour}:{minute}:{second}.{micro_sec}"
            timestamp = datetime.strptime(timestamp_full, "%Y-%m-%d %H:%M:%S.%f")
            
            file_data.append({
                'id': id,
                'datetime': timestamp
            })
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(file_data)

    # Determine the start and end times
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()

    # Prepare the output dictionary with IDs and timestamps
    # ids_and_times = df[['id', 'datetime']].set_index('id').to_dict()['datetime']

    return id, start_time, end_time





def filter_by_time_range(df, start_time, end_time):
    """
    Remove rows from the DataFrame where the timestamp column does not fit within the given time range.
    """
    start_timestamp = timestamp_to_nanoseconds(start_time)
    end_timestamp = timestamp_to_nanoseconds(end_time)

    # Filter the DataFrame
    filtered_df = df[(df['timestamps'] >= start_timestamp) & (df['timestamps'] <= end_timestamp)]

    return filtered_df