import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


##Convert hdf5 to readable dataframe
def dataframe_to_arrow(df):
    table = pa.Table.from_pandas(df)
    columns = [c.cast(pa.float32()) if c.type == pa.float64() else c for c in table]
    return pa.Table.from_arrays(columns, table.column_names)

def generate_parquet(path):
    hdf5_path = path + 'vehicle_data.hdf5'
    hdf5_name = Path(hdf5_path).stem

    #convert hdf5 and append to dict
    if type(hdf5_path) is str and ".hdf5" in hdf5_path:
        dataset_list = []

        with h5py.File(hdf5_path, 'r') as file:
            def get_data(name, object):
                is_dataset = isinstance(object, h5py.Dataset)
                if is_dataset: ##Filter
                    dataset_list.append(name)
            file.visititems(get_data)

            vehicle_control_data = {
                ds: np.array(file[ds]).astype(np.float64)
                for ds in dataset_list
                if "ego_vehicle_controls" in ds
            }
            vehicle_data = {
                ds: np.array(file[ds]).astype(np.float64)
                for ds in dataset_list
                if "ego_vehicle_data" in ds
            }
            satellite_data = {
                ds: np.array(file[ds]).astype(np.float64)
                for ds in dataset_list
                if "satellite" in ds
            }

    for hdf5_name, data in zip(
        ['vehicle_control_data', 'vehicle_data', 'satellite_data'],
        [vehicle_control_data, vehicle_data, satellite_data],
    ):
        filename = f"{path}{hdf5_name}.parquet"

        pq.write_table(
            dataframe_to_arrow(pd.DataFrame(data)),
            filename,
            compression='gzip',
        )
        print(f"Parquet saved {filename}")
    print(f"{hdf5_path} done!")

if __name__ == "__main__":
    data_path="data/signal/"
    dic_name = [str(i).zfill(6) for i in range(0, 1473)]
    for f in dic_name:
        generate_parquet("data/signal/" + f + "/")