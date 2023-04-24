import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path, PosixPath
from typing import List

def read_data(csv_path):
    """
    Read data from a csv file and return a pandas dataframe.
    """
    df = pd.read_csv(csv_path)
    return df

def reduce_memory_usage(df):
    """
    Reduce memory usage of a pandas dataframe.
    """
    int_cols = df.select_dtypes(include=['int']).columns
    float_cols = df.select_dtypes(include=['float']).columns
    object_cols = df.select_dtypes(include=['object']).columns

    df[int_cols] = df[int_cols].astype(np.int16)
    df[float_cols] = df[float_cols].astype(np.float16)
    df[object_cols] = df[object_cols].astype('category')

    return df


def remove_missing(df):
    """
    Remove missing values from a pandas dataframe.
    """
    df = df.dropna(subset='alert')
    return df

def remove_features(df, features):
    """
    Remove features from a pandas dataframe.
    """
    df = df.drop(features, axis=1)
    return df


def prepare_data(df_path: PosixPath, features_to_remove: List[str]):
    """
    Prepare data for training.
    1. Read data from csv file
    2. Reduce memory usage
    3. Remove missing values
    4. Remove features
    5. Reorder columns
    """
    col_order = ['magnitude', 'cdi', 'mmi', 'tsunami', 'sig', 'nst', 'dmin', 'gap', 'depth', 'alert']
    
    df = read_data(df_path)
    df = reduce_memory_usage(df)
    df = remove_missing(df)
    df = remove_features(df, features_to_remove)
    df = df[col_order]
    
    return df


# quakes_path = Path.cwd() / 'data' / 'earthquake-2001-2023.csv'
# earthquakes = read_data(quakes_path)

# # print(earthquakes.info())
# print(f'Initial memory size in bytes: {earthquakes.memory_usage(deep=True).sum()}')
# earthquakes = reduce_memory_usage(earthquakes)
# print(f'Final memory size in bytes: {earthquakes.memory_usage(deep=True).sum()}')

# earthquakes_no_missing = remove_missing(earthquakes)

# features_to_remove = ['title', 'date_time', 'magType', 
#                       'location', 'continent', 'country']

# earthquakes_clean = remove_features(earthquakes_no_missing, features_to_remove)
