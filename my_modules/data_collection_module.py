import pandas as pd


def read_data_pd(path):
    """
    Load the UCI Student Performance dataset from a CSV file.

    The dataset uses a semicolon (;) as the delimiter.
    """
    df = pd.read_csv(path, sep=";")
    return df


def clean_data(df):
    """
    Perform initial cleaning of the Student Performance dataset.

    - Remove duplicate rows
    - Remove leading and trailing spaces from column names
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    return df