import pandas as pd

def read_poi_data():
    """
    Read POIdata_cityD.csv and return a pandas DataFrame
    
    Returns:
        pd.DataFrame: DataFrame containing the POI data
    """
    df = pd.read_csv('data/POIdata_cityD.csv')
    return df