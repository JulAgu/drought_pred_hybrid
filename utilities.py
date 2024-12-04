import os
import pandas as pd

def create_dataFrames(srcPath):
    """
    This function creates dataFrames from the csv files in the srcPath.
        
        Parameters:
            srcPath (str): The path to the csv files.

        Returns:
            dataDic (dict): A dictionary with the dataFrames.    
    """
    if not os.path.exists("data/raw_dataFrames"):
        os.makedirs("data/raw_dataFrames")

    dataDic = {"train": pd.read_csv("../src/train_timeseries/train_timeseries.csv"),
               "test": pd.read_csv("../src/test_timeseries/test_timeseries.csv"),
               "validation": pd.read_csv("../src/validation_timeseries/validation_timeseries.csv"),
               "soil" : pd.read_csv("../src/soil_data.csv"),
               }
    
    return dataDic

def load_XY(
        df,
    random_state=42,
    window_size=180, # how many days in the past (default/competition: 180)
    target_size=6, # how many weeks into the future (default/competition: 6)
    fuse_past=True, # add the past drought observations? (default: True)
    return_fips=False, # return the county identifier (do not use for predictions)
    encode_season=True, # encode the season using the function above (default: True) 
    use_prev_year=False, # add observations from 1 year prior?
):
    """
    A function to load the data and create the X and y arrays.
    Taken from https://www.pure.ed.ac.uk/ws/portalfiles/portal/217133242/DroughtED_MINIXHOFER_DOA18062021_AFV.pdf

        Parameters:
            df (pd.DataFrame): The dataFrame to load the data from.
            random_state (int): The random state to use.
            window_size (int): The number of days in the past used for prediction.
            target_size (int): The number of weeks into the future (the size of the output vector).
            fuse_past (bool): Add the past drought observations.
            return_fips (bool): Return the county identifier.
            encode_season (bool): Encode the season.
            use_prev_year (bool): Add observations from 1 year prior.
        
        Returns:
            X (np.array): The input array.
            y (np.array): The output array.
            fips (np.array): The county identifier.
    """