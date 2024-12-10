import os
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

def create_rawDataFrames(srcPath):
    """
    Creates dataFrames from the csv files in the srcPath and then saves them in the data/raw_dataFrames directory using pickle.
        
    Parameters
    ----------
    srcPath : str
        The path to the csv files.

    Returns
    -------
    dataDic : dict
        A dictionary with the dataFrames.    
    """
    if not os.path.exists("data/raw_dataFrames"):
        os.makedirs("data/raw_dataFrames")

    dataDic = {"train": pd.read_csv(f"{srcPath}train_timeseries/train_timeseries.csv"),
               "test": pd.read_csv(f"{srcPath}test_timeseries/test_timeseries.csv"),
               "validation": pd.read_csv(f"{srcPath}validation_timeseries/validation_timeseries.csv"),
               "soil" : pd.read_csv(f"{srcPath}soil_data.csv"),
               }
    
    dfs = {
    k: dataDic[k].set_index(['fips', 'date'])
    for k in dataDic.keys() if k != "soil"
    }
    dfs["soil"] = dataDic["soil"]
    
    for k in dfs.keys():
        with open(f"data/raw_dataFrames/{k}.pickle", "wb") as f:
            pickle.dump(dfs[k], f)
    return dfs


def load_rawDataFrames():
    """
    This function loads the dataFrames dict from the data/raw_dataFrames directory.
        
    Returns
    -------
    dfs : dict
        A dictionary with the dataFrames.    
    """
    dfs = {}
    for file in os.listdir("data/raw_dataFrames"):
        with open(f"data/raw_dataFrames/{file}", "rb") as f:
            dfs[file.split(".")[0]] = pickle.load(f)
    return dfs

def interpolate_nans(padata, pkind="linear"):
    """
    See: https://stackoverflow.com/a/53050216/2167159

    Parameters
    ----------
    padata : np.array
        The array to interpolate.
    pkind : str
        The interpolation method.
        
    Returns
    -------
    f(aindexes) : np.array
        The interpolated array.
    """
    aindexes = np.arange(padata.shape[0])
    agood_indexes, = np.where(np.isfinite(padata))
    f = interp1d(agood_indexes,
                 padata[agood_indexes],
                 bounds_error=False,
                 copy=False,
                 fill_value="extrapolate",
                 kind=pkind,
               )
    return f(aindexes)


def date_encode(date):
    """
    Encode the cycling feature : the day of the year as a sin and cos function.
    Taken from https://www.pure.ed.ac.uk/ws/portalfiles/portal/217133242/DroughtED_MINIXHOFER_DOA18062021_AFV.pdf

    Parameters
    ----------
    date : str
        Date to encode.
        
    Returns
    -------
    np.sin(2 * np.pi * date.timetuple().tm_yday / 366) : float
        Sin of the date.
    np.cos(2 * np.pi * date.timetuple().tm_yday / 366) : float
        Cos of the date.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    return (
        np.sin(2 * np.pi * date.timetuple().tm_yday / 366),
        np.cos(2 * np.pi * date.timetuple().tm_yday / 366),
    )


def setup_encoders_targets():
    """
    A function to setup the class2id and id2class dictionaries.
    """
    class2id= { 'None': 0, 'D0': 1, 'D1': 2, 'D2': 3, 'D3': 4, 'D4': 5}
    id2class = {v: k for k, v in class2id.items()}

    return class2id, id2class


def load_XY(
    dfs,
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
    Load the data and create the X and y arrays.
    Taken from https://www.pure.ed.ac.uk/ws/portalfiles/portal/217133242/DroughtED_MINIXHOFER_DOA18062021_AFV.pdf

    Parameters
    ----------
    dfs : dict
        Dictionary containing the dataFrames.
    df : pd.DataFrame
        DataFrame to load the data from.
    random_state : int
        Random state to use.
    window_size : int
        Number of days in the past used for prediction.
    target_size : int
        Number of weeks into the future (the size of the output vector).
    fuse_past : bool
        Add the past drought observations.
    return_fips : bool
        Return the county identifier.
    encode_season : bool
        Encode the season.
    use_prev_year : bool
        Add observations from 1 year prior.
        
    Returns
    -------
    X : np.array
        The input array.
    y : np.array
        The output array.
    fips : np.array
        The county identifier.
    """

    dico_trad = {}
    df = dfs[df]
    soil_df = dfs["soil"]
    for cat in ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5", "SQ6", "SQ7"]:
        dico_trad[cat] = {j: i for i,j in enumerate(sorted(soil_df[cat].unique()))}
        soil_df[cat] = soil_df[cat].map(dico_trad[cat])
    time_data_cols = sorted(
        [c for c in df.columns if c not in ["fips", "date", "score"]]
    )
    static_data_cols = sorted(
        [c for c in soil_df.columns if c not in ["fips", "lat", "lon",
                                                 "SQ1", "SQ2", "SQ3",
                                                 "SQ4", "SQ5", "SQ6",
                                                 "SQ7"]]
    )
    static_cat_cols = sorted(
        [c for c in soil_df.columns if c in ["SQ1", "SQ2", "SQ3",
                                             "SQ4", "SQ5", "SQ6",
                                             "SQ7"]]
    )

    count = 0
    score_df = df.dropna(subset=["score"])
    X_static = np.empty((len(df) // window_size, len(static_data_cols)))
    X_static_cat = np.empty((len(df) // window_size, len(static_cat_cols)))
    X_fips_date = []
    add_dim = 0
    if use_prev_year:
        add_dim += len(time_data_cols)
    if fuse_past:
        add_dim += 1
        if use_prev_year:
            add_dim += 1
    if encode_season:
        add_dim += 2
    X_time = np.empty(
        (len(df) // window_size, window_size, len(time_data_cols) + add_dim)
    )
    y_past = np.empty((len(df) // window_size, window_size))
    y_target = np.empty((len(df) // window_size, target_size))
    if random_state is not None:
        np.random.seed(random_state)
    for fips in tqdm(score_df.index.get_level_values(0).unique()):
        if random_state is not None:
            start_i = np.random.randint(1, window_size)
        else:
            start_i = 1
        fips_df = df[(df.index.get_level_values(0) == fips)]
        X = fips_df[time_data_cols].values
        y = fips_df["score"].values
        X_s = soil_df[soil_df["fips"] == fips][static_data_cols].values[0]
        X_s_cat = soil_df[soil_df["fips"] == fips][static_cat_cols].values[0]
        for i in range(start_i, len(y) - (window_size + target_size * 7), window_size):
            X_fips_date.append((fips, fips_df.index[i : i + window_size][-1]))
            X_time[count, :, : len(time_data_cols)] = X[i : i + window_size]
            if use_prev_year:
                if i < 365 or len(X[i - 365 : i + window_size - 365]) < window_size:
                    continue
                X_time[count, :, -len(time_data_cols) :] = X[
                    i - 365 : i + window_size - 365
                ]
            if not fuse_past:
                y_past[count] = interpolate_nans(y[i : i + window_size])
            else:
                X_time[count, :, len(time_data_cols)] = interpolate_nans(
                    y[i : i + window_size]
                )
            if encode_season:
                enc_dates = [
                    date_encode(d) for f, d in fips_df.index[i : i + window_size].values
                ]
                d_sin, d_cos = [s for s, c in enc_dates], [c for s, c in enc_dates]
                X_time[count, :, len(time_data_cols) + (add_dim - 2)] = d_sin
                X_time[count, :, len(time_data_cols) + (add_dim - 2) + 1] = d_cos
            temp_y = y[i + window_size : i + window_size + target_size * 7]
            y_target[count] = np.array(temp_y[~np.isnan(temp_y)][:target_size])
            X_static[count] = X_s
            X_static_cat[count] = X_s_cat
            count += 1
    print(f"loaded {count} samples")
    results = [X_static[:count], X_static_cat[:count], X_time[:count], y_target[:count]]
    if not fuse_past:
        results.append(y_past[:count])
    if return_fips:
        results.append(X_fips_date)
    return results

def normalize(X_static, X_time, y_past=None, dicts=({},{},{}), fit=False):
    """
    Normalize the data using a RobustScaler.
    """
    if fit and dicts != ({},{},{}):
        raise ValueError("You're trying to fit the scalers and provide them at the same time")

    scaler_dict, scaler_dict_static, scaler_dict_past = dicts

    for index in tqdm(range(X_time.shape[-1])):
        if fit:
            scaler_dict[index] = RobustScaler().fit(X_time[:, :, index].reshape(-1, 1))
        X_time[:, :, index] = (
            scaler_dict[index]
            .transform(X_time[:, :, index].reshape(-1, 1))
            .reshape(-1, X_time.shape[-2])
        )
    for index in tqdm(range(X_static.shape[-1])):
        if fit:
            scaler_dict_static[index] = RobustScaler().fit(
                X_static[:, index].reshape(-1, 1)
            )
        X_static[:, index] = (
            scaler_dict_static[index]
            .transform(X_static[:, index].reshape(-1, 1))
            .reshape(1, -1)
        )
    index = 0
    if y_past is not None:
        if fit:
            scaler_dict_past[index] = RobustScaler().fit(y_past.reshape(-1, 1))
        y_past[:, :] = (
            scaler_dict_past[index]
            .transform(y_past.reshape(-1, 1))
            .reshape(-1, y_past.shape[-1])
        )
        final_dict = (scaler_dict, scaler_dict_static, scaler_dict_past)
        return X_static, X_time, y_past, final_dict
    final_dict = (scaler_dict, scaler_dict_static, scaler_dict_past)
    return X_static, X_time, final_dict