import os
import argparse
import pickle
from utilities import (create_rawDataFrames,
                       load_rawDataFrames,
                       load_XY,
                       normalize
                       )

def main():
    parser = argparse.ArgumentParser(description="Create the dataframes offering a two steps process.")
    parser.add_argument("-s", "--srcPath",
                        type=str,
                        default="src/",
                        help='The path to the src csv files.'
                        )
    parser.add_argument("-raw", "--createraw",
                        action="store_true",
                        help='if this flag is set, the script only performs the raw dataframes creation.'
                        )
    parser.add_argument("-pp", "--preprocess",
                        action="store_true",
                        help='if this flag is set, the script only performs the data preprocessing.'
                        )
    args = parser.parse_args()

    if not args.preprocess:
        dfs = create_rawDataFrames(args.srcPath)
        if not args.createraw:
            print("(by default) Creating the raw dataFrames and preprocessing the dataFrames.")
        if args.createraw:
            print("Creating the raw dataFrames.")
            return

    if args.preprocess:
        dfs = load_rawDataFrames()
        print("Preprocessing the dataFrames.")

    X_tabular_train, X_tabular_cat_train, X_time_train, y_target_train, list_cat = load_XY(dfs, "train")
    print("train shape", X_time_train.shape)
    X_tabular_valid, X_tabular_cat_valid, X_time_valid, y_target_valid, valid_fips = load_XY(dfs, "validation", return_fips=True)
    print("validation shape", X_time_valid.shape)
    X_tabular_test, X_tabular_cat_test, X_time_test, y_target_test, test_fips= load_XY(dfs, "test", return_fips=True)
    print("test shape", X_time_test.shape)
    X_tabular_train, X_time_train, scaler_dicts = normalize(X_tabular_train, X_time_train, fit=True)
    X_tabular_valid, X_time_valid, _ = normalize(X_tabular_valid, X_time_valid, dicts=scaler_dicts)
    X_tabular_test, X_time_test, _ = normalize(X_tabular_test, X_time_test, dicts=scaler_dicts)

    if not os.path.exists("data/processed_dataFrames"):
        os.makedirs("data/processed_dataFrames")

    for k in ["X_tabular_train", "X_tabular_cat_train",
              "X_time_train", "y_target_train",
              "X_tabular_valid", "X_tabular_cat_valid",
              "X_time_valid", "y_target_valid",
              "X_tabular_test", "X_tabular_cat_test",
              "X_time_test", "y_target_test",
              "valid_fips", "test_fips",
              "list_cat",
              ]:
        with open(f"data/processed_dataFrames/{k}.pickle", "wb") as f:
            pickle.dump(locals()[k], f)

    print("DataFrames have been created and saved in the processed_dataFrames folder.")

if __name__ == "__main__":
    main()