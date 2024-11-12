import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

__all__ = [
    "edu_train",
    "edu_test",
    "house_train",
    "house_test",
    "pov_train",
    "sample_submission",
    "HOUSE_PREFIX",
    "EDU_PREFIX",
    "Y_COLUMNS",
    "combined_train",
    "combined_train_with_num_pov",
    "combined_test",
    "DATA_DIR",
    "PROCESSED_DIR",
    "PREDICTIONS_DIR",
    "generate_submission",
    "column_types_df",
    "get_preprocessor",
]

Y_COLUMNS = [f"subjective_poverty_{i}" for i in range(1, 11)]
HOUSE_PREFIX = "house_"
EDU_PREFIX = "edu_"

DATA_DIR = "data"
PROCESSED_DIR = "processed"
PREDICTIONS_DIR = "predictions"

edu_train_data = "module_Education_train_set.csv"
edu_test_data = "module_Education_test_set.csv"
house_train_data = "module_HouseholdInfo_train_set.csv"
house_test_data = "module_HouseholdInfo_test_set.csv"
pov_train_data = "module_SubjectivePoverty_train_set.csv"
sample_submission_data = "sample_submission.csv"

edu_train = pd.read_csv(os.path.join(DATA_DIR, edu_train_data))
edu_test = pd.read_csv(os.path.join(DATA_DIR, edu_test_data))
house_train = pd.read_csv(os.path.join(DATA_DIR, house_train_data))
house_test = pd.read_csv(os.path.join(DATA_DIR, house_test_data))
pov_train = pd.read_csv(os.path.join(DATA_DIR, pov_train_data))
sample_submission = pd.read_csv(os.path.join(DATA_DIR, sample_submission_data))
combined_train = pd.read_csv(os.path.join(PROCESSED_DIR, "combined_train.csv"))
combined_train_with_num_pov = combined_train.copy()
combined_train_with_num_pov["num_pov"] = (
    np.argmax(
        combined_train_with_num_pov.filter(like="subjective_poverty").values, axis=1
    )
    + 1
)
combined_test = pd.read_csv(os.path.join(PROCESSED_DIR, "combined_test.csv"))


def generate_submission(y_pred: np.ndarray, filename: str) -> None:
    """
    Generates a submission file for a Kaggle competition.

    This function takes the predicted values and a filename, checks if a file with the same name
    already exists in the specified directory, and if so, increments the filename to avoid
    overwriting. It then creates a DataFrame with the predicted values and saves it as a CSV file.

    Args:
        y_pred (numpy.ndarray): The predicted values to be included in the submission file.
        filename (str): The base name for the submission file.

    Returns:
        None

    Side Effects:
        Saves a CSV file in the PREDICTIONS_DIR directory.
        Prints the name of the saved submission file.
    """
    # check filename exists in dir and increment
    i = 1
    while os.path.exists(os.path.join(PREDICTIONS_DIR, f"{filename}-{i}.csv")):
        i += 1
    filename = f"{filename}-{i}"

    pd.DataFrame(
        np.hstack([sample_submission["psu_hh_idcode"].values.reshape(-1, 1), y_pred]),
        columns=["psu_hh_idcode"] + Y_COLUMNS,
    ).to_csv(os.path.join(PREDICTIONS_DIR, f"{filename}.csv"), index=False)
    print(f"Submission file saved as {filename}.csv")


COLUMN_TYPES_MAP_FILE = "column_classes.xlsx"
column_types_df = pd.read_excel(COLUMN_TYPES_MAP_FILE, sheet_name=1)


def get_preprocessor(
    binary_transformer: Pipeline | None = None,
    categorical_transformer: Pipeline | None = None,
    numerical_transformer: Pipeline | None = None,
    ordinal_transformer: Pipeline | None = None,
    remainder: str = "passthrough",
) -> ColumnTransformer:
    """
    Creates a ColumnTransformer that preprocesses different types of columns using specified transformers. The column types are inferred from the column_types_df DataFrame, which is read from the column_classes.xlsx file.

    Args:
        binary_transformer (Pipeline | None): Transformer for binary columns. Defaults to OneHotEncoder with drop="first".
        categorical_transformer (Pipeline | None): Transformer for categorical columns. Defaults to OneHotEncoder with drop="first".
        numerical_transformer (Pipeline | None): Transformer for numerical columns. Defaults to StandardScaler.
        ordinal_transformer (Pipeline | None): Transformer for ordinal columns. Defaults to MinMaxScaler.
        remainder (str): Strategy for handling remaining columns. Defaults to "passthrough".

    Returns:
        ColumnTransformer: A ColumnTransformer that applies the specified transformers to the corresponding column types.
    """

    if binary_transformer is None:
        binary_transformer = Pipeline(
            [
                ("encoder", OneHotEncoder(sparse_output=False, drop="first"))
            ]  # sparse_output=False for pandas output
        )
    if categorical_transformer is None:
        categorical_transformer = Pipeline(
            [
                ("encoder", OneHotEncoder(sparse_output=False, drop="first"))
            ]  # sparse_output=False for pandas output
        )
    if numerical_transformer is None:
        numerical_transformer = Pipeline([("scaler", StandardScaler())])
    if ordinal_transformer is None:
        ordinal_transformer = Pipeline([("scaler", MinMaxScaler())])

    binary_columns = column_types_df[column_types_df["type"] == "binary"][
        "column"
    ].values
    pseudo_binary_columns = column_types_df[column_types_df["type"] == "pseudo-binary"][
        "column"
    ].values
    categorical_columns = column_types_df[column_types_df["type"] == "categorical"][
        "column"
    ].values
    numerical_columns = column_types_df[column_types_df["type"] == "numerical"][
        "column"
    ].values
    ordinal_columns = column_types_df[column_types_df["type"] == "ordinal"][
        "column"
    ].values
    all_binary_columns = np.append(binary_columns, pseudo_binary_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", binary_transformer, all_binary_columns),
            ("categorical", categorical_transformer, categorical_columns),
            ("numerical", numerical_transformer, numerical_columns),
            ("ordinal", ordinal_transformer, ordinal_columns),
        ],
        remainder=remainder,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
