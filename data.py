import os

from matplotlib.pyplot import step
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

__all__ = [
    "edu_train",
    "edu_test",
    "house_train",
    "house_test",
    "pov_train",
    "sample_submission",
    "Y_COLUMNS",
    "HOUSE_PREFIX",
    "EDU_PREFIX",
    "VALID_NULL",
    "combined_house_edu",
    "combined_train",
    "combined_train_with_num_pov",
    "combined_transformed_train",
    "combined_transformed_train_with_num_pov",
    "combined_test",
    "combined_transformed_test",
    "DATA_DIR",
    "PROCESSED_DIR",
    "PREDICTIONS_DIR",
    "generate_submission",
    "column_types_df",
    "transform_status",
    "transform_mother_age",
    "transform_father_age",
    "copy_mother_info",
    "copy_father_info",
    "transform_all_house",
    "transform_education_levels",
    "get_preprocessor",
    "remove_boring_columns",
    "remove_all_valid_null_columns",
    "get_divided_edu",
    "get_divided_house",
]

Y_COLUMNS = [f"subjective_poverty_{i}" for i in range(1, 11)]
HOUSE_PREFIX = "house_"
EDU_PREFIX = "edu_"
VALID_NULL = -999

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
combined_house_edu = pd.read_csv(os.path.join(PROCESSED_DIR, "combined_house_edu.csv"))
combined_train = pd.read_csv(os.path.join(PROCESSED_DIR, "combined_train.csv"))
combined_train_with_num_pov = combined_train.copy()
combined_train_with_num_pov["num_pov"] = (
    np.argmax(
        combined_train_with_num_pov.filter(like="subjective_poverty").values, axis=1
    )
    + 1
)
combined_transformed_train = pd.read_csv(
    os.path.join(PROCESSED_DIR, "combined_transformed_train.csv")
)
combined_transformed_train_with_num_pov = combined_transformed_train.copy()
combined_transformed_train_with_num_pov["num_pov"] = (
    np.argmax(
        combined_transformed_train_with_num_pov.filter(
            like="subjective_poverty"
        ).values,
        axis=1,
    )
    + 1
)

combined_test = pd.read_csv(os.path.join(PROCESSED_DIR, "combined_test.csv"))
combined_transformed_test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "combined_transformed_test.csv")
)


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


### ============================= ###
### Data Transformation Functions ###
### ============================= ###


def transform_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    young_mask = df["house_q05y"] < 12
    df.loc[young_mask, "house_q06"] = -1
    df.loc[young_mask, "house_q07"] = -1

    single_mask = (df["house_q06"] == 4) | (df["house_q06"] == 5)
    df.loc[single_mask, "house_q07"] = -1
    if "house_q08" in df.columns:
        df = df.drop(columns=["house_q08"])
    return df


def transform_mother_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # slightly worse
    df["house_q15"] = pd.cut(
        df["house_q15"], bins=[-1000, 0, 40, 50, 60, 70, 80, 90], labels=list(range(7))
    )
    df["house_q16"] = pd.cut(
        df["house_q16"], bins=[-1000, 0, 40, 50, 60, 70, 80, 90], labels=list(range(7))
    )

    # much worse
    # df["house_q15"] = df["house_q15"].replace(VALID_NULL, 0)
    # df["house_q15"] = df["house_q15"].replace(np.nan, 0)
    # df["house_q16"] = df["house_q16"].replace(VALID_NULL, 0)
    # df["house_q16"] = df["house_q16"].replace(np.nan, 0)
    # df["house_q15"] = df["house_q15"].astype(int) - df["house_q16"].astype(int)
    # df = df.drop(columns=["house_q16"])
    # df["house_q15"] = df["house_q15"].replace(0, np.nan)
    df = df.drop(columns=["house_q14"])
    return df


def transform_father_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # slightly worse
    df["house_q21"] = pd.cut(
        df["house_q21"], bins=[-1000, 0, 40, 50, 60, 70, 80, 90], labels=list(range(7))
    )
    df["house_q22"] = pd.cut(
        df["house_q22"], bins=[-1000, 0, 40, 50, 60, 70, 80, 90], labels=list(range(7))
    )

    # much worse
    # df["house_q21"] = df["house_q21"].replace(VALID_NULL, 0)
    # df["house_q21"] = df["house_q21"].replace(np.nan, 0)
    # df["house_q22"] = df["house_q22"].replace(VALID_NULL, 0)
    # df["house_q22"] = df["house_q22"].replace(np.nan, 0)
    # df["house_q21"] = df["house_q21"].astype(int) - df["house_q22"].astype(int)
    # df = df.drop(columns=["house_q22"])
    # df["house_q21"] = df["house_q21"].replace(0, np.nan)
    df = df.drop(columns=["house_q20"])
    return df


def map_education_level(edu_level: int) -> int:
    """
    ## HOUSE

    NONE, OR SOME PRIMARY 1

    COMPLETED PRIMARY 4/5 YEARS 2
    COMPLETED PRIMARY 7/8/9 YEARS 3

    SOME SECONDARY GENERAL 4
    COMPLETED SECONDARY 5

    SOME VOCATIONAL SCHOOL 6
    COMPLETED VICATIONAL SCHOOL 7

    SOME UNIVERSITY 8
    COMPLETED UNIVERSITY DEGREE 9

    POSTUNIVERSITY 10

    ## EDU

    NONE 0

    "8 OR 9 YEARS" SCHOOL 1

    GYMNAZIUM(SECONDARY GENERAL) 2

    TECHICUM <2 YEARS 3
    VOCATIONAL 2-3 YEARS 4
    VOCATIONAL 4/5 YEARS 5

    UNIVERSITY- ALBANIA 6
    UNIVERSITY- ABROAD 7

    MASTER- ALBANIA 8
    MASTER- ABROAD 9
    DOCTORATE/PhD-ALBANIA 10
    DOCTORATE/PhD-ABROAD 11
    """
    if edu_level >= 8:
        return 10
    elif edu_level >= 6:
        return 8
    elif edu_level >= 3:
        return 6
    elif edu_level >= 2:
        return 4
    elif edu_level >= 1:
        return 3
    else:
        return 1


def copy_mother_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mothers_in_house_mask = df["house_q11"] == 1
    mothers_id = (
        df[mothers_in_house_mask]["psu_hh_idcode"].str.split("_").str[:-1].agg("_".join)
        + "_"
        + df[mothers_in_house_mask]["house_q12"].astype(int).astype(str)
    )
    df.loc[mothers_in_house_mask, "mother_id"] = mothers_id

    edu_train_copy = combined_house_edu.copy()
    mothers_rows = edu_train_copy[edu_train_copy["psu_hh_idcode"].isin(mothers_id)]

    for index, row in df[mothers_in_house_mask].iterrows():
        mother_id = row["mother_id"]
        mother_row = mothers_rows[mothers_rows["psu_hh_idcode"] == mother_id]
        df.loc[index, "house_q13"] = (
            map_education_level(mother_row["edu_q04"].values[0])
            if not mother_row.empty
            else 1
        )
        df.loc[index, "house_q14"] = 1
        if mother_row.empty:
            continue
        df.loc[index, "house_q16"] = mother_row["house_q05y"].values[0]

    df.reset_index()
    df.drop(columns=["mother_id", "house_q12"], inplace=True)
    return df


def copy_father_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fathers_in_house_mask = df["house_q17"] == 1
    fathers_id = (
        df[fathers_in_house_mask]["psu_hh_idcode"].str.split("_").str[:-1].agg("_".join)
        + "_"
        + df[fathers_in_house_mask]["house_q18"].astype(int).astype(str)
    )
    df.loc[fathers_in_house_mask, "father_id"] = fathers_id

    edu_train_copy = combined_house_edu.copy()
    fathers_rows = edu_train_copy[edu_train_copy["psu_hh_idcode"].isin(fathers_id)]

    for index, row in df[fathers_in_house_mask].iterrows():
        father_id = row["father_id"]
        father_row = fathers_rows[fathers_rows["psu_hh_idcode"] == father_id]
        df.loc[index, "house_q19"] = (
            map_education_level(father_row["edu_q04"].values[0])
            if not father_row.empty
            else 1
        )
        df.loc[index, "house_q20"] = 1
        if father_row.empty:
            continue
        df.loc[index, "house_q22"] = father_row["house_q05y"].values[0]

    df.reset_index()
    df.drop(columns=["father_id", "house_q18"], inplace=True)
    return df


def bin_education_levels(df: pd.DataFrame) -> pd.DataFrame:
    def bin_edu(x):
        if x < 2:
            return 0
        elif x <= 3:
            return 1
        elif x <= 5:
            return 2
        elif x <= 7:
            return 3
        elif x <= 9:
            return 4
        return 5

    df = df.copy()
    if "house_q13" in df.columns:
        df["house_q13"] = df["house_q13"].apply(bin_edu)
    if "house_q17" in df.columns:
        df["house_q17"] = df["house_q17"].apply(bin_edu)
    return df


def transform_all_house(
    df: pd.DataFrame, calls: list[callable] | None = None
) -> pd.DataFrame:
    if not calls:
        calls = [
            transform_status,
            copy_father_info,
            copy_mother_info,
            bin_education_levels,  # ???
            transform_father_age,
            transform_mother_age,
        ]
    for call in calls:
        df = call(df)
    return df


def transform_education_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def transform_q5(grade: int, level: int) -> int:
        match grade:
            case 1:
                return level / 9
            case 2:
                return level / 4
            case 3:
                return level / 2
            case 4:
                return level / 3
            case 5:
                return level / 5
            case 6 | 7:
                return level / 6
            case range(8, 12):
                return level / 5
            case _:
                return level

    if "edu_q04" in df.columns and "edu_q05" in df.columns:
        df["edu_q05"] = df[["edu_q04", "edu_q05"]].apply(
            lambda row: transform_q5(row["edu_q04"], row["edu_q05"]), axis=1
        )

    def transform_q13(grade: int, level: int) -> int:
        match grade:
            case 1:
                return level / 9
            case 2:
                return level / 2
            case 3:
                return level / 4
            case 4:
                return level / 3
            case 5:
                return level / 5
            case 6:
                return level / 3
            case 7:
                return level / 2
            case 8:
                return level / 6
            case 9:
                return level / 5
            case 10:
                return level / 5
            case _:
                return level

    if "edu_q12" in df.columns and "edu_q13" in df.columns:
        df["edu_q13"] = df[["edu_q12", "edu_q13"]].apply(
            lambda row: transform_q13(row["edu_q12"], row["edu_q13"]), axis=1
        )

    transform_q22 = transform_q13
    if "edu_q21" in df.columns and "edu_q22" in df.columns:
        df["edu_q22"] = df[["edu_q21", "edu_q22"]].apply(
            lambda row: transform_q22(row["edu_q21"], row["edu_q22"]), axis=1
        )

    return df


### ======================= ###
### Preprocessing Functions ###
### ======================= ###

COLUMN_TYPES_MAP_FILE = "column_classes.xlsx"
column_types_df = pd.read_excel(COLUMN_TYPES_MAP_FILE, sheet_name=1)


def get_preprocessor(
    binary_transformer: Pipeline | None = None,
    categorical_transformer: Pipeline | None = None,
    numerical_transformer: Pipeline | None = None,
    ordinal_transformer: Pipeline | None = None,
    imputer_strategy: str | list[str | None] | None = "mean",
    remainder: str = "passthrough",
) -> ColumnTransformer:
    """
    Creates a ColumnTransformer that preprocesses different types of columns using specified transformers. The column types are inferred from the column_types_df DataFrame, which is read from the column_classes.xlsx file.

    Args:
        binary_transformer (Pipeline | None): Transformer for binary columns. Defaults to OneHotEncoder with drop="first".
        categorical_transformer (Pipeline | None): Transformer for categorical columns. Defaults to OneHotEncoder with drop="first".
        numerical_transformer (Pipeline | None): Transformer for numerical columns. Defaults to StandardScaler.
        ordinal_transformer (Pipeline | None): Transformer for ordinal columns. Defaults to MinMaxScaler.
        imputer_strategy (str | list[str | None] | None): Strategy for imputing missing values. Defaults to "mean".
        remainder (str): Strategy for handling remaining columns. Defaults to "passthrough".

    Returns:
        ColumnTransformer: A ColumnTransformer that applies the specified transformers to the corresponding column types.
    """
    if not isinstance(imputer_strategy, list):
        imputer_strategy = [imputer_strategy] * 4

    if binary_transformer is None:
        steps = [
            (
                "encoder",
                OneHotEncoder(
                    sparse_output=False,
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                ),  # sparse_output=False for pandas output
            )
        ]
        if imputer_strategy[0] is not None:
            steps.insert(0, ("imputer", SimpleImputer(strategy=imputer_strategy[0])))
        binary_transformer = Pipeline(steps=steps)
    if categorical_transformer is None:
        steps = [
            (
                "encoder",
                OneHotEncoder(
                    sparse_output=False,
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.05,
                ),
            )
        ]  # sparse_output=False for pandas output

        if imputer_strategy[1] is not None:
            steps.insert(0, ("imputer", SimpleImputer(strategy=imputer_strategy[1])))
        categorical_transformer = Pipeline(steps=steps)
    if numerical_transformer is None:
        steps = [("scaler", MinMaxScaler())]
        if imputer_strategy[2] is not None:
            steps.insert(0, ("imputer", SimpleImputer(strategy=imputer_strategy[2])))
        numerical_transformer = Pipeline(steps=steps)
    if ordinal_transformer is None:
        steps = [("scaler", MinMaxScaler())]
        if imputer_strategy[3] is not None:
            steps.insert(0, ("imputer", SimpleImputer(strategy=imputer_strategy[3])))
        ordinal_transformer = Pipeline(steps=steps)

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
            (
                "binary",
                binary_transformer,
                selector(pattern="|".join(all_binary_columns)),
            ),
            (
                "categorical",
                categorical_transformer,
                selector(pattern="|".join(categorical_columns)),
            ),
            (
                "numerical",
                numerical_transformer,
                selector(pattern="|".join(numerical_columns)),
            ),
            (
                "ordinal",
                ordinal_transformer,
                selector(pattern="|".join(ordinal_columns)),
            ),
        ],
        remainder=remainder,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def remove_boring_columns(df: pd.DataFrame):
    # boring_columns = ["edu_q16", "house_q08", "house_q10", "house_q12", "house_q18"]
    boring_columns = ["edu_q16", "house_q04", "house_q05m", "house_q08", "house_q10"]
    for column in boring_columns:
        if column in df.columns:
            df.drop(columns=column, inplace=True)
    return df


def remove_all_valid_null_columns(df: pd.DataFrame):
    """
    Remove columns from a DataFrame where all values are a specific valid null value.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with columns removed where all values are the valid null value.
    """

    columns = df.columns[(df == VALID_NULL).sum(axis=0) == df.shape[0]]
    return df.drop(columns=columns)


### ======================== ###
### Data Splitting Functions ###
### ======================== ###


def get_divided_edu(data: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Divides the input DataFrame into three categories based on education-related columns and returns a list of DataFrames.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing education-related columns 'edu_q03' and 'edu_q14'.

    Returns:
    list[pd.DataFrame]: A list of three DataFrames:
        - The first DataFrame contains rows where 'edu_q03' equals 2 (never attended school).
        - The second DataFrame contains rows where 'edu_q03' does not equal 2 and 'edu_q14' equals 2 (attended school but not enrolled in the past year).
        - The third DataFrame contains rows where 'edu_q03' does not equal 2 and 'edu_q14' equals 1 (attended school and attended in the past year).
    """
    never_attended_school_mask = data["edu_q03"] == 2  # count: 139
    attended_school_but_not_enrolled_in_past_year = (~never_attended_school_mask) & (
        data["edu_q14"] == 2
    )  # count: 5138, including 5134 over 19 years of age and 4 under
    attended_school_and_attended_in_past_year = (~never_attended_school_mask) & (
        data["edu_q14"] == 1
    )  # count: 57

    return [
        remove_all_valid_null_columns(data[never_attended_school_mask]).drop(
            columns=["edu_q03"]
        ),
        remove_all_valid_null_columns(
            data[attended_school_but_not_enrolled_in_past_year]
        ).drop(columns=["edu_q03", "edu_q14"]),
        remove_all_valid_null_columns(
            data[attended_school_and_attended_in_past_year]
        ).drop(columns=["edu_q03", "edu_q14"]),
    ]


def get_divided_house(data: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Splits the input DataFrame into two DataFrames based on whether the father is living in the house or not,
    and removes all columns with only null values from each resulting DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing household data.

    Returns:
        list[pd.DataFrame]: A list containing two DataFrames:
            - The first DataFrame contains rows where the father is not living in the house.
            - The second DataFrame contains rows where the father is living in the house.
    """
    father_not_living_here = data["house_q17"] == 2  # count: 5119
    father_living_here = ~father_not_living_here  # count: 215

    return [
        remove_all_valid_null_columns(filtered_data)
        for filtered_data in [
            data[father_not_living_here],
            data[father_living_here],
        ]
    ]
