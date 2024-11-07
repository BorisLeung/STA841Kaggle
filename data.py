import os

import numpy as np
import pandas as pd

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
