import logging
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from schemas import DatasetFeature, InferenceRequest

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Data:
    train_input: pd.DataFrame
    train_output: pd.Series
    test_input: pd.DataFrame
    test_output: pd.Series


dataset_name = "titanic"

# List of required files from the Titanic dataset
required_files = ["train.csv", "test.csv", "gender_submission.csv"]


# Function to download the Titanic dataset components individually
def download_titanic_dataset(path: Path):
    try:
        # Initialize the Kaggle API
        api = KaggleApi()
        api.authenticate()

        logger.info(f"Downloading {dataset_name} dataset components...")

        # Download each file individually
        for file in required_files:
            file_path = path / file
            api.competition_download_file(dataset_name, file_name=file, path=path)
            logger.info(f"Downloaded {file} to {file_path}")

    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise


def preprocess_train(df: pd.DataFrame):
    # Fill missing values
    df = df.assign(
        Age=df["Age"].fillna(df["Age"].median()),
        Fare=df["Fare"].fillna(df["Fare"].median()),
        Embarked=df["Embarked"].fillna(df["Embarked"].mode()[0]),
    )
    return preprocess(df)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # TODO: naming of the columns follows DatasetFeature/ord
    # TODO: do not drop throw away any features => add them to DatasetFeature

    # Drop irrelevant columns
    df = df.drop(["Ticket", "Cabin"], axis=1)

    # Extract Title from Name and replace rare titles
    df[DatasetFeature.pclass] = df["Pclass"]
    df[DatasetFeature.fare] = df["Fare"]
    df[DatasetFeature.age] = df["Age"]

    df[DatasetFeature.title_] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df[DatasetFeature.title_] = df[DatasetFeature.title_].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    df[DatasetFeature.title_] = (
        df[DatasetFeature.title_]
        .replace("Mlle", "Miss")
        .replace("Ms", "Miss")
        .replace("Mme", "Mrs")
    )

    # Map Title to numerical values
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df[DatasetFeature.title_] = df[DatasetFeature.title_].map(title_mapping).fillna(0)

    # Encoding categorical columns
    df[DatasetFeature.sex] = df["Sex"].map(
        {"male": 0, "female": 1}
    )  # Assuming SEX_MAP has been defined as {'male': 0, 'female': 1}
    df[DatasetFeature.embarked] = df["Embarked"].map(
        {"C": 0, "Q": 1, "S": 2}
    )  # Assuming EMBARKED_MAP has been defined

    # Feature engineering
    df[DatasetFeature.is_alone] = ((df["SibSp"] + df["Parch"]) == 0).astype(int)
    df[DatasetFeature.age_class] = df[DatasetFeature.age] * df[DatasetFeature.pclass]

    ordered_columns = {feature: feature.value for feature in DatasetFeature}

    df = df.rename(columns=ordered_columns)
    print(df.columns)

    # Return the processed dataframe and the target column
    # TODO: return all the columns that are in DatasetFeature
    return df[ordered_columns.values()], df["Survived"]


def prepare_data(data: Data, features: set[DatasetFeature]) -> Data:
    """Return a shallow copy of the passed data, containing only the requested features.
    The columns should be ordered according to DatasetFeature.ord
    """
    # Create a shallow copy of the data
    shallow_copy = copy(data)

    # Subset the train and test input dataframes to only include the requested features
    cols = sorted([f.value for f in features])
    shallow_copy.train_input = shallow_copy.train_input[cols]
    shallow_copy.test_input = shallow_copy.test_input[cols]

    return shallow_copy


def prepare_passenger_data(req: InferenceRequest, features: set[DatasetFeature]):
    """Transform the request into input vector containing given features.
    The transformation follows the one in `preprocess`.
    The features in the output vector are ordered according to DatasetFeature.ord
    """
    feature_sorted = sorted(features)

    vec: list[float] = []
    for f in feature_sorted:
        match f:
            case DatasetFeature.pclass:
                value = req.pclass
            case DatasetFeature.sex:
                value = {"male": 0, "female": 1}[req.sex]
            case DatasetFeature.age:
                value = req.age
            case DatasetFeature.fare:
                value = req.fare
            case DatasetFeature.is_alone:
                value = 1 if req.travelled_alone else 0
            case DatasetFeature.embarked:
                value = {"cherbourg": 0, "queenstown": 1, "southhampton": 2}[
                    req.embarked
                ]
            case DatasetFeature.title_:
                value = {"mr": 1, "miss": 2, "mrs": 3, "master": 4, "rare": 5}[
                    req.title
                ]
            case DatasetFeature.age_class:
                value = req.age * req.pclass
        vec.append(value)

    return pd.DataFrame(
        [vec],
        columns=list(map(lambda f: f.value, feature_sorted)),
    )


def save_preprocessed_data(path: Path, data: Data):
    # Save the preprocessed data to CSV files
    train = data.train_input.assign(survived=data.train_output)
    train.to_csv(path / "train_preprocessed.csv", index=False)
    test = data.test_input.assign(survived=data.test_output)
    test.to_csv(path / "test_preprocessed.csv", index=False)


def load_preprocessed_data(path: Path):
    if (
        not (path / "train_preprocessed.csv").exists()
        or not (path / "test_preprocessed.csv").exists()
    ):
        return None
    # Load the preprocessed data from CSV files

    train = pd.read_csv(path / "train_preprocessed.csv")
    test = pd.read_csv(path / "test_preprocessed.csv")
    return Data(
        train.drop(columns="survived"),
        train["survived"],
        test.drop(columns="survived"),
        test["survived"],
    )


def load_data(path: Path):
    path.mkdir(parents=True, exist_ok=True)

    data = load_preprocessed_data(path)
    # Load the dataset
    if data is None:
        download_titanic_dataset(path)

        # Preprocess the dataset
        train_output, train_input = preprocess_train(pd.read_csv(path / "train.csv"))
        _test = pd.read_csv(path / "test.csv")
        _submission = pd.read_csv(path / "gender_submission.csv")
        # Merge test and submission data to get the same columns as train
        test = pd.merge(_test, _submission, on="PassengerId", how="inner")

        test_output, test_input = preprocess_train(test)

        #
        data = Data(train_output, train_input, test_output, test_input)
        save_preprocessed_data(path, data)

    return data
