from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DatasetFeature(str, Enum):
    """Dataset features available for training"""

    # The change of the enum values should be done in a backward compatible way

    pclass = "pclass"
    sex = "sex"
    age = "age"
    fare = "fare"
    embarked = "embarked"
    title_ = "title"
    is_alone = "is_alone"
    age_class = "age_class"


DEFAULT_FEATURE_SET = set([feature for feature in DatasetFeature])


class AlgoRf(BaseModel):
    name: Literal["rf"] = "rf"
    n_estimators: int = Field(100, ge=1)


class AlgoDt(BaseModel):
    name: Literal["dt"] = "dt"


class AlgoKnn(BaseModel):
    name: Literal["knn"] = "knn"
    # TODO: add parameters and set defaults
    n_neighbours: int = Field(5, ge=1)


class AlgoSvm(BaseModel):
    name: Literal["svm"] = "svm"
    # TODO: add parameters and set defaults


class AlgoLr(BaseModel):
    name: Literal["lr"] = "lr"


class ModelParams[T](BaseModel):
    """All the model training parameters"""

    algo: AlgoRf | AlgoDt | AlgoKnn | AlgoSvm | AlgoLr = Field(discriminator="name")
    random_state: int | None = Field(None)
    features: set[T]


class ModelInfo(BaseModel):
    accuracy: float


class Model(BaseModel):
    """
    Model description
    """

    id: str
    removable: bool = True
    params: ModelParams[DatasetFeature]
    info: ModelInfo


class InferenceRequest(BaseModel):
    """
    Data model for inference request.
    Only fields used by the respective model have to be set, others can be left unset.
    """

    # Some fields are not used at all, that is intended

    # ['Pclass' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']
    # Original features
    pclass: Literal[1, 2, 3] | None = None
    sex: Literal["male", "female"] | None = None
    age: float | None = Field(default=None, ge=0, le=120)
    sibsp: float | None = Field(default=None, ge=0)
    parch: float | None = Field(default=None, ge=0)
    fare: float | None = Field(default=None, ge=0, le=500)
    embarked: Literal["cherbourg", "queenstown", "southhampton"] | None = None
    # Semi-original features
    cabin_known: bool | None = None
    title: Literal["master", "miss", "mr", "mrs", "rare"] | None = None
    # Derived features
    travelled_alone: bool | None = None
    age_times_class: float | None = None


class InferenceResponse(BaseModel):
    """
    Data model for inference response.
    """

    survived: bool
    probability: float
