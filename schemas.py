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


class ModelParams(BaseModel):
    """All the model training parameters"""

    algo: AlgoRf | AlgoDt | AlgoKnn | AlgoSvm | AlgoLr = Field(discriminator="name")
    random_state: int | None = Field(None)
    features: set[DatasetFeature] = Field(DEFAULT_FEATURE_SET)


class ModelInfo(BaseModel):
    accuracy: float


class Model(BaseModel):
    """
    Model description
    """

    id: str
    params: ModelParams
    info: ModelInfo


class InferenceRequest(BaseModel):
    """
    Data model for inference request.
    """

    pclass: Literal[1, 2, 3]
    sex: Literal["male", "female"]
    age: float = Field(..., ge=0, le=100)
    fare: float = Field(..., ge=0, le=500)
    travelled_alone: bool
    embarked: Literal["cherbourg", "queenstown", "southhampton"]
    title: Literal["master", "miss", "mr", "mrs", "rare"]


class InferenceResponse(BaseModel):
    """
    Data model for inference response.
    """

    survived: bool
