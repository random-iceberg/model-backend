from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class DatasetFeature(str, Enum):
    """Dataset features available for training"""

    pclass = "pclass"
    sex = "sex"
    age = "age"
    fare = "fare"
    embarked = "embarked"
    title_ = "title"
    is_alone = "is_alone"
    ageclass = "ageclass"


class AlgoRf(BaseModel):
    name: Literal["rf"] = "rf"
    # TODO: add parameters and set defaults
    some_rf_param: str


class AlgoDt(BaseModel):
    name: Literal["dt"] = "dt"
    # TODO: add parameters and set defaults
    some_dt_param: str


class AlgoKnn(BaseModel):
    name: Literal["knn"] = "knn"
    # TODO: add parameters and set defaults
    some_knn_param: str


class AlgoSvn(BaseModel):
    name: Literal["svn"] = "svn"
    # TODO: add parameters and set defaults
    some_svn_param: str


class AlgoLr(BaseModel):
    name: Literal["lr"] = "lr"
    # TODO: add parameters and set defaults
    some_lr_param: str


class ModelParams(BaseModel):
    """All the model training parameters"""

    algo: AlgoRf | AlgoDt | AlgoKnn | AlgoSvn | AlgoLr = Field(discriminator="name")
    # TODO: add parameters and set defaults
    some_common_param: str
    features: set[DatasetFeature]


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

    pass


class InferenceResponse(BaseModel):
    """
    Data model for inference response.
    """

    survived: bool
