from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Import your Pydantic schemas
from schemas import (
    AlgoDt,
    AlgoKnn,
    AlgoLr,
    AlgoRf,
    AlgoSvm,
    DatasetFeature,
    ModelParams,
)


def make_estimator(params: ModelParams[DatasetFeature]):
    """
    Factory to instantiate an sklearn estimator based on a Pydantic AlgoXxx model.
    """
    algo_params = params.algo
    match algo_params:
        case AlgoRf():
            return RandomForestClassifier(
                n_estimators=algo_params.n_estimators,
                random_state=params.random_state,
            )

        case AlgoDt():
            return DecisionTreeClassifier(
                random_state=params.random_state,
            )

        case AlgoKnn():
            return KNeighborsClassifier(
                n_neighbors=algo_params.n_neighbours,
            )

        case AlgoSvm():
            return SVC(
                random_state=params.random_state,
                probability=True,
            )

        case AlgoLr():
            return LogisticRegression(
                solver="liblinear",
                random_state=params.random_state,
            )
