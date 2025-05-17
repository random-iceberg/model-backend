import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score

from schemas import ModelInfo, ModelParams
from utils.data import Data, load_data, prepare_data
from utils.model_factory import make_estimator


def train_and_write(data_path: Path, model_path: Path, params: ModelParams):
    # 1. Prepare data
    data = load_data(data_path)
    data = prepare_data(data, params.features)

    # 2. Instantiate via factory
    clf = make_estimator(params)

    # 3. Fit & evaluate
    _ = clf.fit(data.train_input, data.train_output)
    preds = clf.predict(data.test_input)
    acc = accuracy_score(data.test_output, preds)

    # 4. Build description and persist
    info = ModelInfo(accuracy=acc)

    # Save the model
    model_path.mkdir(parents=True, exist_ok=True)
    with (model_path / "model.pkl").open("wb") as f:
        pickle.dump(clf, f)
    info_json = info.model_dump_json()
    _ = (model_path / "info.json").write_text(info_json)
    params_json = params.model_dump_json()
    _ = (model_path / "params.json").write_text(params_json)


# Supposed to be started as a subprocess (for asynchronous training)
if __name__ == "__main__":
    from sys import argv

    data_path = Path(argv[1])
    path = Path(argv[2])  # Path to save the model and the metadata
    params_json = argv[3]  # A single JSON following ModelParams

    path.mkdir(parents=True, exist_ok=True)
    params = ModelParams.model_validate_json(params_json)
    train_and_write(data_path, path, params)
