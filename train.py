from pathlib import Path

from schemas import ModelInfo, ModelParams


def train_and_write(path: Path, params: ModelParams):
    # TODO: implement
    # TODO: write path/model.pkl

    # TODO: actual ModelInfo
    info_json = ModelInfo(accuracy=0.123).model_dump_json()
    _ = (path / "info.json").write_text(info_json)
    params_json = params.model_dump_json()
    _ = (path / "params.json").write_text(params_json)


# Supposed to be started as a subprocess (for asynchronous training)
if __name__ == "__main__":
    from sys import argv

    path = Path(argv[1])  # Path to save the model and the metadata
    params_json = argv[2]  # A single JSON following ModelParams

    path.mkdir(parents=True, exist_ok=True)
    params = ModelParams.model_validate_json(params_json)
    train_and_write(path, params)
