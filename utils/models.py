import logging
import pickle
import shutil
import subprocess
import sys
from asyncio import to_thread
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import train
from schemas import (
    DEFAULT_FEATURE_SET,
    AlgoKnn,
    AlgoLr,
    AlgoRf,
    AlgoSvm,
    Model,
    ModelInfo,
    ModelParams,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    desc: Model
    impl: Any  # sklearn something


class LoadedModels:
    def __init__(self, path: Path, dataset_path: Path):
        self.path: Path = path
        self.dataset_path: Path = dataset_path
        self.models: dict[str, LoadedModel] = {}

        self.path.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key: str) -> LoadedModel:
        return self.models[key]

    async def load_existing(self):
        model_dir = self.path
        for pkl in model_dir.glob("*"):
            id = pkl.stem
            _ = await self.load_model(id)

        # Default models
        ALGORITHMS = {
            "knn": ModelParams(
                algo=AlgoKnn(n_neighbours=3),
                random_state=1,
                features=DEFAULT_FEATURE_SET,
            ),
            "rf": ModelParams(
                algo=AlgoRf(n_estimators=100),
                random_state=1,
                features=DEFAULT_FEATURE_SET,
            ),
            "svm": ModelParams(
                algo=AlgoSvm(), random_state=1, features=DEFAULT_FEATURE_SET
            ),
            "lr": ModelParams(
                algo=AlgoLr(), random_state=1, features=DEFAULT_FEATURE_SET
            ),
        }

        # TODO: use the set of features from the requirements
        for key, params in ALGORITHMS.items():
            if key in self.models:
                continue
            _ = await self.train_model(key, params)

    async def load_model(self, model_id: str):
        model_path = self.path / model_id
        if not model_path.exists():
            return False

        info_text = await to_thread((model_path / "info.json").read_text)
        info = ModelInfo.model_validate_json(info_text)
        params_text = await to_thread((model_path / "params.json").read_text)
        params = ModelParams.model_validate_json(params_text)
        desc = Model(id=model_id, params=params, info=info)

        with (model_path / "model.pkl").open("rb") as f:
            impl = pickle.load(f)
            logger.info("Loaded model '%s' from %s", id, model_path / "model.pkl")

        model = LoadedModel(desc, impl)
        self.models[model_id] = model

        return True

    async def delete_model(self, model_id: str):
        model_path = self.path / model_id
        if not model_path.exists():
            logger.warning("Model %s does not exist", model_id)
            return False

        try:
            await to_thread(shutil.rmtree, model_path)
            del self.models[model_id]
            # self.models.pop(model_id, None)
            logger.info("Deleted model %s", model_id)
            return True
        except Exception as e:
            logger.error("Failed to delete model %s: %s", model_id, e)
            return False

    async def train_model(self, model_id: str, params: ModelParams) -> Model:
        model_path = self.path / model_id

        # Run train model in a separate process,
        # the process will save the model at model_path
        # TODO: some troubles with asyncio.create_subprocess_exec 
        res = await to_thread(
            subprocess.run,
            [
                sys.executable, "-m", train.__name__,
                str(self.dataset_path), str(model_path),
                params.model_dump_json(),
            ],
            check=True
        )

        if res.returncode != 0:
            # TODO: do something
            pass

        # Load the newly trained model from disk
        if not await self.load_model(model_id):
            # TODO: do something
            pass

        return self.models[model_id].desc
