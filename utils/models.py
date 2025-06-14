import logging
import pickle
import shutil
import subprocess
import sys
from asyncio import to_thread
from copy import copy
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
    DatasetFeature,
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
            "knn": ModelParams[DatasetFeature](
                algo=AlgoKnn(n_neighbours=3),
                random_state=1,
                features=DEFAULT_FEATURE_SET,
            ),
            "rf": ModelParams[DatasetFeature](
                algo=AlgoRf(n_estimators=100),
                random_state=1,
                features=DEFAULT_FEATURE_SET,
            ),
            "svm": ModelParams[DatasetFeature](
                algo=AlgoSvm(), random_state=1, features=DEFAULT_FEATURE_SET
            ),
            "lr": ModelParams[DatasetFeature](
                algo=AlgoLr(), random_state=1, features=DEFAULT_FEATURE_SET
            ),
        }

        for key, params in ALGORITHMS.items():
            if key not in self.models:
                _ = await self.train_model(key, params)
            self.models[key].desc.removable = False

        # Set default model
        default_model = self.models["knn"]
        default_model_desc = copy(default_model.desc)
        default_model_desc.id = "default"
        self.models["default"] = LoadedModel(
            desc=default_model_desc, impl=default_model.impl
        )

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
            return

        try:
            await to_thread(shutil.rmtree, model_path)
            del self.models[model_id]
            # self.models.pop(model_id, None)
            logger.info("Deleted model %s", model_id)
        except Exception as e:
            logger.error("Failed to delete model %s: %s", model_id, e)
            raise e

    async def train_model(
        self, model_id: str, params: ModelParams[DatasetFeature]
    ) -> Model:
        model_path = self.path / model_id

        # Run train model in a separate process,
        # the process will save the model at model_path
        # TODO: some troubles with asyncio.create_subprocess_exec
        res = await to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                train.__name__,
                str(self.dataset_path),
                str(model_path),
                params.model_dump_json(),
            ],
            check=True,
        )

        if res.returncode != 0:
            # TODO: do something
            pass

        # Load the newly trained model from disk
        if not await self.load_model(model_id):
            # TODO: do something
            pass

        return self.models[model_id].desc
