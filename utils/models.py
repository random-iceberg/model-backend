from asyncio import to_thread
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemas import Model, ModelInfo, ModelParams
import train


@dataclass
class LoadedModel:
    desc: Model
    impl: Any  # sklearn something


class LoadedModels:
    def __init__(self, path: Path):
        self.path: Path = path
        self.models: dict[str, LoadedModel] = {}

        self.path.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key: str) -> LoadedModel:
        return self.models[key]

    async def load_existing(self):
        # TODO: implement
        pass

    async def load_model(self, model_id: str):
        model_path = self.path / model_id
        if not model_path.exists():
            return False

        info_text = await to_thread((model_path / "info.json").read_text)
        info = ModelInfo.model_validate_json(info_text)
        params_text = await to_thread((model_path / "params.json").read_text)
        params = ModelParams.model_validate_json(params_text)
        desc = Model(id=model_id, params=params, info=info)

        impl = None  # TODO: load path/model.pkl

        model = LoadedModel(desc, impl)
        self.models[model_id] = model

        return True

    async def delete_model(self, model_id: str):
        pass

    async def train_model(self, model_id: str, params: ModelParams) -> Model:
        model_path = self.path / model_id

        # Run train model in a separate process,
        # the process will save the model at model_path
        process = await asyncio.create_subprocess_exec(
            "python", "-m", train.__name__, model_path, params.model_dump_json()
        )
        retcode = await process.wait()
        if retcode != 0:
            # TODO: do something
            pass

        # Load the newly trained model from disk
        if not await self.load_model(model_id):
            # TODO: do something
            pass

        return self.models[model_id].desc
