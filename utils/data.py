from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Data:
    train: pd.DataFrame
    test: pd.DataFrame


def load_data(path: Path):
    return Data(train=pd.DataFrame({"age": [40]}), test=pd.DataFrame({"age": [40]}))
