import pickle  # nosec
from typing import Any, Union
from pathlib import Path


def load_file(filename: Union[Path, str], mode: str = "rb") -> Any:
    with open(str(filename), mode=mode) as f:
        return pickle.load(f)  # nosec
