import pickle  # nosec
from typing import Any, Union
from pathlib import Path


def save_file(obj: Any, filename: Union[Path, str], mode: str = "wb") -> None:
    with open(str(filename), mode=mode) as f:
        pickle.dump(obj, f)
