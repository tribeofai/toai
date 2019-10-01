import os
import shutil
from pathlib import Path
from typing import Union

import requests


def download_file(
    url: str, path: Union[Path, str] = Path("."), override: bool = False
) -> str:
    local_filename = f"{str(path)}/{url.split('/')[-1]}"
    if os.path.exists(local_filename) and not override:
        raise FileExistsError(f"File exists: {local_filename}")
    with requests.get(url, stream=True) as r:
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename
