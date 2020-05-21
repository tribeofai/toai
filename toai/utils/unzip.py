import os
import shutil
from pathlib import Path
from typing import Union, Optional


def all_files_in_dir(
    path: Union[Path, str], extension: Optional[str] = None, keep_original: bool = False
):
    path = Path(path)
    pathstring = str(path)
    for filename in os.listdir(pathstring):
        if extension and not filename.endswith(extension):
            continue
        shutil.unpack_archive(filename=str(path / filename), extract_dir=pathstring)
        if not keep_original:
            os.remove(str(path / filename))
