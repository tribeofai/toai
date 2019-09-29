import os
import shutil
from pathlib import Path


def all_files_in_dir(path, extension=None, keep_original=False):
    path = Path(path)
    pathstring = str(path)
    for filename in os.listdir(pathstring):
        if extension and not filename.endswith(extension):
            continue
        shutil.unpack_archive(filename=str(path / filename), extract_dir=pathstring)
        if not keep_original:
            os.remove(str(path / filename))
