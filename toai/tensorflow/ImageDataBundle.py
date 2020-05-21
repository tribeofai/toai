import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from ..data import DataBundle


class ImageDataBundle(DataBundle):
    @classmethod
    def from_subfolders(cls, path: Union[Path, str]) -> "ImageDataBundle":
        path = Path(path)
        paths = []
        labels = []
        for label in os.listdir(path):
            for image_path in os.listdir(path / label):
                paths.append(str(path / label / image_path))
                labels.append(label)

        return cls(np.asarray(paths), np.asarray(labels))

    @classmethod
    def from_re(
        cls,
        path: Union[Path, str],
        regex: str,
        default: Optional[Union[int, float, str, bool]] = None,
    ) -> "ImageDataBundle":
        paths = []
        labels = []
        for value in os.listdir(path):
            match = re.match(regex, value)
            if match:
                labels.append(match.group(1))
            elif default:
                labels.append(str(default))
            else:
                raise ValueError(
                    f"No match found and no default value provided for value: {value}"
                )
            paths.append(f"{path}/{value}")

        return cls(np.asarray(paths), np.asarray(labels))
