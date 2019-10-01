import attr

from .Dataset import Dataset


@attr.s(auto_attribs=True)
class DataContainer:
    train: Dataset
    validation: Dataset
    test: Dataset
