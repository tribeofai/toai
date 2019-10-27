import attr

from .DataBundle import DataBundle


@attr.s(auto_attribs=True)
class DataContainer:
    train: DataBundle
    validation: DataBundle
    test: DataBundle
