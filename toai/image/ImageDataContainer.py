import attr

from ..data import DataContainer
from .ImageDataset import ImageDataset


@attr.s(auto_attribs=True)
class ImageDataContainer(DataContainer):
    train: ImageDataset
    validation: ImageDataset
    test: ImageDataset
