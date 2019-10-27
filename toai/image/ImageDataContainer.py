import attr

from ..data import DataContainer
from .ImageDataBundle import ImageDataBundle


@attr.s(auto_attribs=True)
class ImageDataContainer(DataContainer):
    train: ImageDataBundle
    validation: ImageDataBundle
    test: ImageDataBundle
