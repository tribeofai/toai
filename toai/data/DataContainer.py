from typing import Any


class DataContainer:
    def __init__(self, train: Any, validation: Any, test: Any):
        self.train = train
        self.validation = validation
        self.test = test
