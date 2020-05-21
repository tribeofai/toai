from typing import Dict, Optional

import tensorflow as tf


class DataContainer:
    def __init__(
        self,
        base: tf.data.Dataset,
        label_map: Dict[str, int],
        train: Optional[tf.data.Dataset] = None,
        train_steps: Optional[int] = None,
        validation: Optional[tf.data.Dataset] = None,
        validation_steps: Optional[int] = None,
        test: Optional[tf.data.Dataset] = None,
        test_steps: Optional[int] = None,
        n_classes: Optional[int] = None,
    ):
        self.base = base
        self.train = train
        self.train_steps = train_steps
        self.validation = validation
        self.validation_steps = validation_steps
        self.test = test
        self.test_steps = test_steps
        self.label_map = label_map
        if n_classes:
            self.n_classes = n_classes
        elif label_map:
            self.n_classes = len(label_map.keys())
