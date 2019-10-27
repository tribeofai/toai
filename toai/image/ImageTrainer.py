import time
from typing import Iterable

import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .ImageTrainingCycle import ImageTrainingCycle
from .ImageLearner import ImageLearner
from .ImageDataContainer import ImageDataContainer


@attr.s(auto_attribs=True)
class ImageTrainer:
    learner: ImageLearner
    data_container: ImageDataContainer

    def train(
        self,
        cycles: Iterable[ImageTrainingCycle],
        template: str = (
            "Name: {} Train Time: {:.1f} min. "
            "Eval Time: {:.2f}s Loss: {:.4f} Accuracy: {:.2%}"
        ),
    ):
        start_time = time.time()
        for cycle in cycles:
            self.learner.freeze() if cycle.freeze else self.learner.unfreeze()
            self.learner.compile(optimizer=cycle.optimizer, lr=cycle.lr)
            self.data_container.train.image_pipeline = cycle.feature_pipeline
            self.data_container.train.preprocess()
            self.learner.fit(
                self.data_container.train,
                self.data_container.validation,
                cycle.n_epochs,
            )
        end_time = time.time()

        eval_start_time = time.time()
        evaluation_results = self.evaluate_data_bundle(verbose=0)
        eval_end_time = time.time()

        print("-".center(80, "-"))
        print(
            template.format(
                self.learner.base_model.name,
                (end_time - start_time) / 60,
                (eval_end_time - eval_start_time),
                *evaluation_results,
            )
        )
        print("-".center(80, "-"))

    def evaluate_data_bundle(self, mode: str = "validation", verbose: int = 1):
        data_bundle = getattr(self.data_container, mode)
        return self.learner.model.evaluate(
            data_bundle.data, steps=data_bundle.steps, verbose=verbose
        )

    def predict_data_bundle(self, mode: str = "validation", verbose: int = 0):
        data_bundle = getattr(self.data_container, mode)
        return self.learner.model.predict(
            data_bundle.data, steps=data_bundle.steps, verbose=verbose
        )

    def analyse_data_bundle(self, mode: str = "validation", verbose: int = 0):
        data_bundle = getattr(self.data_container, mode)
        image_ds = tf.data.Dataset.from_tensor_slices(data_bundle.x)
        image_ds = data_bundle.preprocess_with_pipeline(
            image_ds, data_bundle.image_pipeline
        ).batch(1)
        images = [
            img[0].numpy()
            for img in image_ds.take(data_bundle.steps * data_bundle.batch_size)
        ]
        probs = self.learner.model.predict(image_ds)
        pred_code = probs.argmax(axis=1)
        label_code = [data_bundle.label_map[label] for label in data_bundle.y]
        inverse_label_map = {value: key for key, value in data_bundle.label_map.items()}
        pred = [inverse_label_map[x] for x in pred_code]
        return pd.DataFrame.from_dict(
            {
                "path": data_bundle.x,
                "image": images,
                "label": data_bundle.y,
                "label_code": label_code,
                "pred": pred,
                "pred_code": pred_code,
                "label_probs": probs[:, label_code][
                    np.eye(len(data_bundle.y), dtype=bool)
                ],
                "pred_probs": probs[:, pred_code][np.eye(len(pred_code), dtype=bool)],
            }
        )

    def show_predictions(
        self,
        mode: str = "validation",
        correct: bool = False,
        ascending: bool = True,
        cols: int = 8,
        rows: int = 2,
    ):
        df = self.analyse_data_bundle(mode=mode)
        df = df[(df.label == df.pred) if correct else (df.label != df.pred)]
        df.sort_values(by=["label_probs"], ascending=ascending, inplace=True)
        _, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
        for i, row in enumerate(df.head(cols * rows).itertuples()):
            idx = (i // cols, i % cols) if rows > 1 else i % cols
            ax[idx].axis("off")
            ax[idx].imshow(row.image)
            ax[idx].set_title(
                f"{row.label}:{row.pred}\n{row.label_probs:.4f}:{row.pred_probs:.4f}"
            )
