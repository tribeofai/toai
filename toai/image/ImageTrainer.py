import time
from typing import Iterable, Optional

import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from .ImageTrainingCycle import ImageTrainingCycle
from .ImageLearner import ImageLearner
from ..data import DataContainer


@attr.s(auto_attribs=True)
class ImageTrainer:
    learner: ImageLearner
    data_container: DataContainer

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
            self.learner.fit(
                cycle.n_epochs,
                cycle.data,
                cycle.steps,
                self.data_container.validation,
                self.data_container.validation_steps,
            )
        end_time = time.time()

        eval_start_time = time.time()
        evaluation_results = self.evaluate(
            self.data_container.validation,
            self.data_container.validation_steps,
            verbose=0,
        )
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

    def evaluate(
        self, dataset: tf.data.Dataset, steps: Optional[int] = None, verbose: int = 1
    ):
        return self.learner.model.evaluate(dataset, steps=steps, verbose=verbose)

    def predict(
        self, dataset: tf.data.Dataset, steps: Optional[int] = None, verbose: int = 0
    ):
        return self.learner.model.predict(dataset, steps=steps, verbose=verbose)

    def report(
        self, dataset: tf.data.Dataset, steps: Optional[int] = None, verbose: int = 0
    ):
        return classification_report(
            [label.numpy() for _, label in dataset.take(steps).unbatch()],
            self.learner.model.predict(dataset, steps=steps).argmax(axis=1),
        )

    def analyse(
        self, dataset: tf.data.Dataset, steps: Optional[int] = None, verbose: int = 0
    ):
        reverse_label_map = {
            value: key for key, value in self.data_container.label_map.items()
        }
        images = []
        label_codes = []
        for image, label_code in dataset.take(steps).unbatch():
            label_codes.append(label_code.numpy())
            images.append(image.numpy())
        labels = [reverse_label_map[label_code] for label_code in label_codes]
        probs = self.learner.model.predict(dataset, steps=steps)
        pred_codes = probs.argmax(axis=1)
        preds = [reverse_label_map[pred_code] for pred_code in pred_codes]
        return pd.DataFrame.from_dict(
            {
                "image": images,
                "label": labels,
                "label_code": label_codes,
                "pred": preds,
                "pred_code": pred_codes,
                "label_probs": probs[:, label_codes][np.eye(len(labels), dtype=bool)],
                "pred_probs": probs[:, pred_codes][np.eye(len(pred_codes), dtype=bool)],
            }
        )

    def show_predictions(
        self,
        dataset: tf.data.Dataset,
        steps: int,
        correct: bool = False,
        ascending: bool = True,
        rows: int = 4,
        cols: int = 4,
    ):
        df = self.analyse(dataset=dataset, steps=steps)
        df = df[(df.label == df.pred) if correct else (df.label != df.pred)]
        df.sort_values(by=["label_probs"], ascending=ascending, inplace=True)
        _, ax = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))
        for i, row in enumerate(df.head(cols * rows).itertuples()):
            idx = (i // cols, i % cols) if rows > 1 else i % cols
            ax[idx].axis("off")
            ax[idx].imshow(row.image)
            ax[idx].set_title(
                f"{row.label}\n{row.pred}\n{row.label_probs:.4f}\n{row.pred_probs:.4f}"
            )
