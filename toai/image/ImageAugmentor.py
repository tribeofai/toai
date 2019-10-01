import attr
from typing import Optional
import tensorflow as tf


@attr.s(auto_attribs=True)
class ImageAugmentor:
    level: int = 0
    flips: Optional[str] = None

    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        if self.flips in ["horizontal", "both"]:
            image = tf.image.random_flip_left_right(image)
        if self.flips in ["vertical", "both"]:
            image = tf.image.random_flip_up_down(image)

        if self.level > 0:
            lower = 1 - 0.1 * self.level
            upper = 1 + 0.1 * self.level
            min_jpeg_quality = max(0, int((lower - 0.5) * 100))
            max_jpeg_quality = min(100, int((upper - 0.5) * 100))
            image = tf.image.random_jpeg_quality(
                image,
                min_jpeg_quality=min_jpeg_quality,
                max_jpeg_quality=max_jpeg_quality,
            )
            image = tf.image.random_contrast(image, lower=lower, upper=upper)
            image = tf.image.random_saturation(image, lower=lower, upper=upper)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image
