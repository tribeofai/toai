import attr

import tensorflow as tf


@attr.s(auto_attribs=True)
class ImageParser:
    n_channels: int = 3

    def __call__(self, filename: str) -> tf.Tensor:
        image = tf.image.decode_jpeg(
            tf.io.read_file(filename), channels=self.n_channels
        )
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image
