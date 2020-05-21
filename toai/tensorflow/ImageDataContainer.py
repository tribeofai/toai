import matplotlib.pyplot as plt

from .DataContainer import DataContainer


class ImageDataContainer(DataContainer):
    def show(self, mode: str, rows: int = 3, cols: int = 6, debug: bool = False):
        if debug and self.label_map:
            reverse_label_map = {value: key for key, value in self.label_map.items()}
        figsize = (4 * cols, 5 * rows) if debug else (3 * cols, 3 * rows)
        _, ax = plt.subplots(rows, cols, figsize=figsize)

        for i, (x, y) in enumerate(getattr(self, mode).unbatch().take(rows * cols)):
            x = x.numpy()
            y = y.numpy()
            idx = (i // cols, i % cols) if rows > 1 else i % cols
            ax[idx].axis("off")
            ax[idx].imshow(x)
            if debug and self.label_map:
                title = (
                    f"{reverse_label_map[y][:40]}\nLabel code: {y}\nShape: {x.shape}"
                )
            elif debug:
                title = f"Label code: {y}\nShape: {x.shape}"
            else:
                title = y
            ax[idx].set_title(title)
