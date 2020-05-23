from typing import List
import attr


@attr.s(auto_attribs=True)
class DataParams:
    target_col: str
    cat_cols: List[str] = attr.Factory(list)
    cont_cols: List[str] = attr.Factory(list)
    text_cols: List[str] = attr.Factory(list)
    img_cols: List[str] = attr.Factory(list)
    feature_cols: List[str] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.feature_cols = (
            self.cat_cols + self.cont_cols + self.text_cols + self.img_cols
        )
