from dataclasses import dataclass, field

from typing import List


@dataclass
class DataParams:
    target_col: str
    n_cont_cols: List[str] = field(default_factory=list)
    nn_cont_cols: List[str] = field(default_factory=list)
    binary_cols: List[str] = field(default_factory=list)
    lc_cat_cols: List[str] = field(default_factory=list)
    hc_cat_cols: List[str] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=list)

    @property
    def cont_cols(self) -> List[str]:
        return self.n_cont_cols + self.nn_cont_cols

    @property
    def cat_cols(self) -> List[str]:
        return self.lc_cat_cols + self.hc_cat_cols

    @property
    def feature_cols(self) -> List[str]:
        return self.cont_cols + self.binary_cols + self.cat_cols
