# pylama:ignore=W0611

import copy
import json
import math
import pickle
import shutil
import time
from collections import defaultdict, namedtuple
from functools import partial, reduce
from pathlib import Path

import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
    PolynomialFeatures,
    Normalizer,
    MultiLabelBinarizer,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from tensorflow import keras

from fastprogress import master_bar, progress_bar
