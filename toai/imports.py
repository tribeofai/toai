# pylama:ignore=W0611,W0401

import copy
import json
import math
import pickle  # nosec
import re
import shutil
import time
import warnings
from collections import defaultdict, namedtuple
from functools import partial, reduce
from pathlib import Path
from typing import *

import attr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling
import requests
import seaborn as sns
import sklearn as sk
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    Binarizer,
    LabelEncoder,
    MinMaxScaler,
    MultiLabelBinarizer,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, OneClassSVM

import lightgbm as lgb
import skopt
from fastprogress import master_bar, progress_bar

try:
    import kaggle
except OSError as error:
    warnings.warn(str(error))
