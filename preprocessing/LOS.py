from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "Data" / "LengthOfStay.csv"

TARGET = "lengthofstay"


# ==========================================
# 2. FEATURE COLUMNS
# ==========================================

CAT_COLS = [
    "rcount",
    "gender",
    "facid",
]

NUM_COLS = [
    "dialysisrenalendstage",
    "asthma",
    "irondef",
    "pneum",
    "substancedependence",
    "psychologicaldisordermajor",
    "depress",
    "psychother",
    "fibrosisandother",
    "malnutrition",
    "hemo",
    "hematocrit",
    "neutrophils",
    "sodium",
    "glucose",
    "bloodureanitro",
    "creatinine",
    "bmi",
    "pulse",
    "respiration",
    "secondarydiagnosisnonicd9",
    "visit_month",
    "visit_dayofweek",
]


# ==========================================
# 3. LOAD AND CLEAN DATA
# ==========================================

def load_data(path=DATA_PATH):

    df = pd.read_csv(path)

    # Convert visit date to datetime
    vdate_dt = pd.to_datetime(
        df["vdate"],
        format="%m/%d/%Y",
        errors="raise"
    )

    # Extract useful date features
    df["visit_month"] = vdate_dt.dt.month
    df["visit_dayofweek"] = vdate_dt.dt.dayofweek

    # Handle invalid glucose values
    df.loc[df["glucose"] < 0, "glucose"] = np.nan

    # Convert target to numeric
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    # Remove rows with missing target
    df = df.dropna(subset=[TARGET])

    # Remove ID and data leakage columns
    df = df.drop(
        columns=[
            "eid",
            "vdate",
            "discharged"
        ],
        errors="ignore"
    )

    return df


# ==========================================
# 4. SPLIT FEATURES AND TARGET
# ==========================================

def split_xy(df):

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


# ==========================================
# 5. BUILD PREPROCESSING PIPELINE
# ==========================================

def build_preprocessor():

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop"
    )

    return preprocessor


# ==========================================
# 6. FEATURE SELECTION
# ==========================================

def build_feature_selector(k=20):

    selector = SelectKBest(
        score_func=f_regression,
        k=k
    )

    return selector