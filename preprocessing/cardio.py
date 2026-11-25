import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "cardio_train.csv"


def load_cardio_data(path=DATA_PATH, sep=";"):
    return pd.read_csv(path, sep=sep)


def clean_cardio_data(df, verbose=True):
  # Record shape before cleaning
  before_shape = df.shape

  # Drop id column and remove duplicate rows
  df_clean = df.drop(columns=["id"]).drop_duplicates().copy()

  # Feature engineering
  df_clean["age_years"] = df_clean["age"] / 365.25
  df_clean["bmi"] = df_clean["weight"] / ((df_clean["height"] / 100.0) ** 2)
  df_clean["pulse_pressure"] = df_clean["ap_hi"] - df_clean["ap_lo"]
  df_clean["map"] = df_clean["ap_lo"] + (df_clean["pulse_pressure"] / 3.0)

  bp_conditions = [
      (df_clean["ap_hi"] < 120) & (df_clean["ap_lo"] < 80),
      (df_clean["ap_hi"] >= 120)
      & (df_clean["ap_hi"] <= 129)
      & (df_clean["ap_lo"] < 80),
      ((df_clean["ap_hi"] >= 130) & (df_clean["ap_hi"] <= 139))
      | ((df_clean["ap_lo"] >= 80) & (df_clean["ap_lo"] <= 89)),
  ]
  df_clean["bp_stage"] = np.select(bp_conditions, [0, 1, 2], default=3)

  # Remove physiologically implausible outliers
  valid_mask = (
      (df_clean["ap_hi"] >= 70)
      & (df_clean["ap_hi"] <= 240)
      & (df_clean["ap_lo"] >= 40)
      & (df_clean["ap_lo"] <= 150)
      & (df_clean["ap_hi"] > df_clean["ap_lo"])
      & (df_clean["height"] >= 120)
      & (df_clean["height"] <= 215)
      & (df_clean["weight"] >= 35)
      & (df_clean["weight"] <= 200)
  )
  df_clean = df_clean[valid_mask].drop(columns=["age"]).reset_index(drop=True)

  after_shape = df_clean.shape
  removed_rows = before_shape[0] - after_shape[0]

  if verbose:
    print(f"Before cleaning : {before_shape[0]} rows, {before_shape[1]} columns")
    print(f"After cleaning  : {after_shape[0]} rows, {after_shape[1]} columns")
    print(f"Removed rows    : {removed_rows}")

  return df_clean


if __name__ == "__main__":
  raw_df = load_cardio_data()
  processed_df = clean_cardio_data(raw_df)





def clustering_preprocessing(df):

    # Select features for clustering
    cluster_features = ['gender', 'age', 'bmi', 'cholesterol', 'gluc',
                         'ap_hi', 'ap_lo' , 'smoke' , 'alco', 'active']


    # Remove ID and convert age from days to years
    df = df.drop(columns=['id'])
    df['age'] = (df['age'] / 365.25).round(1) 


    # Filter invalid medical measurements
    valid = (
        df['ap_hi'].between(70 , 250) & 
        df['ap_lo'].between(40 , 200) &
        (df['ap_hi'] > df['ap_lo'])   & 
        df['height'].between(120 , 220) &
        df['weight'].between(40 , 200)
    )
    print ("Percentage of rows removed: " , (df.shape[0] - valid.sum()) / df.shape[0] * 100 )
    df = df[valid].reset_index(drop=True)


    # Calculate BMI 
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df = df[df['bmi'].between(10, 50)]

    return df , cluster_features 


