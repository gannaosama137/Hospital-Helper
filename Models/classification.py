import numpy as np
from preprocessing.cardio import clean_cardio_data, load_cardio_data
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import f1_score

def train_cardio_model(df_clean, test_size=0.2, random_state=42):
  # Split features and target
  X = df_clean.drop(columns=["cardio"])
  y = df_clean["cardio"]

  num_cols = [
      "age_years",
      "height",
      "weight",
      "ap_hi",
      "ap_lo",
      "bmi",
      "pulse_pressure",
      "map",
  ]
  cat_cols = [
      "gender",
      "cholesterol",
      "gluc",
      "smoke",
      "alco",
      "active",
      "bp_stage",
  ]

  # Preprocess numerical and categorical features
  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), num_cols),
          (
              "cat",
              OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
              cat_cols,
          ),
      ]
  )

  # Build Random Forest pipeline
  rf_pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("classifier", RandomForestClassifier(random_state=random_state)),
      ]
  )

  # Stratified train/test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state, stratify=y
  )

  # Hyperparameter search space with bounded depth for reasonable runtime
  param_dist = {
      "classifier__n_estimators": [150, 250, 350],
      "classifier__max_depth": [8, 10, 12, 16],
      "classifier__min_samples_split": [2, 5, 10],
      "classifier__min_samples_leaf": [1, 5, 10],
      "classifier__max_features": ["sqrt", "log2"],
  }

  search = RandomizedSearchCV(
      estimator=rf_pipeline,
      param_distributions=param_dist,
      n_iter=10,
      scoring="roc_auc",
      cv=3,
      random_state=random_state,
      n_jobs=-1,
  )
  
  search.fit(X_train, y_train)

  best_model = search.best_estimator_

  # Predictions
  y_pred = best_model.predict(X_test)
  y_prob = best_model.predict_proba(X_test)[:, 1]

  print("Best parameters:", search.best_params_)
  print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
  print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
  print("\nClassification Report:\n", classification_report(y_test, y_pred))

  # Threshold = 0.45
  y_pred_045 = (y_prob >= 0.45).astype(int)

  print(
      f"At threshold=0.45 -> Accuracy: {accuracy_score(y_test, y_pred_045):.4f} | "
      f"Recall(1): {recall_score(y_test, y_pred_045):.4f} | "
      f"Precision(1): {precision_score(y_test, y_pred_045):.4f}"
  )

  # Metrics
  metrics = pd.DataFrame([{
      "Model": "Random Forest",
      "Accuracy": accuracy_score(y_test, y_pred_045),
      "Precision": precision_score(y_test, y_pred_045),
      "Recall": recall_score(y_test, y_pred_045),
      "F1-score": f1_score(y_test, y_pred_045, average="macro"),
      "ROC-AUC": roc_auc_score(y_test, y_prob),
  }])

  # Save model and metrics
  MODELS_DIR = Path(__file__).resolve().parent.parent / "Save_Models"
  MODELS_DIR.mkdir(parents=True, exist_ok=True)

  joblib.dump(best_model, MODELS_DIR / "cardio_classification_model.joblib")
  metrics.to_csv(MODELS_DIR / "classification_results.csv", index=False)

  print("\nCardio model and metrics saved successfully!")

  return best_model


if __name__ == "__main__":
  raw_df = load_cardio_data()
  processed_df = clean_cardio_data(raw_df)
  train_cardio_model(processed_df)