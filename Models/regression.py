import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing.LOS import (
    load_data,
    split_xy,
    build_preprocessor,
    build_feature_selector
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data"
PLOT_DIR = BASE_DIR / "Plots"
MODELS_DIR = BASE_DIR / "Save_Models"

PLOT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

file_path = DATA_DIR / "LengthOfStay.csv"



RANDOM_STATE = 100


# ---------- 2) data ----------

df = load_data()
X, y = split_xy(df)

print("Dataset Shape:", df.shape)
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ---------- 3) all the regression algorithms ----------

models = {
    "Decision Tree": DecisionTreeRegressor(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "KNN": KNeighborsRegressor(),
}


# ---------- 4) hyperparameter search ----------

param_grids = {

    "Decision Tree": {
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": [None, "sqrt", 0.8],
    },

    "Random Forest": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [8, 12, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 5],
        "model__max_features": [0.7, 0.8, 1.0, "sqrt"],
    },

    "KNN": {
        "model__n_neighbors": [5, 7, 9, 11, 15, 21, 25],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
        "model__leaf_size": [20, 30, 40, 50],
    }
}


# ---------- 5) train + evaluate ----------

results = []
trained = {}

for name, estimator in models.items():

    print("\n" + "=" * 50)
    print(f"Training: {name}")
    print("=" * 50)

    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("feature_selection", build_feature_selector(k=20)),
        ("model", estimator),
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grids[name],
        n_iter=10,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Best CV RMSE": -search.best_score_,
        "Best Parameters": search.best_params_,
    })

    trained[name] = best_model

    print("\nBest Parameters:")
    print(search.best_params_)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    print(f"done: {name}")


# ---------- 6) compare ----------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE",
    ascending=True
).reset_index(drop=True)

print("\nFINAL MODEL COMPARISON")
print(
    results_df[
        ["Model", "MAE", "RMSE", "R2", "Best CV RMSE"]
    ].round(4).to_string(index=False)
)


# ---------- 7) best model ----------

best_name = results_df.loc[0, "Model"]

print(f"\nBest model (by RMSE): {best_name}")

print(f"Best RMSE: {results_df.loc[0, 'RMSE']:.4f}")
print(f"Best R2: {results_df.loc[0, 'R2']:.4f}")


# ---------- 8) save ----------

results_df.to_csv(
    MODELS_DIR / "regression_results.csv",
    index=False
)

joblib.dump(
    trained[best_name],
    MODELS_DIR / "hospital_regression_model.joblib"
)

(MODELS_DIR / "best_regression_model_name.txt").write_text(best_name)

print("\nAll regression models trained and saved successfully!")