
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "Data" / "LengthOfStay.csv"

POSSIBLE_MODEL_DIRS = [
    BASE_DIR / "Models" / "Save_Models",
    BASE_DIR / "Models" / "models",
    BASE_DIR / "models",
]

MODELS_DIR = next(
    (path for path in POSSIBLE_MODEL_DIRS if path.exists()),
    POSSIBLE_MODEL_DIRS[0]
)


# ==========================================
# 2. FEATURES CONFIGURATION
# ==========================================

CATEGORICAL_FEATURES = [
    "rcount",
    "gender",
    "facid",
]

BINARY_FEATURES = [
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
]

NUMERIC_FEATURES = [
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

ALL_FEATURES = (
    CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + NUMERIC_FEATURES
)


# ==========================================
# 3. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Hospital Helper | LOS Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 4. CUSTOM STYLE
# ==========================================

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #176B87;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
    }

    .result-box {
        padding: 22px;
        border-radius: 14px;
        background-color: #e8f4f8;
        border-left: 5px solid #176B87;
    }

    .result-number {
        font-size: 42px;
        font-weight: bold;
        color: #176B87;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 5. LOAD DATA AND MODELS
# ==========================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(model_path):

    return joblib.load(model_path)


data = load_dataset()


# ==========================================
# 6. MODEL SELECTION
# ==========================================

RESULTS_PATH = None

for filename in ["results.csv", "model_comparison.csv"]:

    candidate = MODELS_DIR / filename

    if candidate.exists():
        RESULTS_PATH = candidate
        break


MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "KNN": "knn.joblib",
}

best_model_name = "Best Regression Model"

best_model_path = MODELS_DIR / "best_model.joblib"

# Prefer the saved best model if available
if not best_model_path.exists():

    best_model_path = None

    if RESULTS_PATH is not None:

        results = pd.read_csv(RESULTS_PATH)

        if "RMSE" in results.columns and "Model" in results.columns:

            best_row = results.loc[results["RMSE"].idxmin()]
            best_model_name = str(best_row["Model"])

            for model_name, filename in MODEL_FILES.items():

                if model_name.lower() == best_model_name.lower():

                    candidate = MODELS_DIR / filename

                    if candidate.exists():
                        best_model_path = candidate

    if best_model_path is None:

        for filename in MODEL_FILES.values():

            candidate = MODELS_DIR / filename

            if candidate.exists():
                best_model_path = candidate
                best_model_name = Path(filename).stem.replace(
                    "_", " "
                ).title()
                break

else:

    name_path = MODELS_DIR / "best_model_name.txt"

    if name_path.exists():
        best_model_name = name_path.read_text().strip()


if best_model_path is None:

    st.error(
        "No trained regression model found. "
        "Please train the models first and save them "
        "inside Models/Save_Models."
    )

    st.stop()


try:

    model = load_model(str(best_model_path))

except Exception as error:

    st.error(f"Could not load the trained model: {error}")
    st.stop()


# ==========================================
# 7. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🏥 Hospital Helper</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Length of Stay Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Predict the expected hospital stay duration "
    "using patient medical information and machine learning."
)


# ==========================================
# 8. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🏥 Hospital Helper")

    st.write("Healthcare Prediction System")

    st.divider()

    st.subheader("Model Information")

    st.write(f"**Active Model:** {best_model_name}")

    st.write("**Task:** Regression")

    st.write("**Target:** Length of Stay")

    st.write("**Unit:** Days")

    st.divider()

    st.caption(
        "This system is designed to support "
        "healthcare data analysis."
    )


# ==========================================
# 9. TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "🩺 Patient Prediction",
    "📊 Models Comparison",
    "ℹ️ About Project",
])


# ==========================================
# TAB 1: PATIENT PREDICTION
# ==========================================

with tab1:

    st.header("Patient Information")

    st.write(
        "Enter the patient's medical information "
        "to estimate the expected length of hospital stay."
    )

    if data is None:

        st.error(
            "Dataset not found. Please check Data/LengthOfStay.csv."
        )
        st.stop()

    missing_columns = [
        col for col in ALL_FEATURES
        if col not in data.columns
    ]

    if missing_columns:

        st.error(
            "Some expected feature columns are missing "
            "from the dataset: "
            + ", ".join(missing_columns)
        )

        st.stop()

    with st.form("patient_prediction_form"):

        col1, col2, col3 = st.columns(3)

        row = {}

        # --------------------------------------
        # CATEGORICAL FEATURES
        # --------------------------------------

        with col1:

            st.subheader("Patient Details")

            for feature in CATEGORICAL_FEATURES:

                values = data[feature].dropna().unique().tolist()

                values = sorted(values, key=str)

                if not values:
                    st.error(f"No values found for {feature}")
                    st.stop()

                row[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    values,
                    key=f"input_{feature}"
                )

        # --------------------------------------
        # BINARY FEATURES
        # --------------------------------------

        with col2:

            st.subheader("Medical Conditions")

            for feature in BINARY_FEATURES:

                row[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    key=f"input_{feature}"
                )

        # --------------------------------------
        # NUMERIC FEATURES
        # --------------------------------------

        with col3:

            st.subheader("Medical Measurements")

            for feature in NUMERIC_FEATURES:

                median_value = pd.to_numeric(
                    data[feature],
                    errors="coerce"
                ).median()

                if pd.isna(median_value):
                    median_value = 0.0

                row[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(median_value),
                    key=f"input_{feature}"
                )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Length of Stay",
            use_container_width=True
        )

    if submitted:

        try:

            patient_data = pd.DataFrame(
                [row],
                columns=ALL_FEATURES
            )

            prediction = float(model.predict(patient_data)[0])

            st.divider()

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div>Predicted Length of Stay</div>
                        <div class="result-number">
                            {prediction:.2f}
                        </div>
                        <div>Days</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result_col2:

                st.metric(
                    "Active Regression Model",
                    best_model_name
                )

                st.metric(
                    "Estimated Stay",
                    f"{prediction:.2f} days"
                )

            st.success(
                "Prediction completed successfully!"
            )

            st.caption(
                "This is a machine learning estimate, "
                "not a medical decision or a guaranteed discharge date."
            )

        except Exception as error:

            st.error(f"Prediction error: {error}")


# ==========================================
# TAB 2: MODELS COMPARISON
# ==========================================

with tab2:

    st.header("Regression Models Performance")

    st.write(
        "Compare the performance of the trained "
        "regression models."
    )

    if RESULTS_PATH is None:

        st.warning(
            "Model comparison file was not found. "
            "Please generate results.csv or model_comparison.csv "
            "from regression.py."
        )

    else:

        results = pd.read_csv(RESULTS_PATH)

        st.subheader("Performance Metrics")

        st.dataframe(
            results.round(4),
            use_container_width=True,
            hide_index=True
        )

        if "RMSE" in results.columns and "Model" in results.columns:

            st.subheader("RMSE Comparison")

            st.bar_chart(
                results.set_index("Model")[["RMSE"]]
            )

        if "MAE" in results.columns:

            st.subheader("MAE Comparison")

            st.bar_chart(
                results.set_index("Model")[["MAE"]]
            )

        if "R2" in results.columns:

            st.subheader("R² Score Comparison")

            st.bar_chart(
                results.set_index("Model")[["R2"]]
            )

        if "RMSE" in results.columns:

            best_row = results.loc[results["RMSE"].idxmin()]

            st.success(
                f"Best Model: {best_row['Model']} "
                f"| RMSE: {best_row['RMSE']:.4f}"
            )


# ==========================================
# TAB 3: ABOUT PROJECT
# ==========================================

with tab3:

    st.header("About Hospital Helper")

    st.write("""
    **Hospital Helper** is a machine learning project
    designed to predict the Length of Stay (LOS)
    for hospitalized patients.

    The system uses patient medical information,
    health measurements, and hospital-related features
    to estimate the expected number of hospital stay days.
    """)

    st.subheader("Project Objectives")

    st.markdown("""
    - Predict patient Length of Stay.
    - Apply machine learning regression algorithms.
    - Compare different models using evaluation metrics.
    - Support healthcare planning and resource management.
    """)

    st.subheader("Machine Learning Models")

    st.markdown("""
    1. Decision Tree Regressor
    2. Random Forest Regressor
    3. K-Nearest Neighbors Regressor
    """)

    st.subheader("Evaluation Metrics")

    st.markdown("""
    - **MAE:** Mean Absolute Error.
    - **RMSE:** Root Mean Squared Error.
    - **R²:** Coefficient of Determination.
    """)

    st.divider()

    st.caption(
        "Hospital Helper | Machine Learning Regression Project"
    )
