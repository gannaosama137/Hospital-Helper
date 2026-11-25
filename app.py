from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing.LOS import CAT_COLS, NUM_COLS, TARGET, load_data
from preprocessing.cardio import clean_cardio_data, clustering_preprocessing


# =========================================================
# 1. PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "Data"
MODEL_DIR = ROOT / "Save_Models"
PLOT_DIR = ROOT / "Plots"

LOS_DATA_PATH = DATA_DIR / "LengthOfStay.csv"
CARDIO_DATA_PATH = DATA_DIR / "cardio_train.csv"

REGRESSION_MODEL_PATH = MODEL_DIR / "hospital_regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODEL_DIR / "cardio_classification_model.joblib"
CLUSTER_MODEL_PATH = MODEL_DIR / "cluster_model.joblib"

RESULTS_PATH = MODEL_DIR / "regression_results.csv"
CLASSIFICATION_RESULTS_PATH = MODEL_DIR / "classification_results.csv"
CARDIO_CLASSIFICATION_THRESHOLD = 0.45

IMPORTANCE_PLOT_PATH = PLOT_DIR / "feature_importance.png"
CLUSTER_PLOT_PATH = PLOT_DIR / "clustered_data.png"


# =========================================================
# 2. FEATURES AND LABELS
# =========================================================

LOS_BINARY_FEATURES = [
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

CLUSTER_LABELS = {
    0: "Younger · Typical values",
    1: "Older · Near-normal values",
    2: "Higher glucose & cholesterol",
    3: "Higher blood pressure & BMI",
}


# =========================================================
# 3. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hospital Helper",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 4. LIGHT THEME AND CSS
# =========================================================

st.markdown(
    """
    <style>

    :root,
    html,
    body,
    .stApp {
        color-scheme: light !important;
        --background-color: #f5f8fb !important;
        --secondary-background-color: #ffffff !important;
        --text-color: #17324d !important;
        --primary-color: #087e8b !important;
        --border-color: #e1e9ef !important;
    }

    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #f5f8fb !important;
        color: #17324d !important;
    }

    [data-testid="stHeader"] {
        display: none !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    #MainMenu,
    footer {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4, p, label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #17324d !important;
    }

    /* Metrics */

    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e1e9ef !important;
        border-radius: 14px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #647b91 !important;
    }

    [data-testid="stMetricValue"] {
        color: #17324d !important;
    }

    /* Buttons */

    .stButton > button,
    .stFormSubmitButton > button,
    button[kind="primary"] {
        background-color: #087e8b !important;
        color: #ffffff !important;
        border: 1px solid #087e8b !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    .stButton > button *,
    .stFormSubmitButton > button *,
    button[kind="primary"] * {
        color: #ffffff !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    button[kind="primary"]:hover {
        background-color: #05636d !important;
        border-color: #05636d !important;
        color: #ffffff !important;
    }

    /* Inputs */

    input, textarea {
        background-color: #ffffff !important;
        color: #17324d !important;
        border-color: #e1e9ef !important;
    }

    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #e1e9ef !important;
        color: #17324d !important;
    }

    [data-baseweb="select"] *,
    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        color: #17324d !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }

    /* Tabs */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        color: #17324d !important;
        border-radius: 9px 9px 0 0;
    }

    .stTabs [aria-selected="true"] {
        color: #087e8b !important;
        border-bottom: 3px solid #087e8b !important;
    }

    /* Alerts */

    [data-testid="stAlert"] {
        border-radius: 10px;
        background-color: #eaf4f7 !important;
        border: 1px solid #cfe1e6 !important;
        color: #17324d !important;
    }

    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
        color: #17324d !important;
    }

    /* DataFrames and Tables */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background-color: #ffffff !important;
        color: #17324d !important;
        border: 1px solid #e1e9ef !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] * {
        color: #17324d !important;
    }

    /* Charts */

    [data-testid="stVegaLiteChart"] {
        background-color: #ffffff !important;
        color: #17324d !important;
    }

    [data-testid="stVegaLiteChart"] svg text {
        fill: #17324d !important;
    }

    /* Dividers */

    hr {
        border-color: #e1e9ef !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. DATA LOADING
# =========================================================

@st.cache_data(show_spinner=False)
def get_los_data():
    return load_data(LOS_DATA_PATH)


@st.cache_data(show_spinner=False)
def get_cardio_data():

    raw = pd.read_csv(CARDIO_DATA_PATH, sep=";")

    return clean_cardio_data(
        raw.copy(),
        verbose=False
    )


@st.cache_data(show_spinner=False)
def get_cluster_data():

    raw = pd.read_csv(CARDIO_DATA_PATH, sep=";")

    cleaned, _ = clustering_preprocessing(
        raw.copy()
    )

    return cleaned


@st.cache_data(show_spinner=False)
def get_cluster_population(model_path):

    data = get_cluster_data()
    bundle = joblib.load(model_path)

    features = bundle["features"]

    missing = [
        feature for feature in features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing clustering features: " + ", ".join(missing)
        )

    scaled = bundle["scaler"].transform(data[features])

    population = data.copy()

    population["cluster"] = bundle["model"].predict(scaled)

    return population


@st.cache_resource(show_spinner=False)
def get_regression_model():
    return joblib.load(REGRESSION_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_classification_model(model_path, modified_at):
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def get_cluster_bundle():
    return joblib.load(CLUSTER_MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_data(show_spinner=False)
def get_classification_results():
    return pd.read_csv(CLASSIFICATION_RESULTS_PATH)


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================

def load_resource(path, loader, label):

    if not path.exists():
        st.warning(f"{label} is not available yet.")
        return None

    try:
        return loader()

    except Exception as error:
        st.error(f"Could not load {label}: {error}")
        return None


def median_value(data, column, default=0):

    if column not in data.columns:
        return default

    value = pd.to_numeric(
        data[column],
        errors="coerce"
    ).median()

    if pd.isna(value):
        return default

    return float(value)


def mode_value(data, column, default=0):

    if column not in data.columns:
        return default

    values = data[column].mode(dropna=True)

    if values.empty:
        return default

    return values.iloc[0]


def pressure_stage(systolic, diastolic):

    if systolic < 120 and diastolic < 80:
        return 0

    if 120 <= systolic <= 129 and diastolic < 80:
        return 1

    if 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        return 2

    return 3


def get_cluster_name(cluster_id):

    return CLUSTER_LABELS.get(
        int(cluster_id),
        f"Patient segment {int(cluster_id) + 1}"
    )


def get_model_features(model, fallback_features):

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return list(fallback_features)


def make_default_row(data, features):

    row = {}

    for feature in features:

        if feature not in data.columns:
            raise ValueError(
                f"Feature '{feature}' is missing from the dataset."
            )

        if pd.api.types.is_numeric_dtype(data[feature]):

            row[feature] = median_value(data, feature)

        else:

            row[feature] = mode_value(data, feature)

    return pd.DataFrame([row], columns=features)


# =========================================================
# 7. LOAD PROJECT RESOURCES
# =========================================================

los_data = load_resource(
    LOS_DATA_PATH,
    get_los_data,
    "Length of Stay data"
)

cardio_data = load_resource(
    CARDIO_DATA_PATH,
    get_cardio_data,
    "Cardiovascular data"
)

regression_model = load_resource(
    REGRESSION_MODEL_PATH,
    get_regression_model,
    "Regression model"
)

classification_model = load_resource(
    CLASSIFICATION_MODEL_PATH,
    lambda: get_classification_model(
        str(CLASSIFICATION_MODEL_PATH),
        CLASSIFICATION_MODEL_PATH.stat().st_mtime_ns,
    ),
    "Classification model"
)
classification_results = load_resource(
    CLASSIFICATION_RESULTS_PATH,
    get_classification_results,
    "Classification results"
)

cluster_bundle = load_resource(
    CLUSTER_MODEL_PATH,
    get_cluster_bundle,
    "Clustering model"
)

cluster_population = None
results_data = None

if cluster_bundle is not None:

    cluster_population = load_resource(
        CARDIO_DATA_PATH,
        lambda: get_cluster_population(str(CLUSTER_MODEL_PATH)),
        "Cluster population"
    )

if RESULTS_PATH.exists():

    try:
        results_data = get_results()

    except Exception as error:
        st.error(f"Could not load regression_results.csv: {error}")


# =========================================================
# 8. HEADER
# =========================================================

st.title("🏥 Hospital Helper")

st.subheader("Better context for better care.")

st.write(
    "Explore hospital stay estimates, cardiovascular model outputs, "
    "and patient segments in one simple workspace."
)

st.divider()


# =========================================================
# 9. OVERVIEW METRICS
# =========================================================

ready_models = sum([
    regression_model is not None,
    classification_model is not None,
    cluster_bundle is not None,
])

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Available models", f"{ready_models} / 3")

with metric2:
    st.metric(
        "Stay records",
        f"{len(los_data):,}" if los_data is not None else "N/A"
    )

with metric3:
    st.metric(
        "Heart-health records",
        f"{len(cardio_data):,}" if cardio_data is not None else "N/A"
    )

with metric4:
    st.metric(
        "Workspace",
        "Ready" if ready_models == 3 else "Check files"
    )


st.divider()


# =========================================================
# 10. TABS
# =========================================================

overview_tab, stay_tab, heart_tab, cluster_tab, lab_tab = st.tabs([
    "✦ Overview",
    "🏨 Length of Stay",
    "❤️ Heart Health Estimate",
    "🧩 Patient Segments",
    "📊 Model & Data Lab"
])


# =========================================================
# OVERVIEW
# =========================================================

with overview_tab:

    st.header("Welcome to Hospital Helper")

    st.write(
        "This application brings together three machine learning "
        "approaches for exploring hospital and cardiovascular data."
    )

    st.info(
        "Looking for the cardiovascular model? Open the "
        "**❤️ Heart Health Estimate** "
        "section above to enter a profile and view its estimated probability."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("🛏️ Length of Stay")

        st.write(
            "Regression model that estimates the expected "
            "duration of a hospital stay."
        )

        st.caption("Regression · continuous estimate")

    with col2:

        st.subheader("❤️ Cardiovascular Classifier")

        st.write(
            "Classification model that estimates the probability "
            "of the positive class."
        )

        st.caption("Classification · probability estimate")

    with col3:

        st.subheader("🧩 Patient Segments")

        st.write(
            "K-Means clustering model that groups similar "
            "patient records."
        )

        st.caption("K-Means · similarity-based segments")

    st.divider()

    st.info(
        "This application is an educational machine learning project. "
        "Its outputs are not medical diagnoses or treatment recommendations."
    )


# =========================================================
# CARDIOVASCULAR CLASSIFICATION
# =========================================================

with heart_tab:

    st.header("❤️ Cardiovascular Classifier")

    st.write(
        "Enter the patient's profile and measurements to obtain "
        "the model's estimated probability."
    )

    if classification_model is None or cardio_data is None:

        st.warning(
            "The current cardiovascular classifier or its dataset "
            "is unavailable."
        )

    else:

        st.info(
            f"Automatic cutoff: **{CARDIO_CLASSIFICATION_THRESHOLD:.2f}**, "
            "matching the current classifier training script. This is a "
            "model setting, not a clinical guideline."
        )

        cardio_fallback_features = [
            col for col in cardio_data.columns
            if col != "cardio"
        ]

        cardio_features = get_model_features(
            classification_model,
            cardio_fallback_features
        )

        missing_features = [
            feature for feature in cardio_features
            if feature not in cardio_data.columns
        ]

        if missing_features:

            st.error(
                "The classification model requires features not found "
                "in the cleaned dataset: "
                + ", ".join(missing_features)
            )

        else:

            with st.form("cardio_form"):

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.subheader("Personal Information")

                    age = st.number_input(
                        "Age (years)",
                        min_value=18.0,
                        max_value=110.0,
                        value=float(
                            np.clip(
                                median_value(cardio_data, "age_years", 50),
                                18,
                                110
                            )
                        ),
                        step=1.0
                    )

                    gender = st.selectbox(
                        "Gender",
                        [1, 2],
                        format_func=lambda x: (
                            "Female" if x == 1 else "Male"
                        )
                    )

                    height = st.number_input(
                        "Height (cm)",
                        min_value=120,
                        max_value=215,
                        value=int(
                            np.clip(
                                round(median_value(cardio_data, "height", 165)),
                                120,
                                215
                            )
                        )
                    )

                    weight = st.number_input(
                        "Weight (kg)",
                        min_value=35,
                        max_value=200,
                        value=int(
                            np.clip(
                                round(median_value(cardio_data, "weight", 70)),
                                35,
                                200
                            )
                        )
                    )

                with col2:

                    st.subheader("Lifestyle & Blood Tests")

                    cholesterol = st.selectbox(
                        "Cholesterol",
                        [1, 2, 3],
                        format_func=lambda x: {
                            1: "Normal",
                            2: "Above normal",
                            3: "Well above normal"
                        }[x]
                    )

                    gluc = st.selectbox(
                        "Glucose",
                        [1, 2, 3],
                        format_func=lambda x: {
                            1: "Normal",
                            2: "Above normal",
                            3: "Well above normal"
                        }[x]
                    )

                    smoke = st.selectbox(
                        "Smoking",
                        [0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes"
                    )

                    alco = st.selectbox(
                        "Alcohol use",
                        [0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes"
                    )

                    active = st.selectbox(
                        "Physical activity",
                        [1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No"
                    )

                with col3:

                    st.subheader("Blood Pressure")

                    systolic = st.number_input(
                        "Systolic pressure",
                        min_value=70,
                        max_value=240,
                        value=int(
                            np.clip(
                                round(median_value(cardio_data, "ap_hi", 120)),
                                70,
                                240
                            )
                        )
                    )

                    diastolic = st.number_input(
                        "Diastolic pressure",
                        min_value=40,
                        max_value=150,
                        value=int(
                            np.clip(
                                round(median_value(cardio_data, "ap_lo", 80)),
                                40,
                                150
                            )
                        )
                    )

                predict_cardio = st.form_submit_button(
                    "Estimate cardiovascular probability",
                    type="primary",
                    use_container_width=True
                )

            if predict_cardio:

                if systolic <= diastolic:

                    st.error(
                        "Systolic pressure must be greater than diastolic pressure."
                    )

                else:

                    bmi = weight / ((height / 100) ** 2)
                    pulse_pressure = systolic - diastolic
                    map_value = diastolic + pulse_pressure / 3
                    bp_stage = pressure_stage(systolic, diastolic)

                    try:

                        cardio_input = make_default_row(
                            cardio_data,
                            cardio_features
                        )

                        input_values = {
                            "age_years": age,
                            "gender": gender,
                            "height": height,
                            "weight": weight,
                            "ap_hi": systolic,
                            "ap_lo": diastolic,
                            "cholesterol": cholesterol,
                            "gluc": gluc,
                            "smoke": smoke,
                            "alco": alco,
                            "active": active,
                            "bmi": bmi,
                            "pulse_pressure": pulse_pressure,
                            "map": map_value,
                            "bp_stage": bp_stage,
                        }

                        for feature, value in input_values.items():

                            if feature in cardio_features:
                                cardio_input.loc[0, feature] = value

                        probabilities = classification_model.predict_proba(
                            cardio_input
                        )[0]

                        classes = list(classification_model.classes_)

                        positive_index = next(
                            (
                                i for i, value in enumerate(classes)
                                if str(value) == "1"
                            ),
                            None
                        )

                        if positive_index is None:

                            st.error(
                                "Could not identify positive class (1) "
                                "in the saved model."
                            )

                        else:

                            probability = float(
                                probabilities[positive_index]
                            )

                            predicted_class = int(
                                probability >= CARDIO_CLASSIFICATION_THRESHOLD
                            )

                            st.success("Model estimate generated.")

                            result_col1, result_col2 = st.columns(2)

                            with result_col1:

                                st.metric(
                                    "Positive-class probability",
                                    f"{probability:.1%}"
                                )

                            with result_col2:

                                st.metric(
                                    "At selected threshold",
                                    (
                                        "Further review suggested"
                                        if predicted_class
                                        else "Below cutoff — not cleared"
                                    )
                                )

                            st.progress(probability)

                            st.caption(
                                f"Model cutoff: "
                                f"{CARDIO_CLASSIFICATION_THRESHOLD:.2f}."
                            )

                            if predicted_class:
                                st.warning(
                                    "The model score meets the screening "
                                    "cutoff. This is not a diagnosis; seek "
                                    "qualified clinical review."
                                )
                            else:
                                st.warning(
                                    "A score below the cutoff does not rule "
                                    "out cardiovascular disease and must not "
                                    "be interpreted as being healthy."
                                )

                    except Exception as error:

                        st.error(
                            f"Classification prediction failed: {error}"
                        )


# =========================================================
# LENGTH OF STAY
# =========================================================

with stay_tab:

    st.header("🛏️ Hospital Stay Estimate")

    st.write(
        "Enter patient and visit information to estimate "
        "the expected length of hospital stay."
    )

    if regression_model is None or los_data is None:

        st.warning(
            "The regression model or Length of Stay data is unavailable."
        )

    else:

        los_features = get_model_features(
            regression_model,
            list(dict.fromkeys(CAT_COLS + NUM_COLS))
        )

        missing_features = [
            feature for feature in los_features
            if feature not in los_data.columns
        ]

        if missing_features:

            st.error(
                "Missing LOS features: " + ", ".join(missing_features)
            )

        else:

            with st.form("los_form"):

                col1, col2, col3 = st.columns(3)

                patient = {}

                with col1:

                    st.subheader("Visit Information")

                    for feature in CAT_COLS:

                        if feature not in los_features:
                            continue

                        values = (
                            los_data[feature]
                            .dropna()
                            .unique()
                            .tolist()
                        )

                        values = sorted(values, key=str)

                        if values:

                            patient[feature] = st.selectbox(
                                feature.replace("_", " ").title(),
                                values,
                                key=f"los_cat_{feature}"
                            )

                with col2:

                    st.subheader("Medical History")

                    for feature in LOS_BINARY_FEATURES:

                        if feature not in los_features:
                            continue

                        patient[feature] = st.selectbox(
                            feature.replace("_", " ").title(),
                            [0, 1],
                            format_func=lambda x: "Yes" if x else "No",
                            key=f"los_binary_{feature}"
                        )

                with col3:

                    st.subheader("Clinical Measurements")

                    for feature in NUM_COLS:

                        if feature not in los_features:
                            continue

                        if feature in LOS_BINARY_FEATURES:
                            continue

                        patient[feature] = st.number_input(
                            feature.replace("_", " ").title(),
                            value=median_value(los_data, feature),
                            key=f"los_num_{feature}"
                        )

                predict_los = st.form_submit_button(
                    "Predict Length of Stay",
                    type="primary",
                    use_container_width=True
                )

            if predict_los:

                try:

                    input_data = make_default_row(
                        los_data,
                        los_features
                    )

                    for feature, value in patient.items():
                        input_data.loc[0, feature] = value

                    prediction = float(
                        regression_model.predict(input_data)[0]
                    )

                    if not np.isfinite(prediction):

                        st.error("The model returned an invalid prediction.")

                    else:

                        days = max(1, int(np.ceil(prediction)))

                        st.success("Prediction completed.")

                        result_col1, result_col2 = st.columns(2)

                        with result_col1:

                            st.metric(
                                "Estimated Length of Stay",
                                f"{days} days"
                            )

                        with result_col2:

                            st.metric(
                                "Raw model output",
                                f"{prediction:.2f} days"
                            )

                        st.caption(
                            "The displayed estimate is rounded up to a whole "
                            "day. Actual hospital stay may differ."
                        )

                except Exception as error:

                    st.error(f"Prediction failed: {error}")


# =========================================================
# PATIENT CLUSTERING
# =========================================================

with cluster_tab:

    st.header("🧩 Patient Segments")

    st.write(
        "Explore the patient groups discovered by the K-Means model."
    )

    if cluster_bundle is None or cluster_population is None:

        st.warning(
            "The clustering model or patient data is unavailable."
        )

    elif not isinstance(cluster_bundle, dict):

        st.error("The clustering model has an invalid format.")

    elif not all(
        key in cluster_bundle
        for key in ["scaler", "model", "features"]
    ):

        st.error(
            "The clustering bundle must contain scaler, model, and features."
        )

    else:

        cluster_features = cluster_bundle["features"]

        missing_features = [
            feature for feature in cluster_features
            if feature not in cluster_population.columns
        ]

        if missing_features:

            st.error(
                "Missing clustering features: "
                + ", ".join(missing_features)
            )

        else:

            st.subheader("🔎 Find a Patient Segment")

            profile = {}

            numeric_features = [
                "age",
                "bmi",
                "ap_hi",
                "ap_lo",
            ]

            categorical_options = {
                "gender": [1, 2],
                "cholesterol": [1, 2, 3],
                "gluc": [1, 2, 3],
                "smoke": [0, 1],
                "alco": [0, 1],
                "active": [0, 1],
            }

            with st.form("cluster_form"):

                cols = st.columns(3)

                for index, feature in enumerate(cluster_features):

                    with cols[index % 3]:

                        if feature in numeric_features:

                            default = median_value(
                                cluster_population,
                                feature,
                                0
                            )

                            if feature in ["age", "bmi"]:

                                profile[feature] = st.number_input(
                                    feature.replace("_", " ").title(),
                                    min_value=0.0,
                                    value=float(max(0, default)),
                                    step=0.1,
                                    key=f"cluster_{feature}"
                                )

                            else:

                                profile[feature] = st.number_input(
                                    feature.replace("_", " ").title(),
                                    min_value=0,
                                    value=int(max(0, round(default))),
                                    key=f"cluster_{feature}"
                                )

                        elif feature in categorical_options:

                            options = categorical_options[feature]

                            profile[feature] = st.selectbox(
                                feature.replace("_", " ").title(),
                                options,
                                key=f"cluster_{feature}"
                            )

                        else:

                            st.error(
                                f"Unsupported clustering feature: {feature}"
                            )

                predict_cluster = st.form_submit_button(
                    "Predict Patient Segment",
                    type="primary",
                    use_container_width=True
                )

            if predict_cluster:

                if (
                    "ap_hi" in profile
                    and "ap_lo" in profile
                    and profile["ap_hi"] <= profile["ap_lo"]
                ):

                    st.error(
                        "Systolic pressure must be greater than diastolic pressure."
                    )

                elif len(profile) != len(cluster_features):

                    st.error(
                        "Could not prepare all required clustering features."
                    )

                else:

                    try:

                        cluster_input = pd.DataFrame(
                            [[profile[feature] for feature in cluster_features]],
                            columns=cluster_features
                        )

                        scaled_input = cluster_bundle["scaler"].transform(
                            cluster_input
                        )

                        cluster_id = int(
                            cluster_bundle["model"].predict(
                                scaled_input
                            )[0]
                        )

                        st.success("Patient segment identified.")

                        st.subheader(
                            f"Segment {cluster_id + 1} · "
                            f"{get_cluster_name(cluster_id)}"
                        )

                        st.write(
                            "This group was learned from patient profiles with "
                            "similar age, BMI, cholesterol, glucose, and blood "
                            "pressure values."
                        )

                        segment_data = cluster_population[
                            cluster_population["cluster"] == cluster_id
                        ]

                        if not segment_data.empty:

                            metric_cols = st.columns(3)

                            if "age" in segment_data.columns:

                                metric_cols[0].metric(
                                    "Segment age average",
                                    f"{segment_data['age'].mean():.1f} years"
                                )

                            if "bmi" in segment_data.columns:

                                metric_cols[1].metric(
                                    "Segment BMI average",
                                    f"{segment_data['bmi'].mean():.1f}"
                                )

                            if {"ap_hi", "ap_lo"}.issubset(segment_data.columns):

                                metric_cols[2].metric(
                                    "Average blood pressure",
                                    f"{segment_data['ap_hi'].mean():.0f}/"
                                    f"{segment_data['ap_lo'].mean():.0f}"
                                )

                            comparisons = (
                                ("age", "age", "older", "younger"),
                                ("bmi", "bmi", "higher BMI", "lower BMI"),
                                ("ap_hi", "ap_hi", "higher systolic pressure",
                                 "lower systolic pressure"),
                                ("ap_lo", "ap_lo", "higher diastolic pressure",
                                 "lower diastolic pressure"),
                            )
                            differences = []

                            for input_feature, data_feature, higher, lower in comparisons:

                                if (
                                    input_feature not in profile
                                    or data_feature not in segment_data.columns
                                ):
                                    continue

                                average = float(segment_data[data_feature].mean())
                                difference = float(profile[input_feature]) - average
                                tolerance = max(abs(average) * 0.05, 0.1)

                                if difference > tolerance:
                                    differences.append(higher)
                                elif difference < -tolerance:
                                    differences.append(lower)

                            if differences:
                                st.info(
                                    "Compared with this segment's average, "
                                    "your entered profile has "
                                    + ", ".join(differences)
                                    + "."
                                )
                            else:
                                st.info(
                                    "Your entered age, BMI, and pressure values "
                                    "are close to this segment's averages."
                                )

                        st.caption(
                            "Segment numbers are model labels. The group "
                            "description and averages summarize historical "
                            "records, not an individual diagnosis."
                        )

                    except Exception as error:

                        st.error(
                            f"Clustering prediction failed: {error}"
                        )

            # ---------------------------------------------
            # CLUSTER PROFILES
            # ---------------------------------------------

            st.divider()

            st.subheader("📊 Cluster Profiles")

            try:

                profile_table = (
                    cluster_population
                    .groupby("cluster")
                    .agg(
                        Average_Age=("age", "mean"),
                        Average_BMI=("bmi", "mean"),
                        Average_Systolic=("ap_hi", "mean"),
                        Average_Diastolic=("ap_lo", "mean"),
                    )
                    .reset_index()
                )

                profile_table["Segment"] = (
                    profile_table["cluster"].apply(get_cluster_name)
                )

                st.dataframe(
                    profile_table[
                        [
                            "Segment",
                            "Average_Age",
                            "Average_BMI",
                            "Average_Systolic",
                            "Average_Diastolic",
                        ]
                    ].round(2),
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as error:

                st.error(
                    f"Could not display cluster profiles: {error}"
                )

            # ---------------------------------------------
            # INTERACTIVE BMI AND BLOOD PRESSURE VIEW
            # ---------------------------------------------

            st.divider()

            st.subheader("📍 Patient Distribution by Cluster")

            st.write(
                "Each dot is one patient record. Hover over a dot to see its "
                "segment, BMI, age, and blood pressure measurements."
            )

            point_columns = [
                column
                for column in (
                    "age",
                    "bmi",
                    "ap_hi",
                    "ap_lo",
                    "cholesterol",
                    "gluc",
                    "cluster",
                )
                if column in cluster_population.columns
            ]
            chart_data = cluster_population[point_columns].dropna(
                subset=["bmi", "ap_hi", "cluster"]
            )

            if not chart_data.empty:

                sample_size = min(5000, len(chart_data))
                sample = chart_data.sample(
                    n=sample_size,
                    random_state=42
                ).copy()
                sample["Segment"] = sample["cluster"].apply(get_cluster_name)

                tooltip_fields = [
                    {
                        "field": "Segment",
                        "type": "nominal",
                        "title": "Patient segment",
                    },
                    {
                        "field": "bmi",
                        "type": "quantitative",
                        "title": "BMI",
                        "format": ".1f",
                    },
                    {
                        "field": "ap_hi",
                        "type": "quantitative",
                        "title": "Systolic pressure",
                    },
                    {
                        "field": "ap_lo",
                        "type": "quantitative",
                        "title": "Diastolic pressure",
                    },
                ]

                if "age" in sample.columns:
                    tooltip_fields.append(
                        {
                            "field": "age",
                            "type": "quantitative",
                            "title": "Age (years)",
                            "format": ".1f",
                        }
                    )

                if "cholesterol" in sample.columns:
                    tooltip_fields.append(
                        {
                            "field": "cholesterol",
                            "type": "quantitative",
                            "title": "Cholesterol code",
                        }
                    )

                if "gluc" in sample.columns:
                    tooltip_fields.append(
                        {
                            "field": "gluc",
                            "type": "quantitative",
                            "title": "Glucose code",
                        }
                    )

                st.vega_lite_chart(
                    sample,
                    {
                        "$schema": (
                            "https://vega.github.io/schema/vega-lite/v5.json"
                        ),
                        "height": 420,
                        "mark": {
                            "type": "circle",
                            "size": 75,
                            "opacity": 0.62,
                        },
                        "encoding": {
                            "x": {
                                "field": "bmi",
                                "type": "quantitative",
                                "title": "Body Mass Index (BMI)",
                            },
                            "y": {
                                "field": "ap_hi",
                                "type": "quantitative",
                                "title": "Systolic pressure (mmHg)",
                            },
                            "color": {
                                "field": "Segment",
                                "type": "nominal",
                                "title": "Patient segment",
                            },
                            "tooltip": tooltip_fields,
                        },
                    },
                    use_container_width=True,
                )

                st.caption(
                    f"Interactive view of {sample_size:,} sampled patient "
                    "records. Hover on any circle for its measurements."
                )
            else:
                st.info("There are no valid patient records to plot.")

# =========================================================
# MODEL & DATA LAB
# =========================================================

with lab_tab:

    st.header("📊 Model & Data Lab")

    st.write(
        "Explore what the models learned through performance metrics, "
        "data patterns, and the original clustering visuals."
    )

    regression_lab, classification_lab, clustering_lab = st.tabs(
        ["🛏️ Regression", "❤️ Classification", "🧩 Clustering visuals"]
    )

    with regression_lab:

        st.subheader("Length-of-stay model comparison")

        if results_data is None:

            st.info(
                "Regression comparison results are not available yet."
            )

        else:

            metric_columns = [
                column
                for column in ("MAE", "RMSE", "R2")
                if column in results_data.columns
            ]

            if "RMSE" in results_data.columns and "Model" in results_data.columns:

                rmse = pd.to_numeric(
                    results_data["RMSE"],
                    errors="coerce"
                )

                if rmse.notna().any():

                    best_index = rmse.idxmin()
                    best_row = results_data.loc[best_index]
                    st.success(
                        f"Best saved result by RMSE: **{best_row['Model']}** "
                        f"({rmse.loc[best_index]:.3f} days)"
                    )

            if metric_columns and "Model" in results_data.columns:

                chart_data = results_data[["Model"] + metric_columns].copy()
                for column in metric_columns:
                    chart_data[column] = pd.to_numeric(
                        chart_data[column],
                        errors="coerce"
                    )

                st.bar_chart(
                    chart_data.dropna(subset=metric_columns)
                    .set_index("Model")[metric_columns]
                )

            display_columns = [
                column
                for column in (
                    "Model",
                    "MAE",
                    "RMSE",
                    "R2",
                    "Best CV RMSE",
                )
                if column in results_data.columns
            ]
            st.dataframe(
                results_data[display_columns].round(3),
                use_container_width=True,
                hide_index=True
            )

        if los_data is not None and TARGET in los_data.columns:

            values = pd.to_numeric(
                los_data[TARGET],
                errors="coerce"
            ).dropna()

            if not values.empty:

                st.subheader("Length-of-stay distribution")
                fig, ax = plt.subplots(figsize=(9, 3.8))
                ax.hist(
                    values,
                    bins=24,
                    color="#087e8b",
                    edgecolor="white"
                )
                ax.set_xlabel("Length of stay (days)")
                ax.set_ylabel("Patient records")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    with classification_lab:

        st.subheader("Cardiovascular classification")
        st.write(
            "The classifier uses the fixed 0.45 cutoff from its current "
            "training script. The saved metrics are calculated on that "
            "script's held-out test split."
        )

        if classification_results is not None and not classification_results.empty:

            saved_metrics = classification_results.iloc[0]
            metric_definitions = (
                ("Accuracy", "Accuracy"),
                ("Precision", "Precision"),
                ("Recall", "Recall"),
                ("F1-score", "F1-score"),
                ("ROC-AUC", "ROC-AUC"),
            )
            metric_columns = st.columns(len(metric_definitions))

            for column, (label, field) in zip(
                metric_columns,
                metric_definitions,
            ):
                if field in classification_results.columns:
                    metric_value = pd.to_numeric(
                        saved_metrics[field],
                        errors="coerce",
                    )
                    column.metric(
                        label,
                        f"{metric_value:.1%}"
                        if pd.notna(metric_value)
                        else "N/A",
                    )

        else:
            st.write(
                "Saved classifier metrics are unavailable. Train the current "
                "classifier to create its results file."
            )

        if cardio_data is None or "cardio" not in cardio_data.columns:

            st.info(
                "Cardiovascular data is not available for these charts."
            )

        else:

            class_counts = (
                cardio_data["cardio"]
                .value_counts()
                .sort_index()
                .rename(index={0: "Class 0", 1: "Class 1"})
                .rename("Records")
            )
            metric_col, chart_col = st.columns([1, 1.5])

            with metric_col:
                st.metric("Cleaned records", f"{len(cardio_data):,}")
                if 1 in cardio_data["cardio"].values:
                    positive_share = float(cardio_data["cardio"].mean())
                    st.metric("Class 1 in dataset", f"{positive_share:.1%}")
                st.bar_chart(class_counts)

            with chart_col:
                if {"age_years", "bmi"}.issubset(cardio_data.columns):

                    chart_data = cardio_data[
                        ["age_years", "bmi", "cardio"]
                    ].dropna()

                    if not chart_data.empty:

                        sample = chart_data.sample(
                            n=min(4000, len(chart_data)),
                            random_state=42
                        )

                        st.markdown("#### Age and BMI by dataset class")
                        fig, ax = plt.subplots(figsize=(8, 4))

                        for label, color in (
                            (0, "#087e8b"),
                            (1, "#f08a4b"),
                        ):
                            subset = sample[sample["cardio"] == label]
                            if not subset.empty:
                                ax.scatter(
                                    subset["age_years"],
                                    subset["bmi"],
                                    s=12,
                                    alpha=0.45,
                                    color=color,
                                    label=f"Class {label}"
                                )

                        ax.set_xlabel("Age (years)")
                        ax.set_ylabel("BMI")
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.legend(frameon=False)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            st.caption(
                "The chart shows dataset patterns, not model accuracy. "
                "Below-threshold results must not be interpreted as good health."
            )

    with clustering_lab:

        st.subheader("Patient-cluster visual gallery")
        st.write(
            "These project plots show the discovered groups and how the "
            "feature-removal experiment changed normalized mutual information."
        )

        image_col1, image_col2 = st.columns(2)

        with image_col1:

            st.markdown("#### Patient cluster map")

            if CLUSTER_PLOT_PATH.exists():

                st.image(
                    str(CLUSTER_PLOT_PATH),
                    use_container_width=True
                )
                st.caption(
                    "The saved visualization compares unlabelled records "
                    "with the K-Means segments."
                )

            else:

                st.info("The saved cluster map is not available.")

        with image_col2:

            st.markdown("#### Clustering feature analysis")

            if IMPORTANCE_PLOT_PATH.exists():

                st.image(
                    str(IMPORTANCE_PLOT_PATH),
                    use_container_width=True
                )
                st.caption(
                    "Change in NMI after removing each feature; this is an "
                    "exploratory clustering signal, not clinical importance."
                )

            else:

                st.info("The clustering feature plot is not available.")

        if cluster_population is not None and "cluster" in cluster_population:

            st.markdown("#### Segment sizes")
            segment_sizes = (
                cluster_population["cluster"]
                .value_counts()
                .sort_index()
            )
            segment_sizes.index = [
                get_cluster_name(cluster_id)
                for cluster_id in segment_sizes.index
            ]
            st.bar_chart(segment_sizes.rename("Patient records"))

    st.divider()
    st.caption(
        "Hospital Helper · educational machine-learning project. "
        "Outputs are not diagnoses or treatment recommendations."
    )